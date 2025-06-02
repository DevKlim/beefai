# beefai/evaluation/rhythm_visualizer.py
import torch
import torch.nn.functional as F # <<<<<<<<<<<<<<<<<<<<<<< ADD THIS IMPORT
import os
import sys
import yaml
import numpy as np
import soundfile as sf
import json 
from typing import List, Dict, Optional, Tuple
import librosa

# Adjust import paths
sys.path.append(os.getcwd())

from beefai.flow_model.tokenizer import FlowTokenizer
from beefai.flow_model.transformer_model import FlowTransformerDecoder, FlowGPTConfig, get_next_context_ids_for_token
from beefai.data_processing.audio_processor import AudioProcessor 
from beefai.data_processing.beat_feature_extractor import BeatFeatureExtractor 
from beefai.utils.data_types import BeatInfo, FlowData, FlowDatum, BarBeatFeatures, SongBeatFeatures

# --- Configuration ---
DEFAULT_MODEL_CONFIG_PATH = "lite_model_training/model_config_full.yaml"
DEFAULT_TOKENIZER_CONFIG_PATH = "beefai/flow_model/flow_tokenizer_config_v2.json" 
DEFAULT_CHECKPOINT_PATH = "data/checkpoints/flow_model_full/full_final_model.pt" 
DEFAULT_INSTRUMENTAL_PATH = "data/instrumentals/HiiiPower.wav" 
DEFAULT_INSTRUMENTAL_OFFSET_SEC = None 
DEFAULT_INSTRUMENTAL_DURATION_SEC = None 

OUTPUT_DIR = "output/flow_visualizations"
FLOW_DATA_OUTPUT_DIR = "output/generated_flow_data" 
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FLOW_DATA_OUTPUT_DIR, exist_ok=True) 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_syllable_sound_event(
    duration_sec: float, 
    stress_level: int, 
    sample_rate: int = 44100
) -> np.ndarray:
    core_sound_duration_sec = 0.05 
    num_samples_core = int(core_sound_duration_sec * sample_rate)

    if num_samples_core <= 0:
        return np.array([], dtype=np.float32)

    noise = (np.random.rand(num_samples_core) * 2 - 1) * 0.5 
    t_env = np.linspace(0, 1, num_samples_core, endpoint=False)
    decay_rate_sharp = 25 
    envelope = np.exp(-t_env * decay_rate_sharp) 
    
    freq1 = 800  
    freq2 = 1200 
    t_sine = np.linspace(0, core_sound_duration_sec, num_samples_core, endpoint=False)
    tonal_component = 0.3 * np.sin(2 * np.pi * freq1 * t_sine) + \
                      0.2 * np.sin(2 * np.pi * freq2 * t_sine)
    
    percussive_sound = (noise + tonal_component * 0.5) * envelope
    
    amplitude_modifier = 1.0
    if stress_level == 1: # Primary stress
        amplitude_modifier = 1.2
    elif stress_level == 2: # Secondary stress
        amplitude_modifier = 1.1
    
    percussive_sound *= amplitude_modifier
    
    total_samples_for_event_duration = int(duration_sec * sample_rate)
    if total_samples_for_event_duration > num_samples_core:
        padding = np.zeros(total_samples_for_event_duration - num_samples_core, dtype=np.float32)
        final_sound = np.concatenate((percussive_sound, padding))
    elif total_samples_for_event_duration > 0:
        final_sound = percussive_sound[:total_samples_for_event_duration]
    else: 
        final_sound = np.array([], dtype=np.float32)

    if final_sound.size == 0 and duration_sec > 0.001: 
        final_sound = np.zeros(max(1, int(0.001 * sample_rate)), dtype=np.float32)

    return final_sound

def generate_syllable_click_track(
    flow_data: FlowData, 
    beat_info: BeatInfo, 
    tokenizer: FlowTokenizer, 
    instrumental_waveform_mono: Optional[np.ndarray] = None, 
    sample_rate: int = 44100,
) -> Tuple[np.ndarray, int]:
    if not flow_data:
        return (np.array([], dtype=np.float32), sample_rate)

    bpm = beat_info.get("bpm")
    if not bpm or bpm <=0: 
        bpm = 120.0 
    
    beats_per_bar = beat_info.get("beats_per_bar", 4)
    if beats_per_bar <=0: beats_per_bar = 4

    beat_duration_sec = 60.0 / bpm
    bar_duration_sec_from_bpm = beat_duration_sec * beats_per_bar
    subdivision_duration_sec = bar_duration_sec_from_bpm / max(1, tokenizer.max_subdivisions)

    bar_absolute_start_times_sec: Dict[int, float] = {}
    if beat_info.get("downbeat_times") and len(beat_info["downbeat_times"]) > 0 :
        max_bar_idx_flow = max((fd.get("bar_index", 0) for fd in flow_data if isinstance(fd, dict)), default=-1)
        current_downbeats = list(beat_info["downbeat_times"])
        
        if max_bar_idx_flow >= len(current_downbeats):
            last_dt = current_downbeats[-1] if current_downbeats else 0.0
            est_bar_dur = beat_info.get("estimated_bar_duration", bar_duration_sec_from_bpm)
            if est_bar_dur <= 0: est_bar_dur = bar_duration_sec_from_bpm
            
            for i in range(len(current_downbeats), max_bar_idx_flow + 5): 
                last_dt += est_bar_dur
                current_downbeats.append(last_dt)
        
        for i, dt in enumerate(current_downbeats): bar_absolute_start_times_sec[i] = dt

    elif beat_info.get("estimated_bar_duration") and beat_info["estimated_bar_duration"] > 0:
        bar_dur_est = beat_info["estimated_bar_duration"]
        max_bar_idx = max((fd.get("bar_index", 0) for fd in flow_data if isinstance(fd, dict)), default=-1)
        t0 = 0.0 
        for i in range(max_bar_idx + 5): bar_absolute_start_times_sec[i] = t0 + (i * bar_dur_est)
    else: 
        max_bar_idx = max((fd.get("bar_index", 0) for fd in flow_data if isinstance(fd, dict)), default=-1)
        for i in range(max_bar_idx + 5): bar_absolute_start_times_sec[i] = i * bar_duration_sec_from_bpm

    max_event_time_sec = 0.0
    if flow_data: 
        for fd_item in flow_data:
            if not isinstance(fd_item, dict): continue
            bar_idx = fd_item.get("bar_index")
            if bar_idx is None : continue 
            
            if bar_idx not in bar_absolute_start_times_sec:
                if bar_idx >= 0: 
                    if bar_absolute_start_times_sec: 
                        last_known_bar = max(bar_absolute_start_times_sec.keys())
                        last_known_time = bar_absolute_start_times_sec[last_known_bar]
                        est_bar_dur = beat_info.get("estimated_bar_duration", bar_duration_sec_from_bpm)
                        if est_bar_dur <= 0: est_bar_dur = bar_duration_sec_from_bpm
                        bar_absolute_start_times_sec[bar_idx] = last_known_time + (bar_idx - last_known_bar) * est_bar_dur
                    else: 
                        bar_absolute_start_times_sec[bar_idx] = bar_idx * bar_duration_sec_from_bpm 
                else: 
                    continue

            bar_start_sec_val = bar_absolute_start_times_sec[bar_idx]
            syllable_subdivisions = fd_item.get("syllable_start_subdivisions", [])
            syll_dur_bins = fd_item.get("syllable_durations_quantized", [])

            for i in range(len(syllable_subdivisions)):
                syl_start_offset_sec = syllable_subdivisions[i] * subdivision_duration_sec
                approx_syl_dur_beats = tokenizer.dequantize_syllable_duration_bin(syll_dur_bins[i] if i < len(syll_dur_bins) else 0)
                syl_dur_sec = approx_syl_dur_beats * beat_duration_sec
                max_event_time_sec = max(max_event_time_sec, bar_start_sec_val + syl_start_offset_sec + syl_dur_sec)

    min_output_duration_sec = 0.1 
    output_waveform_base: np.ndarray
    if instrumental_waveform_mono is not None and instrumental_waveform_mono.size > 0:
        if instrumental_waveform_mono.ndim > 1:
            instrumental_waveform_mono = librosa.to_mono(instrumental_waveform_mono)
        output_duration_sec = max(max_event_time_sec, len(instrumental_waveform_mono) / sample_rate, min_output_duration_sec)
        output_waveform_base = np.copy(instrumental_waveform_mono)
        required_samples = int(output_duration_sec * sample_rate)
        if len(output_waveform_base) < required_samples:
            output_waveform_base = np.concatenate((output_waveform_base, np.zeros(required_samples - len(output_waveform_base), dtype=np.float32)))
        elif len(output_waveform_base) > required_samples: 
            output_waveform_base = output_waveform_base[:required_samples]
    else:
        output_duration_sec = max(max_event_time_sec, min_output_duration_sec)
        output_waveform_base = np.zeros(max(1, int(output_duration_sec * sample_rate)), dtype=np.float32)

    num_syllables_placed = 0
    if flow_data: 
        for fd_item in flow_data:
            if not isinstance(fd_item, dict): continue
            bar_idx = fd_item.get("bar_index")
            if bar_idx is None or bar_idx not in bar_absolute_start_times_sec: continue
            
            bar_start_sec_val = bar_absolute_start_times_sec[bar_idx]
            syllable_starts = fd_item.get("syllable_start_subdivisions", [])
            syllable_dur_bins = fd_item.get("syllable_durations_quantized", [])
            syllable_stresses = fd_item.get("syllable_stresses", [])

            for i in range(len(syllable_starts)):
                syl_onset_subdiv = syllable_starts[i]
                syl_dur_bin = syllable_dur_bins[i] if i < len(syllable_dur_bins) else 0
                syl_stress = syllable_stresses[i] if i < len(syllable_stresses) else 0

                syl_onset_in_bar_sec = syl_onset_subdiv * subdivision_duration_sec
                abs_syl_onset_sec = bar_start_sec_val + syl_onset_in_bar_sec
                
                syl_dur_beats_approx = tokenizer.dequantize_syllable_duration_bin(syl_dur_bin)
                syl_actual_duration_sec = max(0.01, syl_dur_beats_approx * beat_duration_sec) 

                syllable_sound = generate_syllable_sound_event(syl_actual_duration_sec, syl_stress, sample_rate)
                
                start_sample = int(abs_syl_onset_sec * sample_rate)
                end_sample = start_sample + len(syllable_sound)

                if output_waveform_base.size > 0 and end_sample <= len(output_waveform_base) and len(syllable_sound)>0:
                    output_waveform_base[start_sample:end_sample] += syllable_sound
                    num_syllables_placed +=1
                elif output_waveform_base.size > 0 and start_sample < len(output_waveform_base) and len(syllable_sound)>0: 
                    fit_len = min(len(syllable_sound), len(output_waveform_base) - start_sample)
                    output_waveform_base[start_sample : start_sample+fit_len] += syllable_sound[:fit_len]
                    num_syllables_placed +=1
        
        if num_syllables_placed > 0:
            print(f"Placed {num_syllables_placed} percussive syllable sounds in the audio.")
            
    return (output_waveform_base, sample_rate)

def _create_beat_info_from_custom_features(song_beat_features: SongBeatFeatures) -> BeatInfo:
    if not song_beat_features:
        return { 
            "bpm": 120.0, "beats_per_bar": 4, 
            "estimated_bar_duration": 2.0, "downbeat_times": [i * 2.0 for i in range(16)], 
            "beat_times": [] 
        }

    first_bar = song_beat_features[0]
    bpm = first_bar.get("bpm", 120.0)
    time_sig_num, time_sig_den = first_bar.get("time_signature", (4, 4))
    beats_per_bar = time_sig_num

    if bpm <= 0: bpm = 120.0 
    if beats_per_bar <= 0: beats_per_bar = 4

    beat_duration_sec = 60.0 / bpm
    default_estimated_bar_duration_sec = beat_duration_sec * beats_per_bar

    num_bars = len(song_beat_features)
    downbeat_times = []
    current_time = 0.0
    max_defined_bar_idx = -1
    if num_bars > 0:
        valid_bar_indices = [bf.get("bar_index") for bf in song_beat_features if bf.get("bar_index") is not None]
        if valid_bar_indices:
            max_defined_bar_idx = max(valid_bar_indices)
        else: 
            max_defined_bar_idx = num_bars -1 

    total_bars_to_map = max(num_bars, max_defined_bar_idx + 1) + 4 

    for i in range(total_bars_to_map) : 
        bar_feature_for_this_idx = next((bf for bf in song_beat_features if bf.get("bar_index") == i), None)
        
        if bar_feature_for_this_idx and bar_feature_for_this_idx.get("bar_start_time_sec") is not None:
            current_time = bar_feature_for_this_idx["bar_start_time_sec"]
            downbeat_times.append(current_time)
            bar_dur = bar_feature_for_this_idx.get("bar_duration_sec", default_estimated_bar_duration_sec)
            if i == num_bars - 1 or i >= max_defined_bar_idx : 
                 current_time += bar_dur 
        elif i > 0 and downbeat_times: 
            prev_bar_feature = next((bf for bf in song_beat_features if bf.get("bar_index") == i-1), None)
            if prev_bar_feature and prev_bar_feature.get("bar_duration_sec") is not None:
                current_time = downbeat_times[-1] + prev_bar_feature.get("bar_duration_sec")
            else: 
                current_time = downbeat_times[-1] + default_estimated_bar_duration_sec
            downbeat_times.append(current_time)
        else: 
            downbeat_times.append(current_time) 
            current_time += default_estimated_bar_duration_sec

    all_beat_times = []
    for bar_idx_map in range(len(downbeat_times)): 
        bar_feat_match = next((bf for bf in song_beat_features if bf.get("bar_index") == bar_idx_map), None)
        
        bar_start_time = downbeat_times[bar_idx_map]
        current_bar_bpm = bpm
        current_bar_time_sig_num = beats_per_bar
        
        if bar_feat_match:
            current_bar_bpm = bar_feat_match.get("bpm", bpm)
            current_bar_time_sig_num = bar_feat_match.get("time_signature", (beats_per_bar, 4))[0]
        
        if current_bar_bpm <=0: current_bar_bpm = bpm
        if current_bar_time_sig_num <=0: current_bar_time_sig_num = beats_per_bar
        current_beat_duration_sec = 60.0 / current_bar_bpm

        for beat_in_bar in range(current_bar_time_sig_num):
            all_beat_times.append(bar_start_time + beat_in_bar * current_beat_duration_sec)

    return {
        "bpm": bpm, 
        "beat_times": all_beat_times,
        "downbeat_times": downbeat_times,
        "estimated_bar_duration": default_estimated_bar_duration_sec, 
        "beats_per_bar": beats_per_bar 
    }

def _generate_flow_core(
    prompt_bar_features: SongBeatFeatures,
    all_song_beat_features: SongBeatFeatures, 
    model_checkpoint_path: str,
    model_config_path: str,
    tokenizer_config_path: str,
    context_bar_idx_start: int, 
    output_name_suffix: str,
    force_offset_on_beat_one_bias: Optional[float] = None, # New parameter
    energy_bias_params: Optional[Dict[str, float]] = None # New param for energy {syl_count_bias: val, short_syl_dur_bias: val}
) -> Tuple[Optional[FlowData], Optional[FlowTokenizer], Optional[FlowGPTConfig]]:
    
    if not os.path.exists(tokenizer_config_path):
        print(f"Error: Tokenizer config not found: {tokenizer_config_path}. Exiting."); return None, None, None
    
    tokenizer = FlowTokenizer(config_path=tokenizer_config_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.pad_token_id

    if not os.path.exists(model_config_path):
        print(f"Error: Model config not found: {model_config_path}. Exiting."); return None, tokenizer, None
    model_yaml_config = load_yaml_config(model_config_path)
    
    gpt_config = FlowGPTConfig(
        vocab_size=vocab_size, block_size=model_yaml_config["block_size"],
        n_layer=model_yaml_config["n_layer"], n_head=model_yaml_config["n_head"],
        n_embd=model_yaml_config["n_embd"], max_segment_types=model_yaml_config["max_segment_types"],
        max_intra_line_positions=model_yaml_config["max_intra_line_positions"],
        dropout=model_yaml_config["dropout"], bias=model_yaml_config.get("bias", True),
        pad_token_id=pad_token_id
    )
    model = FlowTransformerDecoder(gpt_config)
    
    if not os.path.exists(model_checkpoint_path):
        print(f"Error: Model checkpoint not found: {model_checkpoint_path}. Exiting."); return None, tokenizer, gpt_config
        
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location=DEVICE, weights_only=False) 
        state_dict_key = next((k for k in ['model_state_dict', 'state_dict', 'model'] if k in checkpoint and isinstance(checkpoint[k], dict)), None)
        
        if state_dict_key: state_dict = checkpoint[state_dict_key]
        elif isinstance(checkpoint, dict) and 'config' in checkpoint and 'model_state_dict' not in checkpoint:
            if any(key.startswith('transformer.') or key.startswith('lm_head.') for key in checkpoint.keys()): state_dict = checkpoint
            else: raise ValueError("Checkpoint is a dict but doesn't look like a state_dict or a structured checkpoint.")
        elif isinstance(checkpoint, dict) : state_dict = checkpoint
        else: raise ValueError(f"Checkpoint is not a dictionary or supported structure: type {type(checkpoint)}")
            
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()): 
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model checkpoint {model_checkpoint_path}: {e}. Exiting."); return None, tokenizer, gpt_config
        
    model.to(DEVICE); model.eval()
    print(f"Model loaded from {model_checkpoint_path} and set to eval mode.")

    current_token_ids_list = [tokenizer.bos_token_id]
    current_segment_ids_list = [0] 
    current_intra_pos_ids_list = [0]
    
    if not prompt_bar_features:
        print("Error: No prompt_bar_features provided. Cannot start generation."); return None, tokenizer, gpt_config

    for bar_feat in prompt_bar_features:
        bar_tokens = tokenizer.encode_bar_features(bar_feat)
        for tok_id in bar_tokens:
            seg_id, intra_id = get_next_context_ids_for_token(current_token_ids_list, tok_id, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
            current_token_ids_list.append(tok_id); current_segment_ids_list.append(seg_id); current_intra_pos_ids_list.append(intra_id)
        
        sep_tok = tokenizer.sep_input_flow_token_id
        seg_id, intra_id = get_next_context_ids_for_token(current_token_ids_list, sep_tok, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
        current_token_ids_list.append(sep_tok); current_segment_ids_list.append(seg_id); current_intra_pos_ids_list.append(intra_id)

    line_start_tok = tokenizer.line_start_token_id
    seg_id, intra_id = get_next_context_ids_for_token(current_token_ids_list, line_start_tok, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
    current_token_ids_list.append(line_start_tok); current_segment_ids_list.append(seg_id); current_intra_pos_ids_list.append(intra_id)

    max_song_bars = len(all_song_beat_features) if all_song_beat_features else context_bar_idx_start + 16 
    avg_tokens_per_bar_features = 25 
    avg_lines_per_bar = 2 
    avg_tokens_per_flow_line = 20 
    max_total_tokens_safety_net = gpt_config.block_size + (avg_tokens_per_bar_features + avg_lines_per_bar * avg_tokens_per_flow_line) * (max_song_bars + 5)

    generated_flow_tokens_for_song: List[int] = [] 
    current_bar_being_generated_flow_for = context_bar_idx_start 
    
    full_song_context_token_ids = list(current_token_ids_list) 
    full_song_context_segment_ids = list(current_segment_ids_list)
    full_song_context_intra_pos_ids = list(current_intra_pos_ids_list)
    generated_flow_tokens_for_song.append(tokenizer.line_start_token_id)

    # State for logit biasing
    # Phase: None, "EXPECTING_SYLLABLES_COUNT", "EXPECTING_OFFSET_BIN", "EXPECTING_DURATION_BIN",
    #        "EXPECTING_SYLLABLE_START", "EXPECTING_SYLLABLE_DURATION", "EXPECTING_SYLLABLE_STRESS"
    current_gen_phase = "EXPECTING_SYLLABLES_COUNT" # Since prompt ends with LINE_START


    print(f"Generating flow for song, starting from bar {current_bar_being_generated_flow_for}, up to ~{max_song_bars} bars or {max_total_tokens_safety_net} tokens...")

    for step_count in range(max_total_tokens_safety_net - len(full_song_context_token_ids)):
        idx_cond = torch.tensor([full_song_context_token_ids[-gpt_config.block_size:]], dtype=torch.long, device=DEVICE)
        seg_ids_cond = torch.tensor([full_song_context_segment_ids[-gpt_config.block_size:]], dtype=torch.long, device=DEVICE)
        intra_pos_ids_cond = torch.tensor([full_song_context_intra_pos_ids[-gpt_config.block_size:]], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            logits, _ = model(idx_cond, segment_ids=seg_ids_cond, intra_line_pos_ids=intra_pos_ids_cond)
        
        # Logit manipulation BEFORE temperature and softmax
        if force_offset_on_beat_one_bias is not None and current_gen_phase == "EXPECTING_OFFSET_BIN":
            offset_bin_0_id = tokenizer.token_to_id.get("[OFFSET_BIN_0]")
            if offset_bin_0_id is not None:
                # print(f"DEBUG: Biasing [OFFSET_BIN_0] (ID: {offset_bin_0_id}) with {force_offset_on_beat_one_bias}")
                logits[0, -1, offset_bin_0_id] += force_offset_on_beat_one_bias
        
        if energy_bias_params:
            if current_gen_phase == "EXPECTING_SYLLABLES_COUNT" and "syl_count_bias" in energy_bias_params:
                # Bias towards higher syllable counts (e.g., 12 to max_syllables)
                for i in range(max(0,tokenizer.max_syllables // 2), tokenizer.max_syllables + 1): 
                    syl_token_id = tokenizer.token_to_id.get(f"[SYLLABLES_{i}]")
                    if syl_token_id is not None:
                        logits[0, -1, syl_token_id] += energy_bias_params["syl_count_bias"]
            
            if current_gen_phase == "EXPECTING_SYLLABLE_DURATION" and "short_syl_dur_bias" in energy_bias_params:
                # Bias towards shorter per-syllable durations (e.g., first third of bins)
                for i in range(tokenizer.num_syllable_duration_bins // 3): 
                    syl_dur_id = tokenizer.token_to_id.get(f"[SYLLABLE_DURATION_BIN_{i}]")
                    if syl_dur_id is not None:
                        logits[0, -1, syl_dur_id] += energy_bias_params["short_syl_dur_bias"]


        logits = logits[:, -1, :] / 0.7 
        
        # Corrected top_k application
        current_top_k = 50 # Make top_k a variable
        if current_top_k is not None and current_top_k > 0: # Check if it's a valid number
            v, _ = torch.topk(logits, min(current_top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        next_token_str = tokenizer.id_to_token.get(next_token_id, "[UNK_ID]")

        # Update generation phase based on the token *just generated*
        if next_token_str.startswith("[SYLLABLES_"): current_gen_phase = "EXPECTING_OFFSET_BIN"
        elif next_token_str.startswith("[OFFSET_BIN_"): current_gen_phase = "EXPECTING_DURATION_BIN"
        elif next_token_str.startswith("[DURATION_BIN_"): current_gen_phase = "EXPECTING_SYLLABLE_START"
        elif next_token_str.startswith("[SYLLABLE_STARTS_SUBDIV_"): current_gen_phase = "EXPECTING_SYLLABLE_DURATION"
        elif next_token_str.startswith("[SYLLABLE_DURATION_BIN_"): current_gen_phase = "EXPECTING_SYLLABLE_STRESS"
        elif next_token_str.startswith("[SYLLABLE_STRESS_"): current_gen_phase = "EXPECTING_SYLLABLE_START" # Ready for next triplet or END
        elif next_token_id == tokenizer.end_syllable_sequence_token_id: current_gen_phase = "EXPECTING_LINE_START_OR_BAR_START" # Line ended
        elif next_token_id == tokenizer.line_start_token_id: current_gen_phase = "EXPECTING_SYLLABLES_COUNT"
        elif next_token_id == tokenizer.bar_start_token_id: current_gen_phase = "EXPECTING_BEAT_FEATURES_THEN_LINE_START" # (This phase is handled by guidance logic below)


        next_seg_id, next_intra_pos_id = get_next_context_ids_for_token(
            full_song_context_token_ids, next_token_id, tokenizer, 
            gpt_config.max_segment_types, gpt_config.max_intra_line_positions
        )
        full_song_context_token_ids.append(next_token_id)
        full_song_context_segment_ids.append(next_seg_id)
        full_song_context_intra_pos_ids.append(next_intra_pos_id)
        generated_flow_tokens_for_song.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            print("  [EOS] token generated. Stopping generation.")
            break
        
        if next_token_id == tokenizer.bar_start_token_id:
            current_bar_being_generated_flow_for += 1 
            if current_bar_being_generated_flow_for >= max_song_bars:
                print(f"  Reached max song bars ({max_song_bars}). Stopping generation.")
                if generated_flow_tokens_for_song[-1] == tokenizer.bar_start_token_id: generated_flow_tokens_for_song.pop()
                if generated_flow_tokens_for_song[-1] != tokenizer.eos_token_id: generated_flow_tokens_for_song.append(tokenizer.eos_token_id)
                break
            
            next_bar_features = next((bf for bf in all_song_beat_features if bf.get("bar_index") == current_bar_being_generated_flow_for), None)
            if not next_bar_features:
                print(f"  No beat features found for bar {current_bar_being_generated_flow_for}. Stopping.")
                if generated_flow_tokens_for_song[-1] != tokenizer.eos_token_id: generated_flow_tokens_for_song.append(tokenizer.eos_token_id)
                break
            
            bar_tokens_for_new_bar_guidance = tokenizer.encode_bar_features(next_bar_features)[1:] 
            for tok_id in bar_tokens_for_new_bar_guidance:
                seg_id, intra_id = get_next_context_ids_for_token(full_song_context_token_ids, tok_id, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
                full_song_context_token_ids.append(tok_id); full_song_context_segment_ids.append(seg_id); full_song_context_intra_pos_ids.append(intra_id)
            
            sep_tok_guidance = tokenizer.sep_input_flow_token_id
            seg_id, intra_id = get_next_context_ids_for_token(full_song_context_token_ids, sep_tok_guidance, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
            full_song_context_token_ids.append(sep_tok_guidance); full_song_context_segment_ids.append(seg_id); full_song_context_intra_pos_ids.append(intra_id)
            
            line_start_tok_guidance = tokenizer.line_start_token_id
            seg_id, intra_id = get_next_context_ids_for_token(full_song_context_token_ids, line_start_tok_guidance, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
            full_song_context_token_ids.append(line_start_tok_guidance); full_song_context_segment_ids.append(seg_id); full_song_context_intra_pos_ids.append(intra_id)
            
            generated_flow_tokens_for_song.append(line_start_tok_guidance)
            current_gen_phase = "EXPECTING_SYLLABLES_COUNT" # Update phase after guidance


    decoded_flow_data_full_song: FlowData = []
    idx_in_gen_flow_tokens = 0
    decoding_bar_idx = context_bar_idx_start 
    decoding_line_idx_in_bar = 0

    while idx_in_gen_flow_tokens < len(generated_flow_tokens_for_song):
        current_token_id = generated_flow_tokens_for_song[idx_in_gen_flow_tokens]

        if current_token_id == tokenizer.eos_token_id: break
        
        if current_token_id == tokenizer.bar_start_token_id:
            decoding_bar_idx +=1
            decoding_line_idx_in_bar = 0
            idx_in_gen_flow_tokens += 1 
            if idx_in_gen_flow_tokens < len(generated_flow_tokens_for_song) and \
               generated_flow_tokens_for_song[idx_in_gen_flow_tokens] == tokenizer.line_start_token_id:
                current_token_id = generated_flow_tokens_for_song[idx_in_gen_flow_tokens]
            else:
                continue 

        if current_token_id != tokenizer.line_start_token_id:
            idx_in_gen_flow_tokens += 1
            continue

        tokens_for_this_line_attempt = []
        temp_line_parser_idx = idx_in_gen_flow_tokens 
        
        while temp_line_parser_idx < len(generated_flow_tokens_for_song):
            token_id = generated_flow_tokens_for_song[temp_line_parser_idx]
            tokens_for_this_line_attempt.append(token_id)
            if token_id == tokenizer.end_syllable_sequence_token_id:
                break
            if token_id in [tokenizer.eos_token_id, tokenizer.bar_start_token_id] and \
               len(tokens_for_this_line_attempt) > 1: 
                tokens_for_this_line_attempt.pop() 
                temp_line_parser_idx-=1 
                break
            temp_line_parser_idx += 1
        
        datum = tokenizer.decode_flow_tokens_to_datum(tokens_for_this_line_attempt, decoding_bar_idx, decoding_line_idx_in_bar)
        if datum:
            decoded_flow_data_full_song.append(datum)
            decoding_line_idx_in_bar += 1
        
        idx_in_gen_flow_tokens = temp_line_parser_idx + 1

    print(f"DEBUG: Total FlowDatum objects decoded: {len(decoded_flow_data_full_song)}")
    return decoded_flow_data_full_song, tokenizer, gpt_config


def save_flow_data_to_json(flow_data: FlowData, beat_info: BeatInfo, output_path: str):
    if not flow_data:
        print("No flow data to save.")
        return

    bpm = beat_info.get("bpm", 120.0)
    output_data_dict = { # Changed to dict for direct JSON dump
        "beat_info": {
            "bpm": bpm,
            "beats_per_bar": beat_info.get("beats_per_bar", 4),
            "estimated_bar_duration_sec": beat_info.get("estimated_bar_duration", (60.0/bpm)*4)
        },
        "flow_lines": []
    }
    for fd in flow_data:
        line_data = dict(fd) 
        line_data["approx_line_duration_sec"] = round(fd["duration_beats"] * (60.0/bpm), 2)
        output_data_dict["flow_lines"].append(line_data) # Use append for list
        
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data_dict, f, indent=2) # Dump the dictionary
        print(f"Generated flow data saved to: {output_path}")
    except Exception as e:
        print(f"Error saving flow data to JSON: {e}")


def visualize_flow_rhythm(
    model_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    model_config_path: str = DEFAULT_MODEL_CONFIG_PATH,
    tokenizer_config_path: str = DEFAULT_TOKENIZER_CONFIG_PATH,
    instrumental_audio_path: str = DEFAULT_INSTRUMENTAL_PATH,
    instrumental_offset_sec: Optional[float] = None, 
    instrumental_duration_sec: Optional[float] = None,
    num_prompt_bars: int = 2, 
    output_filename_prefix: str = "flow_vis_audio",
    output_format: str = "wav",
    # New parameters for controlling generation biases
    force_offset_on_beat_one_bias_strength: Optional[float] = None, # e.g., 2.0
    energy_syl_count_bias_strength: Optional[float] = None,      # e.g., 1.0
    energy_short_syl_dur_bias_strength: Optional[float] = None   # e.g., 0.5
):
    print(f"--- Visualizing Flow Rhythm (Full Song Generation) ---")
    print(f"Using device: {DEVICE}")
    print(f"Model checkpoint: {model_checkpoint_path}")
    print(f"Instrumental: {instrumental_audio_path}" + 
          (f" (Offset: {instrumental_offset_sec}s" if instrumental_offset_sec is not None else "") +
          (f", Duration: {instrumental_duration_sec}s" if instrumental_duration_sec is not None else "") + ")")
    if force_offset_on_beat_one_bias_strength:
        print(f"Applying bias for [OFFSET_BIN_0]: {force_offset_on_beat_one_bias_strength}")
    if energy_syl_count_bias_strength or energy_short_syl_dur_bias_strength:
        print(f"Applying energy biases: SylCount {energy_syl_count_bias_strength}, ShortSylDur {energy_short_syl_dur_bias_strength}")


    audio_processor = AudioProcessor()
    beat_feature_extractor = BeatFeatureExtractor() 

    if not os.path.exists(instrumental_audio_path):
        print(f"Error: Instrumental audio file not found: {instrumental_audio_path}. Exiting."); return
    
    instrumental_sr = beat_feature_extractor.sample_rate 
    
    instrumental_waveform_mono, loaded_sr = audio_processor.load_audio(
        instrumental_audio_path, 
        target_sr=instrumental_sr, mono=True,
        offset_sec=instrumental_offset_sec,
        duration_sec=instrumental_duration_sec
    )
    if instrumental_waveform_mono is None or instrumental_waveform_mono.size == 0:
        print(f"Error: Could not load instrumental segment as mono: {instrumental_audio_path}. Exiting."); return
    
    beat_info_for_clicktrack: BeatInfo = audio_processor.get_beat_info(instrumental_waveform_mono, instrumental_sr)
    if not beat_info_for_clicktrack or beat_info_for_clicktrack.get("bpm", 0) <=0:
        full_wf_mono, _ = audio_processor.load_audio(instrumental_audio_path, target_sr=instrumental_sr, mono=True)
        full_bi = audio_processor.get_beat_info(full_wf_mono, instrumental_sr) if full_wf_mono is not None else None
        if full_bi and full_bi.get("bpm",0)>0:
            beat_info_for_clicktrack["bpm"] = full_bi["bpm"] 
            beat_info_for_clicktrack["beats_per_bar"] = full_bi.get("beats_per_bar",4)
            beat_info_for_clicktrack["estimated_bar_duration"] = full_bi.get("estimated_bar_duration", (60.0/full_bi["bpm"])*4)
        else:
            print(f"Error: Could not extract valid BeatInfo for click track timing from: {instrumental_audio_path}. Exiting."); return
    
    all_beat_features_for_segment: SongBeatFeatures = beat_feature_extractor.extract_features_for_song(
        audio_path=instrumental_audio_path, 
        stems_input_dir=None, 
        audio_offset_sec=instrumental_offset_sec, 
        audio_duration_sec=instrumental_duration_sec 
    )
    if not all_beat_features_for_segment:
        print(f"Error: No beat features extracted. Cannot create prompt. Exiting."); return
    
    num_prompt_bars = min(num_prompt_bars, len(all_beat_features_for_segment))
    if num_prompt_bars == 0 and len(all_beat_features_for_segment) > 0: 
        num_prompt_bars = 1
    elif num_prompt_bars == 0 and len(all_beat_features_for_segment) == 0:
         print("Error: Zero bars for prompt and no beat features. Exiting."); return
    
    prompt_bar_features_for_gen = all_beat_features_for_segment[:num_prompt_bars]
    context_bar_idx_start_for_gen = prompt_bar_features_for_gen[-1]["bar_index"] + 1 if prompt_bar_features_for_gen else 0

    energy_bias_params_dict: Optional[Dict[str, float]] = None
    if energy_syl_count_bias_strength is not None or energy_short_syl_dur_bias_strength is not None:
        energy_bias_params_dict = {}
        if energy_syl_count_bias_strength is not None:
            energy_bias_params_dict["syl_count_bias"] = energy_syl_count_bias_strength
        if energy_short_syl_dur_bias_strength is not None:
            energy_bias_params_dict["short_syl_dur_bias"] = energy_short_syl_dur_bias_strength


    decoded_flow_data, tokenizer_instance, _ = _generate_flow_core(
        prompt_bar_features=prompt_bar_features_for_gen,
        all_song_beat_features=all_beat_features_for_segment, 
        model_checkpoint_path=model_checkpoint_path,
        model_config_path=model_config_path,
        tokenizer_config_path=tokenizer_config_path,
        context_bar_idx_start=context_bar_idx_start_for_gen,
        output_name_suffix=f"from_{os.path.splitext(os.path.basename(instrumental_audio_path))[0]}",
        force_offset_on_beat_one_bias=force_offset_on_beat_one_bias_strength,
        energy_bias_params=energy_bias_params_dict
    )

    if not tokenizer_instance: 
        print("Tokenizer loading failed. Cannot proceed."); return
    
    print("\nDecoded FlowData (Full Song Generation attempt):")
    if decoded_flow_data:
        for i, fd in enumerate(decoded_flow_data):
            syllables_from_token = fd.get('syllables', 'N/A') 
            actual_syl_events = len(fd.get('syllable_start_subdivisions', []))
            print(f"  Line {i+1}: Bar {fd['bar_index']}, LineInBar {fd['line_index_in_bar']}, Syls(token): {syllables_from_token}, Syls(actual): {actual_syl_events}, "
                  f"Offset {fd['start_offset_beats']:.2f}b, Dur {fd['duration_beats']:.2f}b")

        safe_instrumental_name_json = "".join(c if c.isalnum() else '_' for c in os.path.splitext(os.path.basename(instrumental_audio_path))[0])
        offset_suffix_json = f"_offset{instrumental_offset_sec:.0f}s" if instrumental_offset_sec is not None else ""
        duration_suffix_json = f"_dur{instrumental_duration_sec:.0f}s" if instrumental_duration_sec is not None else ""
        flow_json_filename = f"generated_flow_{safe_instrumental_name_json}{offset_suffix_json}{duration_suffix_json}.json"
        flow_json_output_path = os.path.join(FLOW_DATA_OUTPUT_DIR, flow_json_filename)
        save_flow_data_to_json(decoded_flow_data, beat_info_for_clicktrack, flow_json_output_path)
    else:
        print("  No valid flow data decoded for the song.")

    click_track_audio, click_sr = generate_syllable_click_track(
        flow_data=decoded_flow_data if decoded_flow_data else [], 
        beat_info=beat_info_for_clicktrack, 
        tokenizer=tokenizer_instance, 
        instrumental_waveform_mono=instrumental_waveform_mono,
        sample_rate=instrumental_sr 
    )
    
    if click_track_audio.size > 0:
        safe_checkpoint_name = "".join(c if c.isalnum() else '_' for c in os.path.basename(model_checkpoint_path).replace('.pt',''))
        safe_instrumental_name = "".join(c if c.isalnum() else '_' for c in os.path.splitext(os.path.basename(instrumental_audio_path))[0])
        offset_suffix = f"_offset{instrumental_offset_sec:.0f}s" if instrumental_offset_sec is not None else ""
        duration_suffix = f"_dur{instrumental_duration_sec:.0f}s" if instrumental_duration_sec is not None else ""
        
        output_audio_filename = f"{output_filename_prefix}_{safe_checkpoint_name}_on_{safe_instrumental_name}{offset_suffix}{duration_suffix}_FULLSONG_prompt{num_prompt_bars}.{output_format}"
        output_audio_path = os.path.join(OUTPUT_DIR, output_audio_filename)
        
        try:
            sf.write(output_audio_path, click_track_audio, click_sr)
            print(f"\nFlow rhythm visualization audio saved to: {output_audio_path}")
        except Exception as e_sf:
            print(f"Error saving audio with soundfile: {e_sf}")
    elif decoded_flow_data: 
        print("\nFlow data was decoded, but the generated click track audio is empty.")
    else: 
        print("\nNo flow data was decoded, and no visualization audio generated.")


if __name__ == "__main__":
    print("---BeefAI Flow Visualizer---")
    
    main_checkpoint_path = DEFAULT_CHECKPOINT_PATH
    main_model_config = DEFAULT_MODEL_CONFIG_PATH
    main_tokenizer_config = DEFAULT_TOKENIZER_CONFIG_PATH
    
    main_instrumental_audio = DEFAULT_INSTRUMENTAL_PATH
    main_instrumental_offset = DEFAULT_INSTRUMENTAL_OFFSET_SEC 
    main_instrumental_duration = DEFAULT_INSTRUMENTAL_DURATION_SEC

    # --- Parameters for controlling generation biases ---
    # Set to a float (e.g., 1.0 to 3.0) to enable, or None to disable
    param_force_offset_on_beat_one = 2.0  # Example: strong bias for beat 1
    param_energy_syl_count = 1.0         # Example: bias for more syllables
    param_energy_short_syl_dur = 0.75    # Example: bias for shorter per-syllable durations

    # To disable a specific bias, set its strength to None:
    # param_force_offset_on_beat_one = None
    # param_energy_syl_count = None
    # param_energy_short_syl_dur = None
    # --- End Bias Parameters ---

    if not os.path.exists(main_checkpoint_path):
        print(f"ERROR: Default model checkpoint '{main_checkpoint_path}' not found.")
    elif not os.path.exists(main_model_config):
        print(f"ERROR: Default model config '{main_model_config}' not found.")
    elif not os.path.exists(main_tokenizer_config):
        print(f"ERROR: Default tokenizer config '{main_tokenizer_config}' not found.")
    elif not os.path.exists(main_instrumental_audio):
         print(f"ERROR: Instrumental audio file '{main_instrumental_audio}' not found.")
    else:
        visualize_flow_rhythm(
            model_checkpoint_path=main_checkpoint_path,
            model_config_path=main_model_config,
            tokenizer_config_path=main_tokenizer_config,
            instrumental_audio_path=main_instrumental_audio,
            instrumental_offset_sec=main_instrumental_offset,
            instrumental_duration_sec=main_instrumental_duration,
            output_format="wav", 
            num_prompt_bars=2,
            force_offset_on_beat_one_bias_strength=param_force_offset_on_beat_one,
            energy_syl_count_bias_strength=param_energy_syl_count,
            energy_short_syl_dur_bias_strength=param_energy_short_syl_dur
        )
    print("-" * 50)