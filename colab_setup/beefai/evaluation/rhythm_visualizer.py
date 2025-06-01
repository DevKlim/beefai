import torch
import os
import sys
import yaml
import numpy as np
import soundfile as sf
from typing import List, Dict, Optional, Tuple

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
DEFAULT_INSTRUMENTAL_PATH = "data/instrumentals/Alright.mp3" 
OUTPUT_DIR = "output/flow_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_syllable_sound_event(
    duration_sec: float, 
    stress_level: int, # 0: none, 1: primary, 2: secondary
    sample_rate: int = 44100
) -> np.ndarray:
    """Generates a sound for a single syllable event with varying characteristics."""
    num_samples = int(duration_sec * sample_rate)
    if num_samples <= 0:
        return np.array([], dtype=np.float32)

    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    
    base_freq = 440.0 # A4
    amplitude = 0.3

    if stress_level == 1: # Primary stress
        sound_freq = base_freq * 1.5 # Higher pitch
        amplitude *= 1.5
    elif stress_level == 2: # Secondary stress
        sound_freq = base_freq * 1.2 # Slightly higher pitch
        amplitude *= 1.2
    else: # Unstressed
        sound_freq = base_freq
    
    # Simple sine wave for the sound
    syllable_sound = amplitude * np.sin(2 * np.pi * sound_freq * t)
    
    # Apply a short fade in/out to avoid clicks, proportional to duration
    fade_duration_samples = min(num_samples // 8, int(0.01 * sample_rate)) # Max 10ms fade or 1/8th of sound
    if fade_duration_samples > 1:
        fade_in_env = np.linspace(0, 1, fade_duration_samples)
        fade_out_env = np.linspace(1, 0, fade_duration_samples)
        syllable_sound[:fade_duration_samples] *= fade_in_env
        syllable_sound[-fade_duration_samples:] *= fade_out_env
        
    return syllable_sound


def generate_syllable_click_track(
    flow_data: FlowData, 
    beat_info: BeatInfo, 
    tokenizer: FlowTokenizer, 
    instrumental_waveform: Optional[np.ndarray] = None, 
    sample_rate: int = 44100,
) -> Tuple[np.ndarray, int]:
    """
    Generates an audio track with sounds at each syllable's start time,
    representing duration and stress.
    """
    if not flow_data:
        print("No flow data provided to generate click track.")
        return (np.array([], dtype=np.float32), sample_rate)

    bpm = beat_info.get("bpm")
    if not bpm or bpm <=0: 
        print(f"Error: BPM not found or invalid ({bpm}). Assuming 120 BPM for visualization.")
        bpm = 120.0 
    
    beats_per_bar = beat_info.get("beats_per_bar", 4)
    if beats_per_bar <=0: beats_per_bar = 4

    beat_duration_sec = 60.0 / bpm
    bar_duration_sec_from_bpm = beat_duration_sec * beats_per_bar
    subdivision_duration_sec = bar_duration_sec_from_bpm / tokenizer.max_subdivisions

    bar_absolute_start_times_sec: Dict[int, float] = {}
    if beat_info.get("downbeat_times") and len(beat_info["downbeat_times"]) > 0 :
        for i, dt in enumerate(beat_info["downbeat_times"]): bar_absolute_start_times_sec[i] = dt
    elif beat_info.get("estimated_bar_duration") and beat_info["estimated_bar_duration"] > 0:
        bar_dur_est = beat_info["estimated_bar_duration"]
        max_bar_idx = max(fd.get("bar_index", 0) for fd in flow_data if isinstance(fd, dict)) if flow_data else 0
        t0 = beat_info.get("beat_times",[0.0])[0] if beat_info.get("beat_times") else 0.0
        for i in range(max_bar_idx + 4): bar_absolute_start_times_sec[i] = t0 + (i * bar_dur_est)
    else: 
        max_bar_idx = max(fd.get("bar_index", 0) for fd in flow_data if isinstance(fd, dict)) if flow_data else 0
        for i in range(max_bar_idx + 4): bar_absolute_start_times_sec[i] = i * bar_duration_sec_from_bpm

    max_event_time_sec = 0.0
    for fd_item in flow_data:
        if not isinstance(fd_item, dict): continue
        bar_idx = fd_item.get("bar_index")
        if bar_idx is None or bar_idx not in bar_absolute_start_times_sec: continue
        
        bar_start_sec_val = bar_absolute_start_times_sec[bar_idx]
        syllable_subdivisions = fd_item.get("syllable_start_subdivisions", [])
        syll_dur_bins = fd_item.get("syllable_durations_quantized", [])

        for i in range(len(syllable_subdivisions)):
            syl_start_offset_sec = syllable_subdivisions[i] * subdivision_duration_sec
            approx_syl_dur_beats = tokenizer.dequantize_syllable_duration_bin(syll_dur_bins[i] if i < len(syll_dur_bins) else 0)
            syl_dur_sec = approx_syl_dur_beats * beat_duration_sec
            max_event_time_sec = max(max_event_time_sec, bar_start_sec_val + syl_start_offset_sec + syl_dur_sec)

    min_output_duration_sec = 1.0
    if instrumental_waveform is not None and instrumental_waveform.size > 0:
        output_duration_sec = max(max_event_time_sec, len(instrumental_waveform) / sample_rate, min_output_duration_sec)
        output_waveform = np.copy(instrumental_waveform)
        required_samples = int(output_duration_sec * sample_rate)
        if len(output_waveform) < required_samples:
            output_waveform = np.concatenate((output_waveform, np.zeros(required_samples - len(output_waveform), dtype=np.float32)))
        elif len(output_waveform) > required_samples: 
            output_waveform = output_waveform[:required_samples]
    else:
        output_duration_sec = max(max_event_time_sec, min_output_duration_sec)
        output_waveform = np.zeros(int(output_duration_sec * sample_rate), dtype=np.float32)

    num_syllables_placed = 0
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
            
            # Dequantize duration bin to get duration in beats, then seconds
            syl_dur_beats_approx = tokenizer.dequantize_syllable_duration_bin(syl_dur_bin)
            syl_actual_duration_sec = max(0.01, syl_dur_beats_approx * beat_duration_sec) # Ensure min duration

            syllable_sound = generate_syllable_sound_event(syl_actual_duration_sec, syl_stress, sample_rate)
            
            start_sample = int(abs_syl_onset_sec * sample_rate)
            end_sample = start_sample + len(syllable_sound)

            if end_sample <= len(output_waveform) and len(syllable_sound)>0:
                output_waveform[start_sample:end_sample] += syllable_sound
                num_syllables_placed +=1
    
    print(f"Placed {num_syllables_placed} syllable sounds in the audio.")
    return (output_waveform, sample_rate)


def visualize_flow_rhythm(
    model_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    model_config_path: str = DEFAULT_MODEL_CONFIG_PATH,
    tokenizer_config_path: str = DEFAULT_TOKENIZER_CONFIG_PATH,
    instrumental_audio_path: str = DEFAULT_INSTRUMENTAL_PATH,
    num_prompt_bars: int = 2, 
    max_generated_flow_lines: int = 8, 
    output_filename_prefix: str = "flow_vis",
    output_format: str = "wav" # "wav" or "mp3"
):
    print(f"--- Visualizing Flow Rhythm (Enhanced) ---")
    print(f"Using device: {DEVICE}")
    print(f"Model checkpoint: {model_checkpoint_path}")
    print(f"Instrumental: {instrumental_audio_path}")

    audio_processor = AudioProcessor()
    beat_feature_extractor = BeatFeatureExtractor() 

    if not os.path.exists(instrumental_audio_path):
        print(f"Error: Instrumental audio file not found: {instrumental_audio_path}. Exiting."); return
    
    instrumental_sr = beat_feature_extractor.sample_rate 
    instrumental_waveform, loaded_sr = audio_processor.load_audio(instrumental_audio_path, target_sr=instrumental_sr)
    if instrumental_waveform.size == 0:
        print(f"Error: Could not load instrumental: {instrumental_audio_path}. Exiting."); return
    
    beat_info_for_clicktrack: BeatInfo = audio_processor.get_beat_info(instrumental_waveform, instrumental_sr)
    if not beat_info_for_clicktrack or beat_info_for_clicktrack.get("bpm", 0) <=0:
        print(f"Error: Could not extract valid BeatInfo from instrumental: {instrumental_audio_path}. Exiting."); return
    
    if not os.path.exists(tokenizer_config_path):
        print(f"Error: Tokenizer config not found: {tokenizer_config_path}. Exiting."); return
    
    # Global tokenizer instance for generation and click track
    global tokenizer_instance_for_generation 
    tokenizer_instance_for_generation = FlowTokenizer(config_path=tokenizer_config_path)
    vocab_size = tokenizer_instance_for_generation.get_vocab_size()
    pad_token_id = tokenizer_instance_for_generation.pad_token_id

    if not os.path.exists(model_config_path):
        print(f"Error: Model config not found: {model_config_path}. Exiting."); return
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
        print(f"Error: Model checkpoint not found: {model_checkpoint_path}. Exiting."); return
        
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location=DEVICE, weights_only=False)
        state_dict_key = next((k for k in ['model_state_dict', 'state_dict', 'model'] if k in checkpoint and isinstance(checkpoint[k], dict)), None)
        state_dict = checkpoint[state_dict_key] if state_dict_key else checkpoint
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}. Exiting."); return
        
    model.to(DEVICE); model.eval()
    print(f"Model loaded from {model_checkpoint_path} and set to eval mode.")

    song_id_for_stems = os.path.splitext(os.path.basename(instrumental_audio_path))[0]
    _default_pre_separated_stems_root_dir = "data/stems_cache" 
    _default_demucs_model_name = "htdemucs_ft" 
    stems_input_dir_for_song = os.path.join(_default_pre_separated_stems_root_dir, _default_demucs_model_name, song_id_for_stems)
    if not os.path.isdir(stems_input_dir_for_song): stems_input_dir_for_song = None

    song_beat_features_for_prompt: SongBeatFeatures = beat_feature_extractor.extract_features_for_song(
        audio_path=instrumental_audio_path, stems_input_dir=stems_input_dir_for_song 
    )
    if not song_beat_features_for_prompt:
        print(f"Error: No beat features extracted. Cannot create prompt. Exiting."); return
    
    num_prompt_bars = min(num_prompt_bars, len(song_beat_features_for_prompt))
    if num_prompt_bars == 0: print("Error: Zero bars for prompt. Exiting."); return
    prompt_bar_features = song_beat_features_for_prompt[:num_prompt_bars]

    prompt_token_ids_list: List[int] = [tokenizer_instance_for_generation.bos_token_id]
    bos_seg_id, bos_intra_pos_id = get_next_context_ids_for_token([], tokenizer_instance_for_generation.bos_token_id, tokenizer_instance_for_generation, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
    prompt_segment_ids_list: List[int] = [bos_seg_id] 
    prompt_intra_line_pos_ids_list: List[int] = [bos_intra_pos_id]
    
    for bar_features_for_prompt in prompt_bar_features:
        bar_feature_tokens = tokenizer_instance_for_generation.encode_bar_features(bar_features_for_prompt)
        for token_id in bar_feature_tokens:
            seg_id, intra_pos_id = get_next_context_ids_for_token(prompt_token_ids_list, token_id, tokenizer_instance_for_generation, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
            prompt_token_ids_list.append(token_id); prompt_segment_ids_list.append(seg_id); prompt_intra_line_pos_ids_list.append(intra_pos_id)
        
        sep_token_id = tokenizer_instance_for_generation.sep_input_flow_token_id
        sep_seg_id, sep_intra_pos_id = get_next_context_ids_for_token(prompt_token_ids_list, sep_token_id, tokenizer_instance_for_generation, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
        prompt_token_ids_list.append(sep_token_id); prompt_segment_ids_list.append(sep_seg_id); prompt_intra_line_pos_ids_list.append(sep_intra_pos_id)

    line_start_token_id_val = tokenizer_instance_for_generation.line_start_token_id
    line_start_seg_id, line_start_intra_pos_id = get_next_context_ids_for_token(prompt_token_ids_list, line_start_token_id_val, tokenizer_instance_for_generation, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
    prompt_token_ids_list.append(line_start_token_id_val); prompt_segment_ids_list.append(line_start_seg_id); prompt_intra_line_pos_ids_list.append(line_start_intra_pos_id)

    idx_prompt_tensor = torch.tensor([prompt_token_ids_list], dtype=torch.long, device=DEVICE)
    seg_ids_prompt_tensor = torch.tensor([prompt_segment_ids_list], dtype=torch.long, device=DEVICE)
    intra_pos_ids_prompt_tensor = torch.tensor([prompt_intra_line_pos_ids_list], dtype=torch.long, device=DEVICE)

    avg_tokens_per_syllable = 3 # START_SUBDIV, DURATION_BIN, STRESS_BIN
    avg_syllables_per_line = tokenizer_instance_for_generation.max_syllables // 2 
    avg_control_tokens_per_line = 5 # LINE_START, SYL, OFF, DUR, END_SYLL_SEQ
    avg_tokens_per_line = avg_control_tokens_per_line + (avg_syllables_per_line * avg_tokens_per_syllable)
    max_new_tokens = max_generated_flow_lines * avg_tokens_per_line + 20 
    
    print(f"Generating up to {max_new_tokens} new tokens for flow...")
    with torch.no_grad():
        generated_ids_full = model.generate(
            idx_prompt_tensor, seg_ids_prompt_tensor, intra_pos_ids_prompt_tensor,
            max_new_tokens=max_new_tokens, tokenizer=tokenizer_instance_for_generation, 
            temperature=0.7, top_k=50 
        )
    
    start_of_model_generation_idx = len(prompt_token_ids_list)
    model_generated_tokens = generated_ids_full[0, start_of_model_generation_idx:].tolist()
    
    # --- Decoding the generated flow (logic remains same, but FlowDatum has more fields) ---
    decoded_flow_data: FlowData = []
    current_bar_idx_for_flow_context = prompt_bar_features[-1]["bar_index"] if prompt_bar_features else 0
    line_idx_in_bar_counter_for_this_context = 0 
    idx_in_model_generated_tokens = 0
    
    if model_generated_tokens:
        tokens_for_first_line_attempt = [prompt_token_ids_list[-1]] 
        temp_idx = 0
        while temp_idx < len(model_generated_tokens):
            token_id = model_generated_tokens[temp_idx]
            if token_id in [tokenizer_instance_for_generation.eos_token_id, tokenizer_instance_for_generation.bar_start_token_id, tokenizer_instance_for_generation.line_start_token_id]:
                break 
            tokens_for_first_line_attempt.append(token_id)
            temp_idx += 1
        
        datum = tokenizer_instance_for_generation.decode_flow_tokens_to_datum(tokens_for_first_line_attempt, current_bar_idx_for_flow_context, line_idx_in_bar_counter_for_this_context)
        if datum:
            decoded_flow_data.append(datum)
            line_idx_in_bar_counter_for_this_context += 1
            idx_in_model_generated_tokens = temp_idx 

    while idx_in_model_generated_tokens < len(model_generated_tokens):
        current_token_id_from_model = model_generated_tokens[idx_in_model_generated_tokens]
        if current_token_id_from_model == tokenizer_instance_for_generation.bar_start_token_id:
            current_bar_idx_for_flow_context += 1
            line_idx_in_bar_counter_for_this_context = 0
            idx_in_model_generated_tokens += 1; continue
        if current_token_id_from_model == tokenizer_instance_for_generation.eos_token_id: break
        if current_token_id_from_model != tokenizer_instance_for_generation.line_start_token_id: break

        tokens_for_this_line_attempt = [current_token_id_from_model] 
        idx_in_model_generated_tokens += 1 
        temp_idx_current_line = idx_in_model_generated_tokens
        while temp_idx_current_line < len(model_generated_tokens):
            token_id = model_generated_tokens[temp_idx_current_line]
            if token_id in [tokenizer_instance_for_generation.eos_token_id, tokenizer_instance_for_generation.bar_start_token_id, tokenizer_instance_for_generation.line_start_token_id]:
                break 
            tokens_for_this_line_attempt.append(token_id)
            temp_idx_current_line += 1
            
        datum = tokenizer_instance_for_generation.decode_flow_tokens_to_datum(tokens_for_this_line_attempt, current_bar_idx_for_flow_context, line_idx_in_bar_counter_for_this_context)
        if datum:
            decoded_flow_data.append(datum)
            line_idx_in_bar_counter_for_this_context += 1
            idx_in_model_generated_tokens = temp_idx_current_line 
        else: break

    print("\nDecoded FlowData (Enhanced):")
    if decoded_flow_data:
        for i, fd in enumerate(decoded_flow_data):
            print(f"  Line {i+1}: Bar {fd['bar_index']}, LineInBar {fd['line_index_in_bar']}, Syls {fd['syllables']}, "
                  f"Offset {fd['start_offset_beats']:.2f}b, Dur {fd['duration_beats']:.2f}b, "
                  f"Subdivs: {fd.get('syllable_start_subdivisions')}, DurBins: {fd.get('syllable_durations_quantized')}, Stresses: {fd.get('syllable_stresses')}")
    else:
        print("  No valid flow data decoded."); return

    click_track_audio, click_sr = generate_syllable_click_track(
        flow_data=decoded_flow_data, beat_info=beat_info_for_clicktrack,
        tokenizer=tokenizer_instance_for_generation, 
        instrumental_waveform=instrumental_waveform, sample_rate=instrumental_sr 
    )

    if click_track_audio.size > 0:
        safe_checkpoint_name = "".join(c if c.isalnum() else '_' for c in os.path.basename(model_checkpoint_path).replace('.pt',''))
        safe_instrumental_name = "".join(c if c.isalnum() else '_' for c in os.path.splitext(os.path.basename(instrumental_audio_path))[0])
        output_audio_filename = f"{output_filename_prefix}_{safe_checkpoint_name}_on_{safe_instrumental_name}_bars{num_prompt_bars}.{output_format}"
        output_audio_path = os.path.join(OUTPUT_DIR, output_audio_filename)
        
        try:
            sf.write(output_audio_path, click_track_audio, click_sr) # format can be inferred for wav, explicit for mp3
            print(f"\nFlow rhythm visualization audio saved to: {output_audio_path}")
        except Exception as e_sf:
            print(f"Error saving audio with soundfile: {e_sf}")
            if output_format == "mp3":
                print("  This might be due to missing LAME encoder for MP3. Try 'wav' format or install LAME.")
            # Fallback to WAV if MP3 failed
            if output_format == "mp3":
                try:
                    wav_path = output_audio_path.replace(".mp3", ".wav")
                    sf.write(wav_path, click_track_audio, click_sr)
                    print(f"  Saved as WAV instead: {wav_path}")
                except Exception as e_wav:
                    print(f"  Failed to save as WAV as well: {e_wav}")
    else:
        print("\nFailed to generate flow rhythm visualization audio.")

if __name__ == "__main__":
    test_checkpoint = DEFAULT_CHECKPOINT_PATH 
    if not os.path.exists(test_checkpoint):
        print(f"Error: Test checkpoint '{test_checkpoint}' not found.")
    elif not os.path.exists(DEFAULT_INSTRUMENTAL_PATH):
         print(f"Error: Default instrumental '{DEFAULT_INSTRUMENTAL_PATH}' not found.")
    else:
        visualize_flow_rhythm(
            output_format="mp3" # Test mp3 output
        )