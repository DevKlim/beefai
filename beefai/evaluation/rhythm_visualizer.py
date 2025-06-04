import torch
import torch.nn.functional as F
import os
import sys
import yaml
import numpy as np
import soundfile as sf
import json 
from typing import List, Dict, Optional, Tuple, Any

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
DEFAULT_INSTRUMENTAL_PATH = "data/instrumentals/blackertheberry2.mp3" 
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
    stress_level: int, # 0: none, 1: primary, 2: secondary
    sample_rate: int = 44100
) -> np.ndarray:
    """Generates a more percussive, click-like sound for syllables."""
    # Core sound (e.g., a short noise burst or click)
    core_sound_duration_sec = 0.05 # Short, sharp sound
    num_samples_core = int(core_sound_duration_sec * sample_rate)

    if num_samples_core <= 0: # Avoid issues with extremely short durations
        return np.array([], dtype=np.float32)

    # Simple noise burst with an envelope
    noise = (np.random.rand(num_samples_core) * 2 - 1) * 0.5 # Scaled noise
    t_env = np.linspace(0, 1, num_samples_core, endpoint=False)
    decay_rate_sharp = 25 # Makes the envelope decay quickly for a "tick" sound
    envelope = np.exp(-t_env * decay_rate_sharp) 
    
    # Add a bit of tonal character to distinguish stress (optional, can be subtle)
    freq1 = 800  # Base frequency for the click
    freq2 = 1200 # Slightly higher for stressed clicks
    t_sine = np.linspace(0, core_sound_duration_sec, num_samples_core, endpoint=False)
    tonal_component = 0.3 * np.sin(2 * np.pi * freq1 * t_sine) + \
                      0.2 * np.sin(2 * np.pi * freq2 * t_sine)
    
    percussive_sound = (noise + tonal_component * 0.5) * envelope
    
    # Modify amplitude based on stress
    amplitude_modifier = 1.0
    if stress_level == 1: # Primary stress
        amplitude_modifier = 1.2 # Louder
    elif stress_level == 2: # Secondary stress
        amplitude_modifier = 1.1 # Slightly louder
    
    percussive_sound *= amplitude_modifier
    
    # Pad with silence to match the syllable's actual duration from FlowData
    total_samples_for_event_duration = int(duration_sec * sample_rate)
    if total_samples_for_event_duration > num_samples_core:
        padding = np.zeros(total_samples_for_event_duration - num_samples_core, dtype=np.float32)
        final_sound = np.concatenate((percussive_sound, padding))
    elif total_samples_for_event_duration > 0 : # If target duration is shorter than core sound
        final_sound = percussive_sound[:total_samples_for_event_duration]
    else: # If target duration is zero or negative (should not happen for valid syllables)
        final_sound = np.array([], dtype=np.float32)

    # Ensure at least a minimal sound if duration_sec implies it but calculations result in empty
    if final_sound.size == 0 and duration_sec > 0.001: # e.g. 1ms
        final_sound = np.zeros(max(1, int(0.001 * sample_rate)), dtype=np.float32) # Min 1 sample or 1ms of silence

    return final_sound

def generate_syllable_click_track(
    flow_data: FlowData, 
    beat_info: BeatInfo, 
    tokenizer: FlowTokenizer, # Used for dequantizing syllable duration
    instrumental_waveform_mono: Optional[np.ndarray] = None, 
    sample_rate: int = 44100,
    vocal_click_level_db: float = 0.0 # Level of clicks relative to normalized instrumental
) -> Tuple[np.ndarray, int]:
    """Generates a click track for syllables, mixed with an optional instrumental."""
    if not flow_data:
        return (np.array([], dtype=np.float32), sample_rate)

    bpm = beat_info.get("bpm")
    if not bpm or bpm <=0: 
        bpm = 120.0 
    
    beats_per_bar = beat_info.get("beats_per_bar", 4)
    if beats_per_bar <=0: beats_per_bar = 4

    beat_duration_sec = 60.0 / bpm
    # Default bar duration if not perfectly estimated from downbeats
    bar_duration_sec_from_bpm = beat_duration_sec * beats_per_bar
    # Subdivision duration based on the tokenizer's max_subdivisions setting
    subdivision_duration_sec = bar_duration_sec_from_bpm / max(1, tokenizer.max_subdivisions)

    # Determine absolute start times for each bar index present in flow_data
    bar_absolute_start_times_sec: Dict[int, float] = {}
    if beat_info.get("downbeat_times") and len(beat_info["downbeat_times"]) > 0 :
        max_bar_idx_flow = max((fd.get("bar_index", 0) for fd in flow_data if isinstance(fd, dict)), default=-1)
        current_downbeats = list(beat_info["downbeat_times"])
        
        # If flow_data references bars beyond known downbeats, project them
        if max_bar_idx_flow >= len(current_downbeats):
            last_dt = current_downbeats[-1] if current_downbeats else 0.0
            # Use estimated_bar_duration for projection if available and valid, else use BPM-based
            est_bar_dur = beat_info.get("estimated_bar_duration", bar_duration_sec_from_bpm)
            if est_bar_dur <= 0: est_bar_dur = bar_duration_sec_from_bpm # Ensure positive
            
            for i in range(len(current_downbeats), max_bar_idx_flow + 5): # Project a few extra bars
                last_dt += est_bar_dur
                current_downbeats.append(last_dt)
        
        for i, dt in enumerate(current_downbeats): bar_absolute_start_times_sec[i] = dt

    elif beat_info.get("estimated_bar_duration") and beat_info["estimated_bar_duration"] > 0:
        # Fallback if no downbeats but we have an estimated bar duration
        bar_dur_est = beat_info["estimated_bar_duration"]
        max_bar_idx = max((fd.get("bar_index", 0) for fd in flow_data if isinstance(fd, dict)), default=-1)
        t0 = 0.0 # Assume first bar starts at 0.0 if no other info
        for i in range(max_bar_idx + 5): bar_absolute_start_times_sec[i] = t0 + (i * bar_dur_est)
    else: # Last resort: Use BPM and beats_per_bar to calculate bar starts
        max_bar_idx = max((fd.get("bar_index", 0) for fd in flow_data if isinstance(fd, dict)), default=-1)
        for i in range(max_bar_idx + 5): bar_absolute_start_times_sec[i] = i * bar_duration_sec_from_bpm

    # Calculate total duration of the output audio based on the last syllable event
    max_event_time_sec = 0.0
    if flow_data: # Check again as it might have been empty initially
        for fd_item in flow_data:
            if not isinstance(fd_item, dict): continue
            bar_idx = fd_item.get("bar_index")
            if bar_idx is None : continue # Should not happen with valid FlowDatum
            
            # Ensure bar_idx has a timing, even if we had to project it just now
            if bar_idx not in bar_absolute_start_times_sec:
                if bar_idx >= 0: # Only project for non-negative indices
                    if bar_absolute_start_times_sec: # If we have some timings
                        last_known_bar = max(bar_absolute_start_times_sec.keys())
                        last_known_time = bar_absolute_start_times_sec[last_known_bar]
                        est_bar_dur_proj = beat_info.get("estimated_bar_duration", bar_duration_sec_from_bpm)
                        if est_bar_dur_proj <= 0: est_bar_dur_proj = bar_duration_sec_from_bpm
                        bar_absolute_start_times_sec[bar_idx] = last_known_time + (bar_idx - last_known_bar) * est_bar_dur_proj
                    else: # No prior timings at all, start from scratch for this bar
                        bar_absolute_start_times_sec[bar_idx] = bar_idx * bar_duration_sec_from_bpm 
                else: # Negative bar index, skip
                    continue

            bar_start_sec_val = bar_absolute_start_times_sec[bar_idx]
            syllable_subdivisions = fd_item.get("syllable_start_subdivisions", [])
            syll_dur_bins = fd_item.get("syllable_durations_quantized", [])

            for i in range(len(syllable_subdivisions)):
                syl_start_offset_sec = syllable_subdivisions[i] * subdivision_duration_sec
                # Dequantize syllable duration (from beats to seconds)
                approx_syl_dur_beats = tokenizer.dequantize_syllable_duration_bin(syll_dur_bins[i] if i < len(syll_dur_bins) else 0)
                syl_dur_sec = approx_syl_dur_beats * beat_duration_sec
                max_event_time_sec = max(max_event_time_sec, bar_start_sec_val + syl_start_offset_sec + syl_dur_sec)

    min_output_duration_sec = 0.1 # Ensure audio is at least this long
    
    # Also consider instrumental length if provided
    instrumental_duration_samples = len(instrumental_waveform_mono) if instrumental_waveform_mono is not None else 0
    instrumental_duration_sec = instrumental_duration_samples / sample_rate if sample_rate > 0 else 0
    output_duration_sec = max(max_event_time_sec, instrumental_duration_sec, min_output_duration_sec)
    total_output_samples = max(1, int(output_duration_sec * sample_rate)) # Ensure at least 1 sample

    # Base waveform (can be instrumental or silence)
    output_waveform_base = np.zeros(total_output_samples, dtype=np.float32)
    if instrumental_waveform_mono is not None and instrumental_waveform_mono.size > 0:
        len_to_copy = min(len(instrumental_waveform_mono), total_output_samples)
        output_waveform_base[:len_to_copy] = instrumental_waveform_mono[:len_to_copy]
        # Normalize instrumental to avoid overpowering clicks if it's too loud
        if np.max(np.abs(output_waveform_base)) > 1e-5: # Avoid div by zero for silent instrumental
             output_waveform_base = output_waveform_base / np.max(np.abs(output_waveform_base)) * 0.7 # Normalize to 0.7 peak

    # Create clicks on a separate track
    click_track_only = np.zeros_like(output_waveform_base)
    num_syllables_placed = 0

    if flow_data: # Double check, might have been empty from start
        for fd_item in flow_data:
            if not isinstance(fd_item, dict): continue
            bar_idx = fd_item.get("bar_index")
            if bar_idx is None or bar_idx not in bar_absolute_start_times_sec: continue
            
            bar_start_sec_val = bar_absolute_start_times_sec[bar_idx]
            syllable_starts = fd_item.get("syllable_start_subdivisions", [])
            syllable_dur_bins = fd_item.get("syllable_durations_quantized", [])
            syllable_stresses = fd_item.get("syllable_stresses", []) # Get stress info

            for i in range(len(syllable_starts)):
                syl_onset_subdiv = syllable_starts[i]
                syl_dur_bin = syllable_dur_bins[i] if i < len(syllable_dur_bins) else 0 # Fallback if lists unequal
                syl_stress = syllable_stresses[i] if i < len(syllable_stresses) else 0 # Default to no stress

                syl_onset_in_bar_sec = syl_onset_subdiv * subdivision_duration_sec
                abs_syl_onset_sec = bar_start_sec_val + syl_onset_in_bar_sec
                
                # Syllable's actual duration in seconds
                syl_dur_beats_approx = tokenizer.dequantize_syllable_duration_bin(syl_dur_bin)
                syl_actual_duration_sec = max(0.01, syl_dur_beats_approx * beat_duration_sec) # Ensure min duration (e.g. 10ms)

                syllable_sound = generate_syllable_sound_event(syl_actual_duration_sec, syl_stress, sample_rate)
                
                start_sample = int(abs_syl_onset_sec * sample_rate)
                end_sample = start_sample + len(syllable_sound)

                # Add syllable sound to the click track, ensuring it fits
                if click_track_only.size > 0 and end_sample <= len(click_track_only) and len(syllable_sound)>0:
                    click_track_only[start_sample:end_sample] += syllable_sound
                    num_syllables_placed +=1
                elif click_track_only.size > 0 and start_sample < len(click_track_only) and len(syllable_sound)>0: # Partial fit if at end
                    fit_len = min(len(syllable_sound), len(click_track_only) - start_sample)
                    click_track_only[start_sample : start_sample+fit_len] += syllable_sound[:fit_len]
                    num_syllables_placed +=1
        
        if num_syllables_placed > 0:
            print(f"Placed {num_syllables_placed} percussive syllable sounds in the audio.")
            # Normalize click track before applying gain
            if np.max(np.abs(click_track_only)) > 1e-5:
                click_track_only /= np.max(np.abs(click_track_only)) # Normalize to +/- 1.0
            
            # Apply user-defined gain to clicks
            click_gain = 10**(vocal_click_level_db / 20.0)
            click_track_only *= click_gain
            
            # Mix with base (instrumental)
            final_mixed_waveform = output_waveform_base + click_track_only
            
            # Final normalization of the mixed output to prevent clipping
            max_final_amp = np.max(np.abs(final_mixed_waveform))
            if max_final_amp > 1.0: final_mixed_waveform /= max_final_amp
            elif max_final_amp == 0 and np.max(np.abs(output_waveform_base)) > 0 : # If clicks were silent but instrumental wasn't
                final_mixed_waveform = output_waveform_base

            return (final_mixed_waveform, sample_rate)
        
    # If no syllables placed, or flow_data was empty, return just the base (instrumental or silence)
    return (output_waveform_base, sample_rate)

def _create_beat_info_from_custom_features(song_beat_features: SongBeatFeatures) -> BeatInfo:
    """Creates a BeatInfo dict from more detailed SongBeatFeatures.
       Ensures 'downbeat_times' are comprehensive enough for later synthesis steps.
    """
    if not song_beat_features:
        return { "bpm": 120.0, "beats_per_bar": 4, "estimated_bar_duration": 2.0, "downbeat_times": [i * 2.0 for i in range(16)], "beat_times": []  } # Default for 16 bars

    first_bar = song_beat_features[0]
    bpm = first_bar.get("bpm", 120.0)
    time_sig_num, time_sig_den = first_bar.get("time_signature", (4, 4))
    beats_per_bar = time_sig_num

    if bpm <= 0: bpm = 120.0 # Fallback
    if beats_per_bar <= 0: beats_per_bar = 4 # Fallback

    beat_duration_sec = 60.0 / bpm
    default_estimated_bar_duration_sec = beat_duration_sec * beats_per_bar

    # Try to construct downbeat_times from bar_start_time_sec if available
    downbeat_times: List[float] = []
    
    # Max bar index defined in SBF plus some buffer for generation
    num_bars_in_sbf = len(song_beat_features)
    max_defined_bar_idx = -1
    if num_bars_in_sbf > 0:
        valid_bar_indices = [bf.get("bar_index") for bf in song_beat_features if isinstance(bf, dict) and bf.get("bar_index") is not None]
        if valid_bar_indices: max_defined_bar_idx = max(valid_bar_indices)
        else: max_defined_bar_idx = num_bars_in_sbf -1 # Assume 0-indexed if no explicit indices
            
    total_bars_to_map = max(num_bars_in_sbf, max_defined_bar_idx + 1) + 4 # Add a small buffer of bars

    # Populate downbeat_times using bar_start_time_sec if present, otherwise project
    last_known_time = 0.0
    last_known_bar_idx = -1
    last_bar_duration_used = default_estimated_bar_duration_sec

    for i in range(total_bars_to_map) : # Iterate up to the required number of bars
        bar_feature_for_this_idx = next((bf for bf in song_beat_features if isinstance(bf, dict) and bf.get("bar_index") == i), None)
        
        current_bar_start_time = 0.0
        current_bar_duration = default_estimated_bar_duration_sec # Assume default initially

        if bar_feature_for_this_idx and bar_feature_for_this_idx.get("bar_start_time_sec") is not None:
            current_bar_start_time = bar_feature_for_this_idx["bar_start_time_sec"]
            current_bar_duration = bar_feature_for_this_idx.get("bar_duration_sec", default_estimated_bar_duration_sec)
        elif i > 0 and downbeat_times: # If not the first bar and previous downbeats exist
            # Project from previous bar's end
            prev_bar_feature_for_dur_calc = next((bf for bf in song_beat_features if isinstance(bf,dict) and bf.get("bar_index") == i-1), None)
            dur_of_prev_bar = default_estimated_bar_duration_sec # Default if prev bar info also sparse
            if prev_bar_feature_for_dur_calc and prev_bar_feature_for_dur_calc.get("bar_duration_sec") is not None:
                dur_of_prev_bar = prev_bar_feature_for_dur_calc.get("bar_duration_sec")
            
            current_bar_start_time = downbeat_times[-1] + dur_of_prev_bar
            # If this projected bar has a feature, use its duration
            if bar_feature_for_this_idx and bar_feature_for_this_idx.get("bar_duration_sec") is not None:
                 current_bar_duration = bar_feature_for_this_idx.get("bar_duration_sec")

        else: # First bar (i=0) or no previous downbeats to project from
            current_bar_start_time = 0.0 # Assume starts at 0
            first_bar_feat_check = next((bf for bf in song_beat_features if isinstance(bf, dict) and bf.get("bar_index") == 0), None)
            current_bar_duration = first_bar_feat_check.get("bar_duration_sec", default_estimated_bar_duration_sec) if first_bar_feat_check else default_estimated_bar_duration_sec
        
        downbeat_times.append(current_bar_start_time)
        last_known_time = current_bar_start_time
        last_known_bar_idx = i
        last_bar_duration_used = current_bar_duration # Store for potential next iteration if needed

    # Construct all beat_times based on these downbeats and bar-specific BPM/time_sig if available
    all_beat_times = []
    for bar_idx_map in range(len(downbeat_times)): # Iterate through all mapped downbeats
        bar_feat_match = next((bf for bf in song_beat_features if isinstance(bf, dict) and bf.get("bar_index") == bar_idx_map), None)
        bar_start_time = downbeat_times[bar_idx_map]
        
        current_bar_bpm = bpm # Default to overall BPM
        current_bar_time_sig_num = beats_per_bar # Default to overall time sig
        
        if bar_feat_match:
            current_bar_bpm = bar_feat_match.get("bpm", bpm)
            current_bar_time_sig_num = bar_feat_match.get("time_signature", (beats_per_bar, 4))[0]
        
        if current_bar_bpm <=0: current_bar_bpm = bpm # Fallback
        if current_bar_time_sig_num <=0: current_bar_time_sig_num = beats_per_bar # Fallback

        current_beat_duration_sec = 60.0 / current_bar_bpm
        for beat_in_bar in range(current_bar_time_sig_num):
            all_beat_times.append(bar_start_time + beat_in_bar * current_beat_duration_sec)
    
    return {
        "bpm": bpm,
        "beat_times": all_beat_times,
        "downbeat_times": downbeat_times,
        "estimated_bar_duration": default_estimated_bar_duration_sec, # This is the overall estimate
        "beats_per_bar": beats_per_bar,
        "sbf_features_for_timing_ref": song_beat_features # Store original SBF for finer per-bar details later if needed by synthesizer
    }

def _generate_flow_core(
    prompt_bar_features: SongBeatFeatures,
    all_song_beat_features: SongBeatFeatures, # Used for guiding generation beyond prompt
    model_checkpoint_path: str,
    model_config_path: str,
    tokenizer_config_path: str,
    context_bar_idx_start: int, # The bar index *after* the last prompt bar
    output_name_suffix: str,
    generation_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[FlowData], Optional[FlowTokenizer], Optional[FlowGPTConfig]]:
    
    # Default generation parameters (can be overridden)
    # These defaults are now more aligned with the user's request for slower, more controlled flow for visualization
    params = { 
        "temperature": 0.7, 
        "top_k": 50, 
        "max_lines_per_bar_heuristic": 2, # How many lines the model aims for per bar
        "syl_count_bias_strength": None, # e.g., -1.0 for fewer syllables, 1.0 for more
        "short_syl_dur_bias_strength": None, # e.g., 1.0 to favor shorter syllable durations
        "long_syl_dur_bias_strength": None, # e.g., 1.0 to favor longer syllable durations
        "rhythmic_offset_bias_strength": None, # e.g., 0.5 to favor on-beat/simple off-beat offsets
        "rhythmic_line_duration_bias_strength": None, # e.g., 0.5 to favor simpler line durations
        "rhythmic_syllable_duration_bias_strength": None, # e.g., 0.5 to favor simpler syllable durations
        "avoid_short_lines_end_bias_strength": -2.0, # New: Negative bias against very short lines if conditions met
        "min_syllables_threshold": 5, # New: Minimum number of syllables for a "proper" line
        "min_syllables_penalty_strength": 1.5 # New: Strength of penalty for lines shorter than threshold
    }
    if generation_params: # Apply overrides
        for k, v in generation_params.items():
            if k in params: # Only update known params
                 params[k] = v

    # Ensure None for bias/penalty params means they are not applied or removed
    final_params = {}
    for k, v in params.items():
        if ("bias_strength" in k or "penalty_strength" in k) and v is None:
            pass # Skip if bias/penalty is explicitly None (meaning no bias/penalty)
        else:
            final_params[k] = v
    params = final_params # Use the cleaned params


    if not os.path.exists(tokenizer_config_path): print(f"Error: Tokenizer config not found: {tokenizer_config_path}."); return None, None, None
    tokenizer = FlowTokenizer(config_path=tokenizer_config_path)
    vocab_size = tokenizer.get_vocab_size(); pad_token_id = tokenizer.pad_token_id
    if not os.path.exists(model_config_path): print(f"Error: Model config not found: {model_config_path}."); return None, tokenizer, None
    model_yaml_config = load_yaml_config(model_config_path)
    gpt_config = FlowGPTConfig(
        vocab_size=vocab_size, block_size=model_yaml_config["block_size"],
        n_layer=model_yaml_config["n_layer"], n_head=model_yaml_config["n_head"],
        n_embd=model_yaml_config["n_embd"], max_segment_types=model_yaml_config["max_segment_types"],
        max_intra_line_positions=model_yaml_config["max_intra_line_positions"],
        dropout=model_yaml_config["dropout"], bias=model_yaml_config.get("bias", True),
        pad_token_id=pad_token_id )
    model = FlowTransformerDecoder(gpt_config)
    if not os.path.exists(model_checkpoint_path): print(f"Error: Model checkpoint not found: {model_checkpoint_path}."); return None, tokenizer, gpt_config
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location=DEVICE, weights_only=False) # weights_only=False is safer if custom objects in checkpoint
        state_dict_key = next((k for k in ['model_state_dict', 'state_dict', 'model'] if k in checkpoint and isinstance(checkpoint[k], dict)), None)
        if state_dict_key: state_dict = checkpoint[state_dict_key]
        elif isinstance(checkpoint, dict) and 'config' in checkpoint and 'model_state_dict' not in checkpoint: # For some HuggingFace-like saves
            if any(key.startswith('transformer.') or key.startswith('lm_head.') for key in checkpoint.keys()): state_dict = checkpoint
            else: raise ValueError("Checkpoint is a dictionary but not a recognized state_dict format or a structured model save.")
        elif isinstance(checkpoint, dict) : state_dict = checkpoint # Assume it's the state_dict itself
        else: raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")
        # Handle '_orig_mod.' prefix if model was compiled with torch.compile
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()): state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    except Exception as e: print(f"Error loading checkpoint {model_checkpoint_path}: {e}."); return None, tokenizer, gpt_config
    model.to(DEVICE); model.eval()
    print(f"Model loaded from {model_checkpoint_path} and set to eval mode.")

    # --- Prompt Construction ---
    current_token_ids_list = [tokenizer.bos_token_id]; current_segment_ids_list = [0]; current_intra_pos_ids_list = [0]
    if not prompt_bar_features: print("Error: No prompt_bar_features provided."); return None, tokenizer, gpt_config
    for bar_feat in prompt_bar_features:
        bar_tokens = tokenizer.encode_bar_features(bar_feat)
        for tok_id in bar_tokens:
            seg_id, intra_id = get_next_context_ids_for_token(current_token_ids_list, tok_id, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
            current_token_ids_list.append(tok_id); current_segment_ids_list.append(seg_id); current_intra_pos_ids_list.append(intra_id)
        # Add separator after each prompt bar's features
        sep_tok = tokenizer.sep_input_flow_token_id
        seg_id, intra_id = get_next_context_ids_for_token(current_token_ids_list, sep_tok, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
        current_token_ids_list.append(sep_tok); current_segment_ids_list.append(seg_id); current_intra_pos_ids_list.append(intra_id)
    # Start generation with a line_start token
    line_start_tok = tokenizer.line_start_token_id
    seg_id, intra_id = get_next_context_ids_for_token(current_token_ids_list, line_start_tok, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
    current_token_ids_list.append(line_start_tok); current_segment_ids_list.append(seg_id); current_intra_pos_ids_list.append(intra_id)

    # --- Autoregressive Generation Loop ---
    # Determine how many bars to generate flow for (e.g., length of all_song_beat_features minus prompt length)
    max_song_bars = len(all_song_beat_features) if all_song_beat_features else context_bar_idx_start + 16 # Default to 16 bars if no specific length
    # Safety net for token count: block_size for context + estimate for generated part
    max_total_tokens_safety_net = gpt_config.block_size + (25 + 2 * tokenizer.max_syllables) * (max_song_bars + 5) # Rough upper bound

    generated_flow_tokens_for_song: List[int] = [tokenizer.line_start_token_id] # Start with the initial line_start
    full_song_context_token_ids = list(current_token_ids_list); full_song_context_segment_ids = list(current_segment_ids_list); full_song_context_intra_pos_ids = list(current_intra_pos_ids_list)
    current_bar_being_generated_flow_for = context_bar_idx_start # The bar index we are currently generating flow *for*
    current_gen_phase = "EXPECTING_SYLLABLES_COUNT"; target_syllables_for_current_line = -1; syllables_generated_for_current_line = 0; lines_generated_for_current_bar = 1 # Starts at 1 due to initial line_start
    
    # For short line avoidance heuristic
    consecutive_short_syllable_lines = 0 
    max_consecutive_short_before_bias = 1 # Apply bias if more than 1 consecutive short line
    short_syllable_threshold_for_end_avoidance = 2 # Lines with 2 or fewer syllables are "short" for this specific heuristic

    print(f"Generating flow for song, starting from bar {current_bar_being_generated_flow_for}, up to ~{max_song_bars} bars or {max_total_tokens_safety_net} tokens...")
    print(f"  Generation params applied: {params}")

    for step_count in range(max_total_tokens_safety_net - len(full_song_context_token_ids)):
        idx_cond = torch.tensor([full_song_context_token_ids[-gpt_config.block_size:]], dtype=torch.long, device=DEVICE)
        seg_ids_cond = torch.tensor([full_song_context_segment_ids[-gpt_config.block_size:]], dtype=torch.long, device=DEVICE)
        intra_pos_ids_cond = torch.tensor([full_song_context_intra_pos_ids[-gpt_config.block_size:]], dtype=torch.long, device=DEVICE)
        with torch.no_grad(): logits, _ = model(idx_cond, segment_ids=seg_ids_cond, intra_line_pos_ids=intra_pos_ids_cond)
        logit_slice_for_masking = logits[0, -1, :].clone() # Get logits for the next token prediction
        
        # --- Apply Biases (carefully, only if strength is not None) ---
        # Syllable Count Bias
        if params.get('syl_count_bias_strength') is not None and current_gen_phase == "EXPECTING_SYLLABLES_COUNT":
            bias_val = params['syl_count_bias_strength']
            for i in range(1, tokenizer.max_syllables + 1):
                syl_token_id = tokenizer.token_to_id.get(f"[SYLLABLES_{i}]")
                if syl_token_id is not None: 
                    normalized_syl_pos = (i - (tokenizer.max_syllables / 2.0)) / (tokenizer.max_syllables / 2.0)
                    logit_slice_for_masking[syl_token_id] += bias_val * normalized_syl_pos

        # NEW: Apply penalty for lines shorter than min_syllables_threshold
        min_syl_thresh = params.get("min_syllables_threshold", 5)
        min_syl_penalty = params.get("min_syllables_penalty_strength")
        if min_syl_penalty is not None and min_syl_penalty != 0 and current_gen_phase == "EXPECTING_SYLLABLES_COUNT":
            for i in range(min_syl_thresh): # Penalize [SYLLABLES_0] up to [SYLLABLES_{min_syl_thresh-1}]
                syl_token_id = tokenizer.token_to_id.get(f"[SYLLABLES_{i}]")
                if syl_token_id is not None:
                    penalty_factor = 1.0
                    if i < 2: # Stronger penalty for 0 or 1 syllable lines
                        penalty_factor = 1.5 
                    logit_slice_for_masking[syl_token_id] -= (min_syl_penalty * penalty_factor)

        # NEW: Heuristic to avoid too many consecutive short lines (especially towards end of generation)
        avoid_short_lines_bias = params.get('avoid_short_lines_end_bias_strength') # e.g. -2.0
        if avoid_short_lines_bias is not None and current_gen_phase == "EXPECTING_SYLLABLES_COUNT":
            nearing_end_of_song = (max_song_bars - current_bar_being_generated_flow_for) <= 2 # If 2 or fewer bars left
            if (consecutive_short_syllable_lines >= max_consecutive_short_before_bias or nearing_end_of_song) and lines_generated_for_current_bar > 1 :
                for i in range(short_syllable_threshold_for_end_avoidance + 1): # 0, 1, 2 syllables
                    # Only apply this bias if the token is NOT already penalized by min_syllables_penalty (avoid double penalty)
                    if i < min_syl_thresh and min_syl_penalty is not None and min_syl_penalty != 0:
                        continue # Skip, as it's already handled by the stronger min_syllables_penalty

                    syl_token_id = tokenizer.token_to_id.get(f"[SYLLABLES_{i}]")
                    if syl_token_id is not None:
                        logit_slice_for_masking[syl_token_id] += avoid_short_lines_bias # Negative bias makes it less likely


        # Syllable Duration Bias
        if current_gen_phase == "EXPECTING_SYLLABLE_DURATION":
            if params.get('short_syl_dur_bias_strength') is not None:
                bias_val = params['short_syl_dur_bias_strength']
                for i in range(tokenizer.num_syllable_duration_bins // 3): # Bias towards first third (shorter)
                    syl_dur_id = tokenizer.token_to_id.get(f"[SYLLABLE_DURATION_BIN_{i}]")
                    if syl_dur_id is not None: logit_slice_for_masking[syl_dur_id] += bias_val
            if params.get('long_syl_dur_bias_strength') is not None:
                bias_val = params['long_syl_dur_bias_strength']
                for i in range( (tokenizer.num_syllable_duration_bins // 3) * 2, tokenizer.num_syllable_duration_bins): # Bias towards last third (longer)
                    syl_dur_id = tokenizer.token_to_id.get(f"[SYLLABLE_DURATION_BIN_{i}]")
                    if syl_dur_id is not None: logit_slice_for_masking[syl_dur_id] += bias_val
            # Rhythmic Syllable Duration Bias (favors common rhythmic values)
            if params.get('rhythmic_syllable_duration_bias_strength') is not None:
                bias_val = params['rhythmic_syllable_duration_bias_strength']
                target_syl_dur_beats = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0] # Emphasize simpler rhythmic values
                for i in range(tokenizer.num_syllable_duration_bins):
                    syl_dur_token_id = tokenizer.token_to_id.get(f"[SYLLABLE_DURATION_BIN_{i}]")
                    if syl_dur_token_id is not None:
                        dequantized_syl_dur = tokenizer.dequantize_syllable_duration_bin(i)
                        for target_dur in target_syl_dur_beats:
                            if abs(dequantized_syl_dur - target_dur) < 0.1: logit_slice_for_masking[syl_dur_token_id] += bias_val; break
        
        # Rhythmic Offset Bias
        if params.get('rhythmic_offset_bias_strength') is not None and current_gen_phase == "EXPECTING_OFFSET_BIN":
            bias_val = params['rhythmic_offset_bias_strength']
            # Bias towards offsets that are multiples of 0.5 beats (on-beat or simple off-beats)
            strong_offset_beat_multiples = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] # Common in 4/4
            for i in range(tokenizer.num_offset_bins):
                offset_token_id = tokenizer.token_to_id.get(f"[OFFSET_BIN_{i}]")
                if offset_token_id is not None:
                    dequantized_offset = tokenizer._dequantize_flow_value(i, tokenizer.flow_offset_max_beats, tokenizer.num_offset_bins)
                    for strong_offset in strong_offset_beat_multiples:
                        if abs(dequantized_offset - strong_offset) < 0.1: logit_slice_for_masking[offset_token_id] += bias_val; break
        
        # Rhythmic Line Duration Bias
        if params.get('rhythmic_line_duration_bias_strength') is not None and current_gen_phase == "EXPECTING_DURATION_BIN":
            bias_val = params['rhythmic_line_duration_bias_strength']
            rhythmic_line_dur_multiples = [1.0, 2.0, 3.0, 4.0] # Simpler, more common line durations in beats
            for i in range(tokenizer.num_duration_bins):
                dur_token_id = tokenizer.token_to_id.get(f"[DURATION_BIN_{i}]")
                if dur_token_id is not None:
                    dequantized_dur = tokenizer._dequantize_flow_value(i, tokenizer.flow_duration_max_beats, tokenizer.num_duration_bins)
                    if dequantized_dur < 0.25: continue # Ignore very short, likely unmusical durations
                    for rhythmic_dur in rhythmic_line_dur_multiples:
                        if abs(dequantized_dur - rhythmic_dur) < 0.25: logit_slice_for_masking[dur_token_id] += bias_val; break
        
        # --- Grammar Masking ---
        grammar_mask = torch.ones_like(logit_slice_for_masking, dtype=torch.bool) # Mask everything initially
        allowed_ids: List[int] = []
        if current_gen_phase == "EXPECTING_SYLLABLES_COUNT": allowed_ids = tokenizer.get_token_ids_for_category("[SYLLABLES_")
        elif current_gen_phase == "EXPECTING_OFFSET_BIN": allowed_ids = tokenizer.get_token_ids_for_category("[OFFSET_BIN_")
        elif current_gen_phase == "EXPECTING_DURATION_BIN": allowed_ids = tokenizer.get_token_ids_for_category("[DURATION_BIN_")
        elif current_gen_phase == "EXPECTING_SYLLABLE_START":
            if target_syllables_for_current_line > 0 and syllables_generated_for_current_line < target_syllables_for_current_line:
                allowed_ids = tokenizer.get_token_ids_for_category("[SYLLABLE_STARTS_SUBDIV_")
            else: # Reached target syllables for this line
                allowed_ids = [tokenizer.end_syllable_sequence_token_id]
        elif current_gen_phase == "EXPECTING_SYLLABLE_DURATION": allowed_ids = tokenizer.get_token_ids_for_category("[SYLLABLE_DURATION_BIN_")
        elif current_gen_phase == "EXPECTING_SYLLABLE_STRESS": allowed_ids = tokenizer.get_token_ids_for_category("[SYLLABLE_STRESS_")
        elif current_gen_phase == "EXPECTING_END_SYLLABLE_SEQUENCE" or current_gen_phase == "EXPECTING_END_SYLLABLE_SEQUENCE_DIRECTLY":
            allowed_ids = [tokenizer.end_syllable_sequence_token_id]
        elif current_gen_phase == "EXPECTING_LINE_START_OR_BAR_START":
            allowed_ids = [tokenizer.line_start_token_id, tokenizer.bar_start_token_id]
            # Heuristic: if max lines per bar reached, strongly prefer bar_start
            max_lines_h = params.get('max_lines_per_bar_heuristic', 3) # Default to 3 if not set
            if lines_generated_for_current_bar >= max_lines_h :
                if tokenizer.line_start_token_id in allowed_ids: logit_slice_for_masking[tokenizer.line_start_token_id] -= 10.0 # Heavily penalize new line
                if tokenizer.bar_start_token_id in allowed_ids: logit_slice_for_masking[tokenizer.bar_start_token_id] += 5.0
        
        if allowed_ids: # Unmask allowed tokens
            temp_mask = torch.ones_like(logit_slice_for_masking, dtype=torch.bool); temp_mask[allowed_ids] = False; grammar_mask = temp_mask
        else: # If no specific tokens allowed by current phase (should be rare), allow EOS
            grammar_mask[tokenizer.eos_token_id] = False 
        
        grammar_mask[tokenizer.pad_token_id] = True # Always mask PAD
        logit_slice_for_masking[grammar_mask] = -float('Inf') # Apply grammar mask

        # --- Sampling ---
        final_logits = logit_slice_for_masking / params['temperature']
        if params['top_k'] > 0:
            v, _ = torch.topk(final_logits, min(params['top_k'], final_logits.size(-1)))
            final_logits[final_logits < v[[-1]]] = -float('Inf')
        probs = F.softmax(final_logits, dim=-1)
        
        # Check for non-finite probabilities (can happen if all logits are -inf after masking/biasing)
        if not (torch.isfinite(probs).all()):
            print(f"Warning: Non-finite probabilities at step {step_count}, phase {current_gen_phase}. Logits min/max: {final_logits.min().item()}, {final_logits.max().item()}");
            # Fallback: try to pick EOS if allowed, otherwise break if stuck
            next_token_id = tokenizer.eos_token_id # Default to EOS
            if grammar_mask[tokenizer.eos_token_id].item(): # If EOS itself is masked (bad state)
                print("CRITICAL: EOS token masked and probabilities non-finite. Stopping generation."); break
        else: next_token_id = torch.multinomial(probs, num_samples=1).item()
            
        # --- Update Context and State ---
        next_token_str = tokenizer.id_to_token.get(next_token_id, f"[UNK_ID:{next_token_id}]")
        next_seg_id, next_intra_pos_id = get_next_context_ids_for_token(full_song_context_token_ids, next_token_id, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
        full_song_context_token_ids.append(next_token_id); full_song_context_segment_ids.append(next_seg_id); full_song_context_intra_pos_ids.append(next_intra_pos_id)
        generated_flow_tokens_for_song.append(next_token_id)

        # --- State Machine for Generation Phases ---
        if next_token_id == tokenizer.eos_token_id: print("  [EOS] token generated. Stopping."); break
        if next_token_id == tokenizer.bar_start_token_id:
            current_bar_being_generated_flow_for += 1; lines_generated_for_current_bar = 0 # Reset for new bar
            consecutive_short_syllable_lines = 0 # Reset for new bar
            if current_bar_being_generated_flow_for >= max_song_bars:
                print(f"  Reached max song bars ({max_song_bars}). Stopping.");
                if generated_flow_tokens_for_song[-1] == tokenizer.bar_start_token_id: generated_flow_tokens_for_song.pop() # Remove trailing bar_start
                if generated_flow_tokens_for_song[-1] != tokenizer.eos_token_id: generated_flow_tokens_for_song.append(tokenizer.eos_token_id)
                break
            # Inject beat features for the new bar into the context
            next_bar_features = next((bf for bf in all_song_beat_features if bf.get("bar_index") == current_bar_being_generated_flow_for), None)
            if not next_bar_features: # If no specific features, use last known or a default
                if all_song_beat_features:
                    last_available_bar_feat = all_song_beat_features[-1]; next_bar_features = last_available_bar_feat.copy(); next_bar_features['bar_index'] = current_bar_being_generated_flow_for
                else: print(f"  Critical: No beat features for bar {current_bar_being_generated_flow_for} and no previous features to adapt. Stopping."); generated_flow_tokens_for_song.append(tokenizer.eos_token_id); break
            
            bar_tokens_for_guidance = tokenizer.encode_bar_features(next_bar_features)[1:] # Skip the [BAR_START] already emitted
            for tok_id in bar_tokens_for_guidance:
                seg_id, intra_id = get_next_context_ids_for_token(full_song_context_token_ids, tok_id, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
                full_song_context_token_ids.append(tok_id); full_song_context_segment_ids.append(seg_id); full_song_context_intra_pos_ids.append(intra_id)
            # Add separator and line_start for the new bar's flow
            sep_tok_guidance = tokenizer.sep_input_flow_token_id
            seg_id, intra_id = get_next_context_ids_for_token(full_song_context_token_ids, sep_tok_guidance, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
            full_song_context_token_ids.append(sep_tok_guidance); full_song_context_segment_ids.append(seg_id); full_song_context_intra_pos_ids.append(intra_id)
            line_start_tok_guidance = tokenizer.line_start_token_id
            seg_id, intra_id = get_next_context_ids_for_token(full_song_context_token_ids, line_start_tok_guidance, tokenizer, gpt_config.max_segment_types, gpt_config.max_intra_line_positions)
            full_song_context_token_ids.append(line_start_tok_guidance); full_song_context_segment_ids.append(seg_id); full_song_context_intra_pos_ids.append(intra_id)
            generated_flow_tokens_for_song.append(line_start_tok_guidance) # This is the start of the flow for the new bar
            
            current_gen_phase = "EXPECTING_SYLLABLES_COUNT"; lines_generated_for_current_bar = 1; syllables_generated_for_current_line = 0; target_syllables_for_current_line = -1
        elif next_token_id == tokenizer.line_start_token_id:
            lines_generated_for_current_bar += 1; syllables_generated_for_current_line = 0; target_syllables_for_current_line = -1; current_gen_phase = "EXPECTING_SYLLABLES_COUNT"
        elif next_token_str.startswith("[SYLLABLES_"):
            val = tokenizer.get_value_from_special_token(next_token_str); target_syllables_for_current_line = val if val is not None else 0; current_gen_phase = "EXPECTING_OFFSET_BIN"
            # Update consecutive short line counter
            if target_syllables_for_current_line <= short_syllable_threshold_for_end_avoidance and target_syllables_for_current_line != 0: # Count 0 syl lines as non-short for this heuristic
                consecutive_short_syllable_lines += 1
            else:
                consecutive_short_syllable_lines = 0 # Reset if not a short line
        elif next_token_str.startswith("[OFFSET_BIN_"): current_gen_phase = "EXPECTING_DURATION_BIN"
        elif next_token_str.startswith("[DURATION_BIN_"):
            if target_syllables_for_current_line == 0: current_gen_phase = "EXPECTING_END_SYLLABLE_SEQUENCE_DIRECTLY" # Skip to end_syl for 0-syl lines
            elif target_syllables_for_current_line > 0 : current_gen_phase = "EXPECTING_SYLLABLE_START"
            else: current_gen_phase = "EXPECTING_END_SYLLABLE_SEQUENCE_DIRECTLY" 
        elif next_token_str.startswith("[SYLLABLE_STARTS_SUBDIV_"): current_gen_phase = "EXPECTING_SYLLABLE_DURATION"
        elif next_token_str.startswith("[SYLLABLE_DURATION_BIN_"): current_gen_phase = "EXPECTING_SYLLABLE_STRESS"
        elif next_token_str.startswith("[SYLLABLE_STRESS_"):
            syllables_generated_for_current_line += 1
            if syllables_generated_for_current_line >= target_syllables_for_current_line: current_gen_phase = "EXPECTING_END_SYLLABLE_SEQUENCE"
            else: current_gen_phase = "EXPECTING_SYLLABLE_START"
        elif next_token_id == tokenizer.end_syllable_sequence_token_id:
            current_gen_phase = "EXPECTING_LINE_START_OR_BAR_START"; syllables_generated_for_current_line = 0; target_syllables_for_current_line = -1
    
    # --- Decode Generated Flow Tokens ---
    print("\n--- Raw Generated Flow Tokens (Full Song) ---")
    decoded_tokens_for_print = [tokenizer.id_to_token.get(token_id, str(token_id)) for token_id in generated_flow_tokens_for_song]
    tokens_per_line_log = 15 # For readability
    for i in range(0, len(decoded_tokens_for_print), tokens_per_line_log): print(" ".join(decoded_tokens_for_print[i:i+tokens_per_line_log]))
    print("--- End of Raw Generated Flow Tokens ---\n")

    decoded_flow_data_full_song: FlowData = []
    idx_in_gen_flow_tokens = 0
    decoding_bar_idx = context_bar_idx_start # Start decoding from the first generated bar
    decoding_line_idx_in_bar = 0
    # Ensure the generated flow starts with a line_start, as expected by decoder
    if generated_flow_tokens_for_song and generated_flow_tokens_for_song[0] == tokenizer.line_start_token_id: pass
    else: print("Warning: Generated flow tokens do not start with expected [LINE_START]. Decoding may be affected.")

    while idx_in_gen_flow_tokens < len(generated_flow_tokens_for_song):
        current_token_id = generated_flow_tokens_for_song[idx_in_gen_flow_tokens]
        if current_token_id == tokenizer.eos_token_id: break # End of generation
        
        # Handle bar transitions in the generated sequence
        if current_token_id == tokenizer.bar_start_token_id:
            decoding_bar_idx +=1; decoding_line_idx_in_bar = 0; idx_in_gen_flow_tokens += 1 # Move past [BAR_START]
            # Expect a [LINE_START] immediately after [BAR_START] to begin the new bar's flow
            if idx_in_gen_flow_tokens < len(generated_flow_tokens_for_song) and generated_flow_tokens_for_song[idx_in_gen_flow_tokens] == tokenizer.line_start_token_id:
                current_token_id = generated_flow_tokens_for_song[idx_in_gen_flow_tokens] # Set current_token_id to [LINE_START]
            else: # Unexpected token after [BAR_START], skip to next parsable unit or break
                continue # Continue scanning for a [LINE_START] or [BAR_START]
        
        # We must be at a [LINE_START] token to decode a FlowDatum
        if current_token_id != tokenizer.line_start_token_id:
            idx_in_gen_flow_tokens += 1; continue # Scan until a line_start is found
        
        # Collect tokens for one line (from [LINE_START] to [END_SYLLABLE_SEQUENCE])
        tokens_for_this_line_attempt = [] 
        temp_line_parser_idx = idx_in_gen_flow_tokens
        while temp_line_parser_idx < len(generated_flow_tokens_for_song):
            token_id = generated_flow_tokens_for_song[temp_line_parser_idx] 
            tokens_for_this_line_attempt.append(token_id) 
            if token_id == tokenizer.end_syllable_sequence_token_id: break # End of current line's flow info
            # Early exit if we hit EOS or another structural token prematurely
            if token_id == tokenizer.eos_token_id or \
               (token_id == tokenizer.bar_start_token_id and len(tokens_for_this_line_attempt) > 1) or \
               (token_id == tokenizer.line_start_token_id and len(tokens_for_this_line_attempt) > 1) :
                tokens_for_this_line_attempt.pop() # Remove the premature structural token
                temp_line_parser_idx -=1 
                break
            temp_line_parser_idx += 1
        
        datum = tokenizer.decode_flow_tokens_to_datum(tokens_for_this_line_attempt, decoding_bar_idx, decoding_line_idx_in_bar)
        if datum: decoded_flow_data_full_song.append(datum); decoding_line_idx_in_bar += 1
        
        idx_in_gen_flow_tokens = temp_line_parser_idx + 1 # Move to token after the processed line/sequence

    print(f"Total FlowDatum objects decoded: {len(decoded_flow_data_full_song)}")
    return decoded_flow_data_full_song, tokenizer, gpt_config

def save_flow_data_to_json(flow_data: FlowData, beat_info: BeatInfo, output_path: str):
    if not flow_data: print("No flow data to save."); return
    bpm = beat_info.get("bpm", 120.0)
    if bpm <=0: bpm = 120.0 # Fallback
    
    output_data_dict = { 
        "beat_info": { # Save key beat info used for flow generation / interpretation
            "bpm": bpm, 
            "beats_per_bar": beat_info.get("beats_per_bar", 4),
            "estimated_bar_duration_sec": beat_info.get("estimated_bar_duration", (60.0/bpm)*beat_info.get("beats_per_bar", 4))
        },
        "flow_lines": [] # Store FlowDatum items directly
    }
    for fd in flow_data:
        if not isinstance(fd, dict): continue # Should be FlowDatum dicts
        line_data = dict(fd) # Make a copy
        # Add an approximate duration in seconds for easier interpretation if needed
        line_data["approx_line_duration_sec"] = round(fd.get("duration_beats",0) * (60.0/bpm), 3)
        output_data_dict["flow_lines"].append(line_data) 
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f: json.dump(output_data_dict, f, indent=2) 
        print(f"Generated flow data saved to: {output_path}")
    except Exception as e: print(f"Error saving flow data to JSON: {e}")

def visualize_flow_rhythm(
    model_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    model_config_path: str = DEFAULT_MODEL_CONFIG_PATH,
    tokenizer_config_path: str = DEFAULT_TOKENIZER_CONFIG_PATH,
    instrumental_audio_path: str = DEFAULT_INSTRUMENTAL_PATH,
    instrumental_offset_sec: Optional[float] = None, # Offset within the instrumental
    instrumental_duration_sec: Optional[float] = None, # Duration of segment to process
    num_prompt_bars: int = 2, # How many bars of instrumental features to use as prompt
    output_filename_prefix: str = "flow_vis_audio",
    output_format: str = "wav", # "wav" or "mp3"
    flow_generation_params: Optional[Dict[str, Any]] = None,
    click_track_level_db: float = 0.0 # Gain for the click track
):
    print(f"--- Visualizing Flow Rhythm (Full Song Generation) ---")
    print(f"Using device: {DEVICE}\nModel: {model_checkpoint_path}\nInstrumental: {instrumental_audio_path}" + 
          (f" (Offset: {instrumental_offset_sec}s" if instrumental_offset_sec is not None else "") +
          (f", Duration: {instrumental_duration_sec}s" if instrumental_duration_sec is not None else "") + ")")
    
    # Define default generation parameters (can be overridden by flow_generation_params)
    # These are tailored for visualization - often good to test different rhythmic densities
    current_gen_params = { 
        "temperature": 0.7, "top_k": 50, "max_lines_per_bar_heuristic": 2,
        "syl_count_bias_strength": -1.0, # Fewer syllables -> sparser flow
        "long_syl_dur_bias_strength": 1.8, # Longer syllable durations -> slower feel
        "rhythmic_offset_bias_strength": 0.6,
        "rhythmic_line_duration_bias_strength": 0.6,
        "rhythmic_syllable_duration_bias_strength": 0.8,
        "avoid_short_lines_end_bias_strength": -2.0, # New default for visualizer
        "min_syllables_threshold": 5, # New default for visualizer
        "min_syllables_penalty_strength": 1.5 # New default for visualizer
    }
    if flow_generation_params: # User overrides
        for k,v in flow_generation_params.items():
            if v is not None: # Only override if user provided a non-None value
                current_gen_params[k] = v
            elif k in current_gen_params and v is None : # If user explicitly set to None (to turn off a default bias)
                 current_gen_params[k] = None # Allow user to nullify a default bias

    # The _generate_flow_core function now handles the None values internally for bias strengths,
    # so we can pass current_gen_params directly.
    final_gen_params_for_core = current_gen_params

    print(f"Generation params applied: {final_gen_params_for_core}")

    # --- Initialize Components ---
    audio_processor = AudioProcessor(); beat_feature_extractor = BeatFeatureExtractor(sample_rate=44100) # Standard SR for feature extraction
    
    # --- Load and Process Instrumental Segment ---
    if not os.path.exists(instrumental_audio_path): print(f"Error: Instrumental not found: {instrumental_audio_path}."); return
    target_sr_for_processing = 44100 # Consistent SR for all processing steps
    # Load the segment of the instrumental that will be used for prompting and as background
    instrumental_waveform_mono, loaded_sr = audio_processor.load_audio(
        instrumental_audio_path, target_sr=target_sr_for_processing, mono=True,
        offset_sec=instrumental_offset_sec, duration_sec=instrumental_duration_sec
    )
    if instrumental_waveform_mono is None or instrumental_waveform_mono.size == 0: print(f"Error: Could not load instrumental segment: {instrumental_audio_path}."); return
    if loaded_sr != target_sr_for_processing: print(f"Warning: Loaded SR {loaded_sr} != target SR {target_sr_for_processing}. Resampling occurred.")

    # Extract beat features for the *entire segment* (this will be used to guide generation beyond the prompt)
    all_beat_features_for_segment: SongBeatFeatures = beat_feature_extractor.extract_features_for_song(
        audio_path=instrumental_audio_path, # Path to original full audio
        stems_input_dir=None, # Assuming no stems for visualization simplicity
        audio_offset_sec=instrumental_offset_sec, # Analyze the specific segment
        audio_duration_sec=instrumental_duration_sec
    )
    if not all_beat_features_for_segment: print(f"Error: No beat features extracted for the instrumental segment."); return
    
    # Create a BeatInfo dict from these features for the click track generation
    beat_info_for_clicktrack: BeatInfo = _create_beat_info_from_custom_features(all_beat_features_for_segment)
    # Sanity check BPM from BeatInfo, fallback if necessary
    if beat_info_for_clicktrack.get("bpm",0) <=0:
        bi_direct = audio_processor.get_beat_info(instrumental_waveform_mono, target_sr_for_processing) # Analyze the loaded segment
        if bi_direct and bi_direct.get("bpm",0)>0: beat_info_for_clicktrack = bi_direct
        else: print(f"Error: Could not extract valid BeatInfo for click track generation."); return
    
    # --- Prepare Prompt for Flow Model ---
    num_prompt_bars_actual = min(num_prompt_bars, len(all_beat_features_for_segment))
    if num_prompt_bars_actual == 0 and len(all_beat_features_for_segment) > 0: num_prompt_bars_actual = 1 # Must have at least one prompt bar if features exist
    elif num_prompt_bars_actual == 0 and len(all_beat_features_for_segment) == 0: print("Error: Zero bars for prompt and no beat features available."); return
    
    prompt_bar_features_for_gen = all_beat_features_for_segment[:num_prompt_bars_actual]
    context_bar_idx_start_for_gen = prompt_bar_features_for_gen[-1]["bar_index"] + 1 if prompt_bar_features_for_gen else 0

    # --- Generate Flow ---
    decoded_flow_data, tokenizer_instance, _ = _generate_flow_core(
        prompt_bar_features=prompt_bar_features_for_gen,
        all_song_beat_features=all_beat_features_for_segment, # Guide generation for the whole segment
        model_checkpoint_path=model_checkpoint_path,
        model_config_path=model_config_path,
        tokenizer_config_path=tokenizer_config_path,
        context_bar_idx_start=context_bar_idx_start_for_gen,
        output_name_suffix=f"from_{os.path.splitext(os.path.basename(instrumental_audio_path))[0]}", # For any internal logging if _generate_flow_core had it
        generation_params=final_gen_params_for_core # Pass the potentially modified params
    )
    if not tokenizer_instance: print("Tokenizer loading failed during flow generation."); return
    
    print("\nDecoded FlowData (Full Song Generation attempt):")
    if decoded_flow_data:
        for i, fd in enumerate(decoded_flow_data):
            if not isinstance(fd, dict): continue
            print(f"  Line {i+1}: Bar {fd['bar_index']}, LineInBar {fd['line_index_in_bar']}, Syls(token): {fd.get('syllables', 'N/A')}, Syls(actual): {len(fd.get('syllable_start_subdivisions', []))}, Offset {fd.get('start_offset_beats','N/A'):.2f}b, Dur {fd.get('duration_beats','N/A'):.2f}b")
        # Save the generated flow data to JSON for inspection
        safe_instrumental_name_json = "".join(c if c.isalnum() else '_' for c in os.path.splitext(os.path.basename(instrumental_audio_path))[0])
        offset_suffix_json = f"_offset{instrumental_offset_sec:.0f}s" if instrumental_offset_sec is not None else ""
        duration_suffix_json = f"_dur{instrumental_duration_sec:.0f}s" if instrumental_duration_sec is not None else ""
        flow_json_filename = f"generated_flow_{safe_instrumental_name_json}{offset_suffix_json}{duration_suffix_json}.json"; flow_json_output_path = os.path.join(FLOW_DATA_OUTPUT_DIR, flow_json_filename)
        save_flow_data_to_json(decoded_flow_data, beat_info_for_clicktrack, flow_json_output_path)
    else: print("  No valid flow data decoded for the song.")

    # --- Generate Click Track Audio ---
    click_track_audio, click_sr = generate_syllable_click_track(
        flow_data=decoded_flow_data if decoded_flow_data else [], # Pass empty list if no flow
        beat_info=beat_info_for_clicktrack,
        tokenizer=tokenizer_instance,
        instrumental_waveform_mono=instrumental_waveform_mono, # Use the loaded segment
        sample_rate=target_sr_for_processing,
        vocal_click_level_db=click_track_level_db
    )

    # --- Save Output Audio ---
    if click_track_audio.size > 0:
        safe_checkpoint_name = "".join(c if c.isalnum() else '_' for c in os.path.basename(model_checkpoint_path).replace('.pt','')); safe_instrumental_name = "".join(c if c.isalnum() else '_' for c in os.path.splitext(os.path.basename(instrumental_audio_path))[0])
        offset_suffix = f"_offset{instrumental_offset_sec:.0f}s" if instrumental_offset_sec is not None else ""; duration_suffix = f"_dur{instrumental_duration_sec:.0f}s" if instrumental_duration_sec is not None else ""
        
        # Construct filename part from bias strengths for clarity
        bias_str_parts = []; gen_p = final_gen_params_for_core # Use the final params for filename
        if gen_p.get("syl_count_bias_strength"): bias_str_parts.append(f"scb{gen_p['syl_count_bias_strength']:.1f}")
        if gen_p.get("long_syl_dur_bias_strength"): bias_str_parts.append(f"lsdb{gen_p['long_syl_dur_bias_strength']:.1f}")
        if gen_p.get("rhythmic_syllable_duration_bias_strength"): bias_str_parts.append(f"rsdb{gen_p['rhythmic_syllable_duration_bias_strength']:.1f}")
        if gen_p.get("avoid_short_lines_end_bias_strength"): bias_str_parts.append(f"asleb{gen_p['avoid_short_lines_end_bias_strength']:.1f}")
        if gen_p.get("min_syllables_penalty_strength"): bias_str_parts.append(f"msp{gen_p['min_syllables_penalty_strength']:.1f}")

        bias_filename_part = ("_bias_" + "_".join(bias_str_parts)) if bias_str_parts else ""
        
        output_audio_filename = f"{output_filename_prefix}_{safe_checkpoint_name}_on_{safe_instrumental_name}{offset_suffix}{duration_suffix}{bias_filename_part}_prompt{num_prompt_bars_actual}.{output_format}"
        output_audio_path = os.path.join(OUTPUT_DIR, output_audio_filename)
        try: sf.write(output_audio_path, click_track_audio, click_sr); print(f"\nFlow rhythm visualization audio saved to: {output_audio_path}")
        except Exception as e_sf: print(f"Error saving audio with soundfile: {e_sf}")
    elif decoded_flow_data: print("\nFlow data decoded, but generated click track audio is empty.")
    else: print("\nNo flow data decoded, and no visualization audio generated.")

if __name__ == "__main__":
    print("---BeefAI Flow Visualizer---")
    main_checkpoint_path = DEFAULT_CHECKPOINT_PATH; main_model_config = DEFAULT_MODEL_CONFIG_PATH; main_tokenizer_config = DEFAULT_TOKENIZER_CONFIG_PATH
    main_instrumental_audio = DEFAULT_INSTRUMENTAL_PATH; main_instrumental_offset = DEFAULT_INSTRUMENTAL_OFFSET_SEC; main_instrumental_duration = DEFAULT_INSTRUMENTAL_DURATION_SEC 
    
    # Example of using more specific flow generation parameters for visualization
    # These defaults are now set inside visualize_flow_rhythm, but can be overridden here
    test_flow_gen_params_slower_defaults = { 
        "temperature": 0.7, "top_k": 50, "max_lines_per_bar_heuristic": 2,
        "syl_count_bias_strength": -1.0, # Fewer syllables
        "long_syl_dur_bias_strength": 1.8, # Longer syllable durations
        "rhythmic_offset_bias_strength": 0.6,
        "rhythmic_line_duration_bias_strength": 0.6,
        "rhythmic_syllable_duration_bias_strength": 0.8,
        "avoid_short_lines_end_bias_strength": -2.0, # Explicitly set for testing
        "min_syllables_threshold": 5,
        "min_syllables_penalty_strength": 1.5
    }
    chosen_params = test_flow_gen_params_slower_defaults # Or None to use function's internal defaults
    
    if not os.path.exists(main_checkpoint_path): print(f"ERROR: Model checkpoint '{main_checkpoint_path}' not found.")
    else: visualize_flow_rhythm( model_checkpoint_path=main_checkpoint_path, model_config_path=main_model_config, tokenizer_config_path=main_tokenizer_config, instrumental_audio_path=main_instrumental_audio, instrumental_offset_sec=main_instrumental_offset, instrumental_duration_sec=main_instrumental_duration, output_format="wav", num_prompt_bars=2, flow_generation_params=chosen_params, click_track_level_db = -3.0 ) # Clicks are slightly quieter than instrumental peak
    print("-" * 50)