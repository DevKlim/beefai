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
from beefai.data_processing.audio_processor import AudioProcessor # For BeatInfo
from beefai.data_processing.beat_feature_extractor import BeatFeatureExtractor # To get BarBeatFeatures for prompt
from beefai.utils.data_types import BeatInfo, FlowData, FlowDatum, BarBeatFeatures, SongBeatFeatures

# --- Configuration ---
# Default to FULL model and a common instrumental path (user should ensure it exists)
DEFAULT_MODEL_CONFIG_PATH = "lite_model_training/model_config_full.yaml"
DEFAULT_TOKENIZER_CONFIG_PATH = "beefai/flow_model/flow_tokenizer_config_v2.json" 
DEFAULT_CHECKPOINT_PATH = "data/checkpoints/flow_model_full/full_final_model.pt" 
DEFAULT_INSTRUMENTAL_PATH = "data/instrumentals/Alright.mp3" # Example: Use an actual instrumental from your dataset
OUTPUT_DIR = "output/flow_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_syllable_click_track(
    flow_data: FlowData, 
    beat_info: BeatInfo, 
    instrumental_waveform: Optional[np.ndarray] = None, 
    sample_rate: int = 44100,
    click_freq: float = 1000.0, # Frequency of the click sound
    click_duration_sec: float = 0.02 # Duration of the click sound
) -> Tuple[np.ndarray, int]:
    """
    Generates an audio track with clicks at each syllable's start time,
    potentially overlaid on an instrumental.

    Args:
        flow_data: List of FlowDatum, containing syllable_start_subdivisions.
        beat_info: BeatInfo for BPM and bar timing.
        instrumental_waveform: Optional original instrumental to overlay clicks on.
        sample_rate: Sample rate for the output audio.
        click_freq: Frequency of the click sound.
        click_duration_sec: Duration of the click sound.

    Returns:
        Tuple (waveform, sample_rate)
    """
    if not flow_data:
        print("No flow data provided to generate click track.")
        return (np.array([], dtype=np.float32), sample_rate)

    bpm = beat_info.get("bpm")
    if not bpm or bpm <=0: # Added bpm <= 0 check
        print(f"Error: BPM not found or invalid ({bpm}) in beat_info. Cannot calculate timings.")
        return (np.array([], dtype=np.float32), sample_rate)
    
    beats_per_bar = beat_info.get("beats_per_bar", 4)
    if beats_per_bar <=0: # Added check for valid beats_per_bar
        print(f"Error: Invalid beats_per_bar ({beats_per_bar}) in beat_info. Assuming 4.")
        beats_per_bar = 4

    beat_duration_sec = 60.0 / bpm
    # Assuming 16th note subdivisions (4 per beat if beats_per_bar is 4, or more generally subdivisions_per_beat = 4)
    subdivisions_per_beat = tokenizer.max_subdivisions / beats_per_bar if hasattr(tokenizer, 'max_subdivisions') else 4 # Use tokenizer's max_subdivisions if available
    subdivision_duration_sec = beat_duration_sec / subdivisions_per_beat


    # Determine absolute bar start times
    bar_absolute_start_times_sec: Dict[int, float] = {}
    if beat_info.get("downbeat_times"):
        for i, dt in enumerate(beat_info["downbeat_times"]):
            bar_absolute_start_times_sec[i] = dt
    elif beat_info.get("estimated_bar_duration") and beat_info["estimated_bar_duration"] > 0:
        bar_dur = beat_info["estimated_bar_duration"]
        # Estimate number of bars needed from flow_data
        max_bar_idx = 0
        if flow_data:
            max_bar_idx = max(fd.get("bar_index", 0) for fd in flow_data) if flow_data else 0
        for i in range(max_bar_idx + 4): # Buffer, increased for safety
            bar_absolute_start_times_sec[i] = i * bar_dur
    else:
        print("Error: Cannot determine bar start times from beat_info (missing downbeat_times and valid estimated_bar_duration).")
        return (np.array([], dtype=np.float32), sample_rate)

    # Calculate total duration needed
    max_event_time_sec = 0
    if flow_data: # Check if flow_data is not empty
        for fd in flow_data:
            bar_idx = fd.get("bar_index")
            if bar_idx is None or bar_idx not in bar_absolute_start_times_sec:
                print(f"Warning: FlowDatum for bar_index {bar_idx} skipped, bar start time unknown.")
                continue
            bar_start_sec = bar_absolute_start_times_sec[bar_idx]
            
            syllable_subdivisions = fd.get("syllable_start_subdivisions", [])
            if syllable_subdivisions:
                latest_syllable_sub_div_time_in_bar = max(syllable_subdivisions) * subdivision_duration_sec
                max_event_time_sec = max(max_event_time_sec, bar_start_sec + latest_syllable_sub_div_time_in_bar + click_duration_sec)
            else: # If no subdivisions, consider the line's nominal duration for overall length
                line_start_offset_sec = fd.get("start_offset_beats", 0) * beat_duration_sec
                line_duration_sec = fd.get("duration_beats", 0) * beat_duration_sec
                max_event_time_sec = max(max_event_time_sec, bar_start_sec + line_start_offset_sec + line_duration_sec)
    else: # No flow data, so max_event_time_sec remains 0
        pass


    if instrumental_waveform is not None:
        output_duration_sec = max(max_event_time_sec, len(instrumental_waveform) / sample_rate)
        output_waveform = np.copy(instrumental_waveform)
        required_samples = int(output_duration_sec * sample_rate)
        if len(output_waveform) < required_samples:
            padding = np.zeros(required_samples - len(output_waveform), dtype=np.float32)
            output_waveform = np.concatenate((output_waveform, padding))
        elif len(output_waveform) > required_samples: 
            output_waveform = output_waveform[:required_samples]

    else:
        output_duration_sec = max_event_time_sec + 1.0 
        output_waveform = np.zeros(int(output_duration_sec * sample_rate), dtype=np.float32)
        if output_duration_sec <= 1.0 and not flow_data : 
             output_waveform = np.zeros(sample_rate, dtype=np.float32) 


    click_samples = int(click_duration_sec * sample_rate)
    t_click = np.linspace(0, click_duration_sec, click_samples, endpoint=False)
    click_signal = 0.5 * np.sin(2 * np.pi * click_freq * t_click)
    fade_len = min(click_samples // 4, int(0.005 * sample_rate)) 
    if fade_len > 1: 
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        click_signal[:fade_len] *= fade_in
        click_signal[-fade_len:] *= fade_out

    num_syllables_placed = 0
    for fd_idx, fd in enumerate(flow_data):
        bar_idx = fd.get("bar_index")
        syllable_subdivisions = fd.get("syllable_start_subdivisions", [])

        if bar_idx is None or bar_idx not in bar_absolute_start_times_sec:
            print(f"Warning: Skipping FlowDatum {fd_idx} due to missing bar_index or bar timing info for bar {bar_idx}.")
            continue
        
        bar_start_sec = bar_absolute_start_times_sec[bar_idx]

        for sub_div_idx in syllable_subdivisions:
            syllable_time_in_bar_sec = sub_div_idx * subdivision_duration_sec 
            abs_syllable_time_sec = bar_start_sec + syllable_time_in_bar_sec
            
            start_sample = int(abs_syllable_time_sec * sample_rate)
            end_sample = start_sample + click_samples

            if end_sample <= len(output_waveform):
                output_waveform[start_sample:end_sample] += click_signal
                num_syllables_placed +=1
            else:
                print(f"Warning: Syllable click for FlowDatum {fd_idx}, subdiv {sub_div_idx} at {abs_syllable_time_sec:.2f}s (sample {start_sample}) extends beyond waveform duration ({len(output_waveform)/sample_rate:.2f}s, total samples {len(output_waveform)}).")
    
    print(f"Placed {num_syllables_placed} syllable clicks in the audio.")
    if num_syllables_placed == 0 and flow_data:
        print("Warning: No syllable clicks were placed. Check flow_data structure (syllable_start_subdivisions), beat_info, and timings.")

    return (output_waveform, sample_rate)


def visualize_flow_rhythm(
    model_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    model_config_path: str = DEFAULT_MODEL_CONFIG_PATH,
    tokenizer_config_path: str = DEFAULT_TOKENIZER_CONFIG_PATH,
    instrumental_audio_path: str = DEFAULT_INSTRUMENTAL_PATH,
    num_prompt_bars: int = 2, 
    max_generated_flow_lines: int = 8, 
    output_filename_prefix: str = "flow_vis"
):
    print(f"--- Visualizing Flow Rhythm ---")
    print(f"Using device: {DEVICE}")
    print(f"Model checkpoint: {model_checkpoint_path}")
    print(f"Instrumental: {instrumental_audio_path}")

    audio_processor = AudioProcessor()
    beat_feature_extractor = BeatFeatureExtractor() 

    if not os.path.exists(instrumental_audio_path):
        print(f"Error: Instrumental audio file not found at {instrumental_audio_path}. Exiting.")
        return

    instrumental_sr = beat_feature_extractor.sample_rate 
    instrumental_waveform, loaded_sr = audio_processor.load_audio(instrumental_audio_path, target_sr=instrumental_sr)
    if instrumental_waveform.size == 0:
        print(f"Error: Could not load instrumental audio from {instrumental_audio_path}. Exiting.")
        return
    if loaded_sr != instrumental_sr: 
        print(f"Warning: Loaded SR {loaded_sr} differs from target SR {instrumental_sr}. This might affect processing.")
    
    beat_info: BeatInfo = audio_processor.get_beat_info(instrumental_waveform, instrumental_sr)
    if not beat_info or not beat_info.get("bpm") or beat_info.get("bpm") <=0:
        print(f"Error: Could not extract valid BeatInfo (BPM: {beat_info.get('bpm')}) from instrumental: {instrumental_audio_path}. Exiting.")
        return
    
    print(f"Instrumental Beat Info: BPM={beat_info['bpm']:.2f}, Estimated Bars (from downbeats): {len(beat_info.get('downbeat_times',[]))}, Beats per bar: {beat_info.get('beats_per_bar', 'N/A')}")

    if not os.path.exists(tokenizer_config_path):
        print(f"Error: Tokenizer config not found at {tokenizer_config_path}. Exiting.")
        return
    
    # Make tokenizer global or pass to generate_syllable_click_track if needed for subdivisions_per_beat
    global tokenizer
    tokenizer = FlowTokenizer(config_path=tokenizer_config_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.pad_token_id

    if not os.path.exists(model_config_path):
        print(f"Error: Model config not found at {model_config_path}. Exiting.")
        return
    model_yaml_config = load_yaml_config(model_config_path)
    
    gpt_config = FlowGPTConfig(
        vocab_size=vocab_size,
        block_size=model_yaml_config["block_size"],
        n_layer=model_yaml_config["n_layer"],
        n_head=model_yaml_config["n_head"],
        n_embd=model_yaml_config["n_embd"],
        max_segment_types=model_yaml_config["max_segment_types"],
        max_intra_line_positions=model_yaml_config["max_intra_line_positions"],
        dropout=model_yaml_config["dropout"],
        bias=model_yaml_config.get("bias", True),
        pad_token_id=pad_token_id
    )
    model = FlowTransformerDecoder(gpt_config)
    
    if not os.path.exists(model_checkpoint_path):
        print(f"Error: Model checkpoint not found at {model_checkpoint_path}. Exiting.")
        return
        
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        elif 'state_dict' in checkpoint: 
             model.load_state_dict(checkpoint['state_dict'])
        else: 
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model checkpoint from {model_checkpoint_path}: {e}")
        print("Ensure the checkpoint is compatible with the current model architecture.")
        return
        
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {model_checkpoint_path} and set to eval mode.")

    song_id_for_stems = os.path.splitext(os.path.basename(instrumental_audio_path))[0]
    _default_pre_separated_stems_base_dir_from_preprocess = "data/temp_demucs_separated" 
    _default_demucs_model_name_from_preprocess = "htdemucs_ft" 
    
    stems_input_dir_for_song = os.path.join(
        _default_pre_separated_stems_base_dir_from_preprocess, 
        _default_demucs_model_name_from_preprocess, 
        song_id_for_stems
    )
    
    print(f"Attempting to use pre-separated stems from: {stems_input_dir_for_song}")
    if not os.path.isdir(stems_input_dir_for_song):
        print(f"Warning: Stems directory not found at {stems_input_dir_for_song}. BeatFeatureExtractor will simulate from full mix or use empty features for stems.")
        stems_input_dir_for_song = None 


    print(f"Extracting beat features for the first {num_prompt_bars} bar(s) as prompt...")
    # CORRECTED METHOD NAME HERE:
    song_beat_features: SongBeatFeatures = beat_feature_extractor.extract_features_for_song(
        audio_path=instrumental_audio_path,
        stems_input_dir=stems_input_dir_for_song 
    )

    if not song_beat_features:
        print(f"Error: No beat features extracted from {instrumental_audio_path}. Cannot create prompt. Exiting.")
        return
    if len(song_beat_features) < num_prompt_bars:
        print(f"Warning: Not enough bar features extracted ({len(song_beat_features)}) to form a full prompt of {num_prompt_bars} bars. Using all available {len(song_beat_features)} bars.")
        num_prompt_bars = len(song_beat_features)
        if num_prompt_bars == 0:
            print("Error: Zero bars available for prompt after feature extraction. Exiting.")
            return
    
    prompt_bar_features = song_beat_features[:num_prompt_bars]

    prompt_token_ids_list: List[int] = [tokenizer.bos_token_id]
    prompt_segment_ids_list: List[int] = [0] 
    prompt_intra_line_pos_ids_list: List[int] = [0]
    
    current_input_bar_count_for_segment_id = 0 
    
    for bar_idx_in_prompt, bar_features in enumerate(prompt_bar_features):
        bar_feature_tokens = tokenizer.encode_bar_features(bar_features)
        seg_id_for_beat_block = current_input_bar_count_for_segment_id * 2
        prompt_token_ids_list.extend(bar_feature_tokens)
        prompt_segment_ids_list.extend([seg_id_for_beat_block] * len(bar_feature_tokens))
        prompt_intra_line_pos_ids_list.extend(list(range(len(bar_feature_tokens)))) 
        
        seg_id_for_sep_and_flow = seg_id_for_beat_block + 1
        prompt_token_ids_list.append(tokenizer.sep_input_flow_token_id)
        prompt_segment_ids_list.append(seg_id_for_sep_and_flow)
        prompt_intra_line_pos_ids_list.append(0) 

        current_input_bar_count_for_segment_id += 1
    
    line_start_seg_id, line_start_intra_pos_id = get_next_context_ids_for_token(
        prompt_token_ids_list, 
        tokenizer.line_start_token_id, 
        tokenizer,
        gpt_config.max_segment_types,
        gpt_config.max_intra_line_positions
    )

    prompt_token_ids_list.append(tokenizer.line_start_token_id)
    prompt_segment_ids_list.append(line_start_seg_id)
    prompt_intra_line_pos_ids_list.append(line_start_intra_pos_id)

    idx_prompt_tensor = torch.tensor([prompt_token_ids_list], dtype=torch.long, device=DEVICE)
    seg_ids_prompt_tensor = torch.tensor([prompt_segment_ids_list], dtype=torch.long, device=DEVICE)
    intra_pos_ids_prompt_tensor = torch.tensor([prompt_intra_line_pos_ids_list], dtype=torch.long, device=DEVICE)

    print(f"Prompt (len {idx_prompt_tensor.size(1)}): {' '.join(tokenizer.id_to_token.get(tid, '[UNK]') for tid in prompt_token_ids_list[:70])}...") 

    avg_tokens_per_line = tokenizer.max_syllables + 3 + (tokenizer.max_subdivisions // 2) 
    max_new_tokens = max_generated_flow_lines * avg_tokens_per_line + 20 
    
    print(f"Generating up to {max_new_tokens} new tokens for flow...")
    with torch.no_grad():
        generated_ids_full = model.generate(
            idx_prompt_tensor,
            seg_ids_prompt_tensor,
            intra_pos_ids_prompt_tensor,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer, 
            temperature=0.7,    
            top_k=50 
        )
    
    start_of_model_generation_idx = len(prompt_token_ids_list)
    model_generated_tokens = generated_ids_full[0, start_of_model_generation_idx:].tolist()
    
    print(f"Model generated {len(model_generated_tokens)} flow tokens: {' '.join(tokenizer.id_to_token.get(tid, '[UNK]') for tid in model_generated_tokens[:60])}...")

    decoded_flow_data: FlowData = []
    current_bar_idx_for_flow_context = prompt_bar_features[-1]["bar_index"] if prompt_bar_features else 0
    line_idx_in_bar_counter_for_this_context = 0 
    
    idx_in_model_generated_tokens = 0
    while idx_in_model_generated_tokens < len(model_generated_tokens):
        tokens_for_this_line_attempt: List[int] = []
        
        if not decoded_flow_data: 
            tokens_for_this_line_attempt.append(prompt_token_ids_list[-1]) 
        else: 
            if model_generated_tokens[idx_in_model_generated_tokens] == tokenizer.line_start_token_id:
                tokens_for_this_line_attempt.append(model_generated_tokens[idx_in_model_generated_tokens])
                idx_in_model_generated_tokens += 1 
            else:
                break 
        
        while idx_in_model_generated_tokens < len(model_generated_tokens):
            token_id = model_generated_tokens[idx_in_model_generated_tokens]
            if token_id == tokenizer.eos_token_id or \
               token_id == tokenizer.bar_start_token_id or \
               token_id == tokenizer.line_start_token_id: 
                break 
            tokens_for_this_line_attempt.append(token_id)
            idx_in_model_generated_tokens += 1
        
        if tokens_for_this_line_attempt and tokens_for_this_line_attempt[0] == tokenizer.line_start_token_id:
            datum = tokenizer.decode_flow_tokens_to_datum(
                tokens_for_this_line_attempt, 
                bar_idx_context=current_bar_idx_for_flow_context, 
                line_idx_context=line_idx_in_bar_counter_for_this_context
            )
            if datum:
                decoded_flow_data.append(datum)
                line_idx_in_bar_counter_for_this_context += 1
                if idx_in_model_generated_tokens < len(model_generated_tokens) and \
                   model_generated_tokens[idx_in_model_generated_tokens] == tokenizer.bar_start_token_id:
                    current_bar_idx_for_flow_context += 1
                    line_idx_in_bar_counter_for_this_context = 0 
                    idx_in_model_generated_tokens += 1 
                    if idx_in_model_generated_tokens < len(model_generated_tokens) and \
                       model_generated_tokens[idx_in_model_generated_tokens] == tokenizer.line_start_token_id:
                        pass
                    else:
                        break
            else:
                break
        else:
            break

        if idx_in_model_generated_tokens < len(model_generated_tokens) and \
           model_generated_tokens[idx_in_model_generated_tokens] == tokenizer.eos_token_id:
            break

    print("\nDecoded FlowData:")
    if decoded_flow_data:
        for i, fd in enumerate(decoded_flow_data):
            print(f"  Line {i+1}: Bar {fd['bar_index']}, LineInBar {fd['line_index_in_bar']}, Syls {fd['syllables']}, "
                  f"Offset {fd['start_offset_beats']:.2f}b, Dur {fd['duration_beats']:.2f}b, "
                  f"Subdivisions: {fd.get('syllable_start_subdivisions')}")
    else:
        print("  No valid flow data decoded from generated tokens.")
        if model_generated_tokens:
            print("  This might indicate the model generated EOS or BAR_START immediately after the initial primed LINE_START, "
                  "or the generated sequence for the first line was not decodable (e.g., missing syllable/offset/duration tokens), "
                  "or no further LINE_START tokens were generated if expecting multiple lines.")
        return

    click_track_audio, click_sr = generate_syllable_click_track(
        flow_data=decoded_flow_data,
        beat_info=beat_info,
        instrumental_waveform=instrumental_waveform, 
        sample_rate=instrumental_sr 
    )

    if click_track_audio.size > 0:
        checkpoint_basename = os.path.basename(model_checkpoint_path)
        safe_checkpoint_name = "".join(c if c.isalnum() or c in ['_','.'] else '_' for c in checkpoint_basename).replace('.pt','')
        
        instrumental_basename = os.path.basename(instrumental_audio_path)
        safe_instrumental_name = "".join(c if c.isalnum() or c in ['_','.'] else '_' for c in instrumental_basename)
        safe_instrumental_name = os.path.splitext(safe_instrumental_name)[0] 

        output_audio_filename = f"{output_filename_prefix}_{safe_checkpoint_name}_on_{safe_instrumental_name}_bars{num_prompt_bars}.wav"
        output_audio_path = os.path.join(OUTPUT_DIR, output_audio_filename)
        sf.write(output_audio_path, click_track_audio, click_sr)
        print(f"\nFlow rhythm visualization audio saved to: {output_audio_path}")
        print(f"This track contains the instrumental with clicks at predicted syllable landings.")
    else:
        print("\nFailed to generate flow rhythm visualization audio (click track was empty).")

if __name__ == "__main__":
    test_checkpoint = DEFAULT_CHECKPOINT_PATH 

    if not os.path.exists(test_checkpoint):
        print(f"Error: Test checkpoint '{test_checkpoint}' not found. Please specify a valid checkpoint path.")
        print(f"If you haven't trained the 'full' model yet, you might need to use a 'lite' model checkpoint or train one.")
    elif not os.path.exists(DEFAULT_INSTRUMENTAL_PATH):
         print(f"Error: Default instrumental '{DEFAULT_INSTRUMENTAL_PATH}' not found. Please specify a valid instrumental path.")
    else:
        visualize_flow_rhythm(
            model_checkpoint_path=test_checkpoint, 
            model_config_path=DEFAULT_MODEL_CONFIG_PATH, 
            tokenizer_config_path=DEFAULT_TOKENIZER_CONFIG_PATH,
            instrumental_audio_path=DEFAULT_INSTRUMENTAL_PATH, 
            num_prompt_bars=2,       
            max_generated_flow_lines=4, 
            output_filename_prefix="full_model_flow_vis"
        )