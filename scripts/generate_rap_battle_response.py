import os
import sys
import argparse
import json
import yaml
import time 
import re 
from datetime import datetime 
from typing import List, Optional, Tuple, Dict, Any

sys.path.append(os.getcwd())

import torch
import numpy as np

from beefai.data_processing.audio_processor import AudioProcessor
from beefai.data_processing.beat_feature_extractor import BeatFeatureExtractor
from beefai.evaluation.rhythm_visualizer import _generate_flow_core, _create_beat_info_from_custom_features, save_flow_data_to_json as save_generated_flow_json_util
from beefai.flow_model.tokenizer import FlowTokenizer
from beefai.lyric_generation.agent import LyricAgent, LYRIC_GENERATION_FAILED_MARKER 
from beefai.lyric_generation.prompt_formatter import PromptFormatter 
from beefai.synthesis.synthesizer import RapSynthesizer
from beefai.utils.data_types import FlowData, BeatInfo, SongBeatFeatures, AudioData, FlowDatum
from beefai.data_processing.text_processor import TextProcessor

DEFAULT_MODEL_CHECKPOINT = "data/checkpoints/flow_model_full/full_final_model.pt"
DEFAULT_MODEL_CONFIG_YAML = "lite_model_training/model_config_full.yaml" 
DEFAULT_TOKENIZER_CONFIG_JSON = "beefai/flow_model/flow_tokenizer_config_v2.json"
DEFAULT_OUTPUT_DIR = "output/rap_battle_responses"
DEFAULT_GENERATED_FLOW_FILENAME = "generated_flow_details.json"

# Filename for the final reviewed/edited lyric text that goes into espeak-ng
DEFAULT_LYRICS_FOR_TTS_FILENAME = "generated_lyrics_for_tts.txt" 
DEFAULT_INTERMEDIATE_VOCALS_FILENAME = "intermediate_vocals_only.mp3"

DEFAULT_LYRIC_BATCH_SIZE = 8 
DEFAULT_LYRIC_CONTEXT_LINES = 4 
DEFAULT_SYNTH_SYLLABLE_FADE_MS = 5 # Fade for syllable segments after extraction
DEFAULT_ESPEAK_VOICE = "en-us" 

LOG_PREFIX_MAIN = "[MainScript]"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an AI rap battle response.")
    parser.add_argument("--instrumental_path", type=str, required=True, help="Path to instrumental audio.")
    parser.add_argument("--diss_text", type=str, required=True, help="Input diss text.")
    parser.add_argument("--output_filename", type=str, default="ai_rap_response.mp3")
    parser.add_argument("--model_checkpoint", type=str, default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument("--model_config", type=str, default=DEFAULT_MODEL_CONFIG_YAML)
    parser.add_argument("--tokenizer_config", type=str, default=DEFAULT_TOKENIZER_CONFIG_JSON)
    parser.add_argument("--num_prompt_bars", type=int, default=2)
    
    # Flow generation parameters
    parser.add_argument("--flow_temperature", type=float, default=0.7)
    parser.add_argument("--flow_top_k", type=int, default=50)
    parser.add_argument("--flow_max_lines_heuristic", type=int, default=2)
    parser.add_argument("--flow_syl_count_bias", type=float, default=None)
    parser.add_argument("--flow_short_syl_dur_bias", type=float, default=None)
    parser.add_argument("--flow_long_syl_dur_bias", type=float, default=None) 
    parser.add_argument("--flow_rhythmic_offset_bias", type=float, default=None) 
    parser.add_argument("--flow_rhythmic_line_dur_bias", type=float, default=None) 
    parser.add_argument("--flow_rhythmic_syl_dur_bias", type=float, default=None) 
    parser.add_argument("--flow_min_syllable_threshold", type=int, default=5)
    parser.add_argument("--flow_min_syllable_penalty_strength", type=float, default=1.5)
    parser.add_argument("--flow_avoid_short_lines_end_bias", type=float, default=-2.0)

    # Lyric generation parameters
    parser.add_argument("--lyric_theme", type=str, default="AI superiority, witty comeback, roasting opponent")
    parser.add_argument("--lyric_agent_model", type=str, default="gemini-1.5-flash-latest") 
    parser.add_argument("--lyric_batch_size", type=int, default=DEFAULT_LYRIC_BATCH_SIZE)
    parser.add_argument("--lyric_context_lines", type=int, default=DEFAULT_LYRIC_CONTEXT_LINES)

    # Synthesis parameters (updated for stress modulation)
    parser.add_argument("--synth_sample_rate", type=int, default=44100)
    parser.add_argument("--synth_espeak_voice", type=str, default=DEFAULT_ESPEAK_VOICE)
    parser.add_argument("--synth_vocal_level_db", type=float, default=0.0)
    parser.add_argument("--synth_instrumental_level_db", type=float, default=-20.0)
    parser.add_argument("--synth_syllable_fade_ms", type=int, default=DEFAULT_SYNTH_SYLLABLE_FADE_MS, help="Fade (ms) for segmented syllables.") 
    parser.add_argument("--synth_max_workers_tts", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--synth_primary_stress_pitch_shift", type=float, default=0.75, help="Pitch shift in semitones for primary stressed syllables.")
    parser.add_argument("--synth_secondary_stress_pitch_shift", type=float, default=0.35, help="Pitch shift in semitones for secondary stressed syllables.")
    parser.add_argument("--synth_primary_stress_gain_db", type=float, default=1.5, help="Gain in dB for primary stressed syllables.")
    parser.add_argument("--synth_secondary_stress_gain_db", type=float, default=0.75, help="Gain in dB for secondary stressed syllables.")
    
    # File handling and control flow (updated paths)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip_instrumental_analysis", action="store_true")
    parser.add_argument("--skip_lyric_input_and_review", action="store_true", help="Skip Step 3 (Manual LLM Lyric Input & Text Review for TTS). Requires --load_lyrics_for_tts_path.")
    
    parser.add_argument("--load_flow_json_path", type=str, default=None)
    # This path is for loading the *initial* LLM output if user wants to reuse/edit it, before the TTS review step.
    parser.add_argument("--load_initial_lyrics_path", type=str, default=None, help="Path to TXT file with user's initial LLM-generated Actual Lyric Text. If not given, defaults to <output_dir>/initial_llm_lyrics.txt.") 
    # This path is for loading the *final, reviewed* text that goes directly to espeak-ng for line synthesis.
    parser.add_argument("--load_lyrics_for_tts_path", type=str, default=None, help=f"Path to TXT file with final lyric text for TTS. Default: <output_dir>/{DEFAULT_LYRICS_FOR_TTS_FILENAME}. Used if --skip_lyric_input_and_review.") 
    
    parser.add_argument("--save_lyrics_for_tts", action="store_true", help=f"Save the final (user-reviewed/edited) lyric text for TTS to <output_dir>/{DEFAULT_LYRICS_FOR_TTS_FILENAME}.") 
    parser.add_argument("--save_generated_flow_json", action="store_true")
    parser.add_argument("--save_intermediate_vocals", action="store_true", help=f"Save raw synthesized vocals to <output_dir>/{DEFAULT_INTERMEDIATE_VOCALS_FILENAME}.")
        
    parser.add_argument("--skip_lyric_text_review_for_tts", action="store_true", help="Skip interactive review of lyric text before TTS. Uses initial LLM/loaded lyrics directly.")

    return parser.parse_args()

def _initialize_components(args) -> Tuple[AudioProcessor, BeatFeatureExtractor, LyricAgent, RapSynthesizer, FlowTokenizer, TextProcessor]:
    print(f"{LOG_PREFIX_MAIN} [{time.strftime('%H:%M:%S')}] Initializing components...")
    audio_processor = AudioProcessor(default_sample_rate=args.synth_sample_rate)
    beat_extractor = BeatFeatureExtractor(sample_rate=args.synth_sample_rate) 
    if not os.path.exists(args.tokenizer_config): raise FileNotFoundError(f"Tokenizer config not found: {args.tokenizer_config}.")
    flow_tokenizer = FlowTokenizer(config_path=args.tokenizer_config)
    lyric_agent = LyricAgent(llm_model_name=args.lyric_agent_model) 
    text_processor = TextProcessor()
    rap_synthesizer = RapSynthesizer( 
        sample_rate=args.synth_sample_rate, 
        flow_tokenizer_config_path=args.tokenizer_config, 
        espeak_voice=args.synth_espeak_voice,
        max_workers_tts=args.synth_max_workers_tts,
        primary_stress_pitch_shift_semitones=args.synth_primary_stress_pitch_shift,
        secondary_stress_pitch_shift_semitones=args.synth_secondary_stress_pitch_shift,
        primary_stress_gain_db=args.synth_primary_stress_gain_db,
        secondary_stress_gain_db=args.synth_secondary_stress_gain_db
    )
    print(f"{LOG_PREFIX_MAIN} [{time.strftime('%H:%M:%S')}] Components initialized.")
    return audio_processor, beat_extractor, lyric_agent, rap_synthesizer, flow_tokenizer, text_processor

def _process_instrumental(args, audio_processor: AudioProcessor, beat_extractor: BeatFeatureExtractor) -> Tuple[SongBeatFeatures, Optional[AudioData], BeatInfo]:
    step_start_time = time.time(); print(f"\n{LOG_PREFIX_MAIN} [{time.strftime('%H:%M:%S')}] Step 1: Analyzing instrumental '{args.instrumental_path}'...")
    if not os.path.exists(args.instrumental_path): raise FileNotFoundError(f"Instrumental not found: {args.instrumental_path}.")
    song_beat_features: Optional[SongBeatFeatures] = beat_extractor.extract_features_for_song( audio_path=args.instrumental_path )
    if not song_beat_features: raise RuntimeError("Could not extract beat features.")
    print(f"{LOG_PREFIX_MAIN}   Extracted {len(song_beat_features)} bars of beat features.")
    instrumental_audio_for_synth: Optional[AudioData] = audio_processor.load_audio( args.instrumental_path, target_sr=args.synth_sample_rate, mono=True )
    if instrumental_audio_for_synth is None or instrumental_audio_for_synth[0].size == 0:
        print(f"{LOG_PREFIX_MAIN}   Warning: Could not load instrumental. Acapella synthesis will be attempted."); instrumental_audio_for_synth = (np.zeros(int(args.synth_sample_rate * 1.0)), args.synth_sample_rate)
    beat_info_for_synthesis: BeatInfo = _create_beat_info_from_custom_features(song_beat_features)
    if not beat_info_for_synthesis or beat_info_for_synthesis.get("bpm", 0) <= 0:
        print(f"{LOG_PREFIX_MAIN}   Warning: BPM from SBF-derived BeatInfo invalid. Trying direct analysis.")
        if instrumental_audio_for_synth and instrumental_audio_for_synth[0].size > 0 :
            wf_for_bi, sr_for_bi = instrumental_audio_for_synth
            beat_info_direct = audio_processor.get_beat_info(wf_for_bi, sr_for_bi)
            if beat_info_direct and beat_info_direct.get("bpm",0) > 0:
                beat_info_for_synthesis["bpm"] = beat_info_direct["bpm"]
                if not beat_info_for_synthesis.get("beat_times") and beat_info_direct.get("beat_times"): beat_info_for_synthesis["beat_times"] = beat_info_direct["beat_times"]
                if not beat_info_for_synthesis.get("downbeat_times") and beat_info_direct.get("downbeat_times"): beat_info_for_synthesis["downbeat_times"] = beat_info_direct["downbeat_times"]
                if not beat_info_for_synthesis.get("estimated_bar_duration") or beat_info_for_synthesis["estimated_bar_duration"] <=0 : beat_info_for_synthesis["estimated_bar_duration"] = beat_info_direct.get("estimated_bar_duration", (60.0/beat_info_direct["bpm"])*beat_info_direct.get("beats_per_bar",4))
                print(f"{LOG_PREFIX_MAIN}     Updated BeatInfo using BPM from direct analysis: {beat_info_for_synthesis['bpm']:.2f}")
        if (not beat_info_for_synthesis or beat_info_for_synthesis.get("bpm", 0) <= 0) and song_beat_features and song_beat_features[0].get("bpm",0) > 0 :
            first_sbf_bpm = song_beat_features[0]["bpm"]; first_sbf_bpb = song_beat_features[0].get("time_signature",(4,4))[0]; first_sbf_bpb = max(1, first_sbf_bpb)
            beat_info_for_synthesis = { "bpm": first_sbf_bpm, "beat_times": beat_info_for_synthesis.get("beat_times", []), "downbeat_times": beat_info_for_synthesis.get("downbeat_times", []), "estimated_bar_duration": (60.0 / first_sbf_bpm) * first_sbf_bpb if first_sbf_bpm > 0 else 2.0, "beats_per_bar": first_sbf_bpb, "sbf_features_for_timing_ref": song_beat_features }
            print(f"{LOG_PREFIX_MAIN}     Fallback BeatInfo constructed using BPM from first SBF: {beat_info_for_synthesis['bpm']:.2f}")
    if not beat_info_for_synthesis or beat_info_for_synthesis.get("bpm", 0) <= 0: raise RuntimeError("BeatInfo (BPM) extraction failed critically.")
    print(f"{LOG_PREFIX_MAIN}   Instrumental BPM for synthesis: {beat_info_for_synthesis['bpm']:.2f}")
    print(f"{LOG_PREFIX_MAIN} Step 1 finished in {time.time() - step_start_time:.2f}s.")
    return song_beat_features, instrumental_audio_for_synth, beat_info_for_synthesis

def _generate_rap_flow(args, song_beat_features: SongBeatFeatures, beat_info_for_synthesis: BeatInfo) -> FlowData:
    step_start_time = time.time(); print(f"\n{LOG_PREFIX_MAIN} [{time.strftime('%H:%M:%S')}] Step 2: Generating rap flow...")
    num_prompt = min(args.num_prompt_bars, len(song_beat_features)); 
    if num_prompt == 0 and len(song_beat_features) > 0: num_prompt = 1
    elif num_prompt == 0 and not song_beat_features: raise ValueError("No beat features available for flow model prompt.")
    prompt_sbf = song_beat_features[:num_prompt]
    context_bar_idx_start = prompt_sbf[-1]["bar_index"] + 1 if prompt_sbf else 0
    if not all(os.path.exists(p) for p in [args.model_checkpoint, args.model_config, args.tokenizer_config]): raise FileNotFoundError(f"Missing model/tokenizer config.")
    flow_gen_params = { "temperature": args.flow_temperature, "top_k": args.flow_top_k, "max_lines_per_bar_heuristic": args.flow_max_lines_heuristic, "syl_count_bias_strength": args.flow_syl_count_bias, "short_syl_dur_bias_strength": args.flow_short_syl_dur_bias, "long_syl_dur_bias_strength": args.flow_long_syl_dur_bias, "rhythmic_offset_bias_strength": args.flow_rhythmic_offset_bias, "rhythmic_line_duration_bias_strength": args.flow_rhythmic_line_dur_bias, "rhythmic_syllable_duration_bias_strength": args.flow_rhythmic_syl_dur_bias, "min_syllables_threshold": args.flow_min_syllable_threshold, "min_syllables_penalty_strength": args.flow_min_syllable_penalty_strength, "avoid_short_lines_end_bias_strength": args.flow_avoid_short_lines_end_bias }
    flow_gen_params_cleaned = {k: v for k, v in flow_gen_params.items() if v is not None}
    print(f"{LOG_PREFIX_MAIN}   Using flow generation parameters: {flow_gen_params_cleaned}") 
    generated_flow_data, _, _ = _generate_flow_core( prompt_bar_features=prompt_sbf, all_song_beat_features=song_beat_features, model_checkpoint_path=args.model_checkpoint, model_config_path=args.model_config, tokenizer_config_path=args.tokenizer_config, context_bar_idx_start=context_bar_idx_start, output_name_suffix="rap_battle_response_gen", generation_params=flow_gen_params_cleaned )
    if not generated_flow_data: raise RuntimeError("Failed to generate flow from model.")
    print(f"{LOG_PREFIX_MAIN}   Generated {len(generated_flow_data)} lines of flow data.")
    if args.save_generated_flow_json:
        flow_json_path = os.path.join(args.output_dir, DEFAULT_GENERATED_FLOW_FILENAME)
        save_generated_flow_json_util(generated_flow_data, beat_info_for_synthesis, flow_json_path)
    print(f"{LOG_PREFIX_MAIN} Step 2 finished in {time.time() - step_start_time:.2f}s.")
    return generated_flow_data

def _get_lyrics_manually( 
    args,
    flow_data_for_manual_prompt: FlowData,
    prompt_formatter: PromptFormatter,
    lyric_agent_instance: LyricAgent, 
    output_dir: str,
    bpm_for_prompt: Optional[float] = None,
    previous_lyric_context: Optional[List[str]] = None 
) -> List[Dict[str, Any]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    manual_prompt_filename = f"manual_llm_LYRIC_prompt_{timestamp}.txt"
    manual_input_lyrics_filename = f"manual_llm_LYRIC_input_{timestamp}.txt" 
    manual_prompt_filepath = os.path.join(output_dir, manual_prompt_filename)
    user_editable_lyrics_filepath = os.path.join(output_dir, manual_input_lyrics_filename)

    print(f"\n{LOG_PREFIX_MAIN} ### MANUAL LYRIC INPUT REQUIRED ###")
    
    if args.load_initial_lyrics_path: # Renamed from manual_llm_input_lyrics_file
        if not os.path.exists(args.load_initial_lyrics_path):
            raise FileNotFoundError(f"Manual LLM input lyrics file not found: {args.load_initial_lyrics_path}")
        user_editable_lyrics_filepath = args.load_initial_lyrics_path
        print(f"{LOG_PREFIX_MAIN} Using pre-existing LLM input lyrics file: {user_editable_lyrics_filepath}")
    else:
        persona = "a confident, witty, and cleverly boastful AI rapper named BeefAI"
        manual_prompt_text = prompt_formatter.format_lyric_generation_prompt_V2(
            args.diss_text,
            flow_data_for_manual_prompt,
            bpm=bpm_for_prompt,
            theme=args.lyric_theme,
            model_persona=persona,
            previous_lines_context=previous_lyric_context,
            num_creative_lines_suggestion=len(flow_data_for_manual_prompt) + max(2, args.lyric_batch_size // 2)
        )

        with open(manual_prompt_filepath, 'w', encoding='utf-8') as f:
            f.write(manual_prompt_text)
        
        with open(user_editable_lyrics_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Paste your LLM-generated LYRICS here, one lyric line per actual line in this file.\n")
            f.write(f"# Each lyric line should try to match the TARGET ORTHOGRAPHIC SYLLABLE COUNT from the prompt.\n")
            f.write(f"# Ensure you provide entries for all {len(flow_data_for_manual_prompt)} lines described in the prompt.\n")
            f.write(f"# Delete these instructional lines before saving.\n")

        print(f"{LOG_PREFIX_MAIN} 1. A detailed prompt (for LYRICS) has been saved to: {manual_prompt_filepath}")
        print(f"{LOG_PREFIX_MAIN} 2. The full prompt is also printed below. Copy this entire prompt.")
        print(f"{LOG_PREFIX_MAIN} 3. Paste it into your preferred LLM (e.g., Gemini Advanced, Claude, ChatGPT).")
        print(f"{LOG_PREFIX_MAIN} 4. Ensure the LLM generates LYRICS for ALL {len(flow_data_for_manual_prompt)} lines from the prompt, aiming for the target orthographic syllable counts.")
        print(f"{LOG_PREFIX_MAIN} 5. Copy ONLY the raw LYRIC lines from the LLM response (no extra text, no markdown).")
        print(f"{LOG_PREFIX_MAIN} 6. Open the file created at: {user_editable_lyrics_filepath}")
        print(f"{LOG_PREFIX_MAIN}    Paste your LYRICS into it (one lyric per line), remove instructional comments, and SAVE the file.")
        print(f"{LOG_PREFIX_MAIN} 7. After saving, press Enter here to continue...")
        
        print(f"\n{'-'*20} MANUAL LLM LYRIC PROMPT (for {len(flow_data_for_manual_prompt)} lines) {'-'*20}\n") 
        print(manual_prompt_text)
        print(f"\n{'-'*70}\n")
        
        input(f"Press Enter to continue once you have saved the LYRICS to '{user_editable_lyrics_filepath}'...")

    if not os.path.exists(user_editable_lyrics_filepath):
        raise FileNotFoundError(f"User-editable lyrics file not found: {user_editable_lyrics_filepath}")
    
    with open(user_editable_lyrics_filepath, 'r', encoding='utf-8') as f:
        raw_llm_lyric_output_text = f.read()

    parsed_lyric_data = lyric_agent_instance._parse_llm_lyric_response_lines(
        raw_llm_lyric_output_text,
        flow_data_for_manual_prompt 
    )
            
    print(f"{LOG_PREFIX_MAIN} Successfully parsed {len(parsed_lyric_data)} lyric lines from '{user_editable_lyrics_filepath}'.")
    return parsed_lyric_data

def _get_lyrics_and_prepare_for_tts(
    args, 
    full_flow_data: FlowData, 
    lyric_agent: LyricAgent, 
    bpm_for_prompt: Optional[float]
) -> List[str]: # Returns list of final lyric strings for TTS
    step_start_time = time.time()
    print(f"\n{LOG_PREFIX_MAIN} [{time.strftime('%H:%M:%S')}] Step 3: Obtaining & Reviewing Lyric Text for TTS...")
    
    final_lyric_lines_for_tts: List[str] = []
    
    # Default path for loading final lyrics IF skipping this whole step
    lyrics_to_load_for_tts_path = args.load_lyrics_for_tts_path if args.load_lyrics_for_tts_path else os.path.join(args.output_dir, DEFAULT_LYRICS_FOR_TTS_FILENAME)

    if args.skip_lyric_input_and_review:
        if not os.path.exists(lyrics_to_load_for_tts_path):
            raise FileNotFoundError(f"If --skip_lyric_input_and_review, final lyric text file '{lyrics_to_load_for_tts_path}' must exist.")
        print(f"{LOG_PREFIX_MAIN}   SKIPPING lyric input and review. Loading final lyric text for TTS from: {lyrics_to_load_for_tts_path}")
        with open(lyrics_to_load_for_tts_path, 'r', encoding='utf-8') as f:
            final_lyric_lines_for_tts = [line.strip() for line in f if not line.strip().startswith("#")]
        if len(final_lyric_lines_for_tts) != len(full_flow_data):
            raise ValueError(f"Loaded lyric text lines ({len(final_lyric_lines_for_tts)}) from '{lyrics_to_load_for_tts_path}' mismatches flow data lines ({len(full_flow_data)}).")
        print(f"{LOG_PREFIX_MAIN}   Loaded {len(final_lyric_lines_for_tts)} lines of final lyric text for TTS.")
        return final_lyric_lines_for_tts

    # --- Interactive Lyric Acquisition (User-LLM) and Review for TTS ---
    # 1. Get initial lyrics (from user-LLM interaction or loaded file)
    obtained_lyric_data_list: List[Dict[str, Any]] 
    initial_llm_lyrics_load_path = args.load_initial_lyrics_path if args.load_initial_lyrics_path else os.path.join(args.output_dir, "initial_llm_lyrics.txt") # Default name for initial lyrics

    if args.load_initial_lyrics_path and os.path.exists(args.load_initial_lyrics_path):
        print(f"{LOG_PREFIX_MAIN}   Loading initial LLM-generated lyrics from --load_initial_lyrics_path: {args.load_initial_lyrics_path}")
        with open(args.load_initial_lyrics_path, 'r', encoding='utf-8') as f:
            loaded_lyrics_text = f.read()
        obtained_lyric_data_list = lyric_agent._parse_llm_lyric_response_lines(loaded_lyrics_text, full_flow_data)
        print(f"{LOG_PREFIX_MAIN}   Parsed {len(obtained_lyric_data_list)} initial lyric lines from file.")
    else:
        # This calls the function that presents PromptFormatterV2 output to the user
        obtained_lyric_data_list = _get_lyrics_manually(
            args, full_flow_data, lyric_agent.prompt_formatter, lyric_agent, 
            args.output_dir, bpm_for_prompt=bpm_for_prompt, previous_lyric_context=None 
        )

    if not obtained_lyric_data_list or len(obtained_lyric_data_list) != len(full_flow_data):
        raise RuntimeError(f"Initial lyric acquisition failed or mismatched line count ({len(obtained_lyric_data_list)} vs {len(full_flow_data)}).")

    # 2. Review/Edit Lyric Text specifically for espeak-ng Pronunciation
    if args.skip_lyric_text_review_for_tts:
        print(f"{LOG_PREFIX_MAIN}   Skipping interactive review of lyric text for TTS as per --skip_lyric_text_review_for_tts.")
        final_lyric_lines_for_tts = [item["lyric"] for item in obtained_lyric_data_list]
    else:
        print(f"\n{LOG_PREFIX_MAIN} --- Review/Edit Lyric Text for espeak-ng TTS ---")
        print(f"{LOG_PREFIX_MAIN} Review each lyric line. You can make minor edits to the text if you anticipate")
        print(f"{LOG_PREFIX_MAIN} 'espeak-ng' might mispronounce words/abbreviations based on the current text.")
        print(f"{LOG_PREFIX_MAIN} Example: Change 'Dr. Dre' to 'Doctor Dre', or 'ft.' to 'featuring'.")
        
        for i, lyric_data_item in enumerate(obtained_lyric_data_list):
            original_lyric_text = lyric_data_item["lyric"]
            target_syl_count = full_flow_data[i].get("syllables", 0)
            
            print(f"\nLine {i+1}/{len(obtained_lyric_data_list)} (Target Ortho Syllables: {target_syl_count}):")
            print(f"  Current Lyric Text: \"{original_lyric_text}\"")
            
            if target_syl_count == 0:
                final_lyric_lines_for_tts.append("")
                print("  (Skipping review for 0-syllable line, will be silence in TTS)")
                continue

            user_input = input(f"  Enter text for espeak-ng to pronounce (or Enter to use current): ").strip()
            final_text_for_line = user_input if user_input else original_lyric_text
            final_lyric_lines_for_tts.append(final_text_for_line)
    
    # Save the final, possibly user-tweaked, lyrics that will be sent to TTS
    if args.save_lyrics_for_tts:
        final_lyrics_for_tts_save_path = os.path.join(args.output_dir, DEFAULT_LYRICS_FOR_TTS_FILENAME)
        os.makedirs(os.path.dirname(final_lyrics_for_tts_save_path), exist_ok=True)
        with open(final_lyrics_for_tts_save_path, 'w', encoding='utf-8') as f_out:
            for line_text in final_lyric_lines_for_tts: f_out.write(f"{line_text}\n")
        print(f"{LOG_PREFIX_MAIN}   Final lyric text for TTS saved to: {final_lyrics_for_tts_save_path}")
        
    print(f"{LOG_PREFIX_MAIN} Step 3 finished in {time.time() - step_start_time:.2f}s.")
    return final_lyric_lines_for_tts


def _synthesize_rap_audio(args, 
                          final_lyric_lines_for_tts: List[str], 
                          generated_flow_data: FlowData, 
                          beat_info_for_synthesis: BeatInfo, 
                          instrumental_audio_for_synth: Optional[AudioData], 
                          rap_synthesizer: RapSynthesizer):
    step_start_time = time.time()
    print(f"\n{LOG_PREFIX_MAIN} [{time.strftime('%H:%M:%S')}] Step 4: Synthesizing rap audio...")
    print(f"{LOG_PREFIX_MAIN}   Stage 4a: Generating vocal track using espeak-ng (whole lines, segmented, stress-modulated)...")
    raw_vocals_data: Optional[AudioData] = rap_synthesizer.synthesize_vocal_track( # Renamed method
        lyric_lines_for_tts=final_lyric_lines_for_tts, 
        flow_data_for_lyrics=generated_flow_data, 
        beat_info=beat_info_for_synthesis,
        syllable_segment_fade_ms=args.synth_syllable_fade_ms, # Corrected arg name
        vocal_level_db=args.synth_vocal_level_db 
    )
    if not raw_vocals_data or raw_vocals_data[0].size == 0: raise RuntimeError("Vocal synthesis failed or produced empty audio.")
    if args.save_intermediate_vocals:
        intermediate_vocals_path = os.path.join(args.output_dir, DEFAULT_INTERMEDIATE_VOCALS_FILENAME)
        rap_synthesizer.save_audio(raw_vocals_data, intermediate_vocals_path, format="mp3")
    print(f"{LOG_PREFIX_MAIN}   Stage 4b: RVC/Vocaloid smoothing (Conceptual - Current: using processed espeak-ng vocals directly).")
    smoothed_vocals_data: AudioData = raw_vocals_data 
    print(f"{LOG_PREFIX_MAIN}   Stage 4c: Mixing vocals with instrumental...")
    final_mixed_audio: Optional[AudioData] = rap_synthesizer.synthesize_verse_with_instrumental(
        phonetic_vocals_data=smoothed_vocals_data, instrumental_audio=instrumental_audio_for_synth,
        instrumental_level_db=args.synth_instrumental_level_db
    )
    if not final_mixed_audio or final_mixed_audio[0].size == 0: raise RuntimeError("Final audio mixing failed or produced empty audio.")
    output_audio_path = os.path.join(args.output_dir, args.output_filename)
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True) 
    output_format = os.path.splitext(args.output_filename)[1][1:].lower() 
    if output_format not in ["wav", "mp3"]:
        print(f"{LOG_PREFIX_MAIN}   Warning: Unsupported output format '{output_format}'. Defaulting to mp3.")
        output_format = "mp3"; output_audio_path = os.path.join(args.output_dir, os.path.splitext(args.output_filename)[0] + ".mp3")
    rap_synthesizer.save_audio(final_mixed_audio, output_audio_path, format=output_format) 
    print(f"{LOG_PREFIX_MAIN} Step 4 finished in {time.time() - step_start_time:.2f}s.")


def main():
    args = parse_arguments()
    script_start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{LOG_PREFIX_MAIN} --- AI Rap Battle Response Generation START (Workflow V4 - Stress Modulation) ---")
    print(f"{LOG_PREFIX_MAIN} Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{LOG_PREFIX_MAIN} Output Dir: {args.output_dir}, Output File: {args.output_filename}")
    
    audio_processor, beat_extractor, lyric_agent_main_scope, rap_synthesizer, _, text_processor_main_scope = _initialize_components(args)
    print(f"{LOG_PREFIX_MAIN} [{time.strftime('%H:%M:%S')}] Initialization phase complete.")

    instrumental_audio_for_synth: Optional[AudioData] = None
    beat_info_for_synthesis: Optional[BeatInfo] = None
    generated_flow_data: Optional[FlowData] = None
    final_lyric_lines_for_tts: Optional[List[str]] = None
    
    flow_json_path_effective = args.load_flow_json_path if args.load_flow_json_path else os.path.join(args.output_dir, DEFAULT_GENERATED_FLOW_FILENAME)
    
    try: 
        if args.skip_instrumental_analysis:
            print(f"\n{LOG_PREFIX_MAIN} [{time.strftime('%H:%M:%S')}] SKIPPING Steps 1 & 2 (Instrumental Analysis & Flow Generation).")
            if not os.path.exists(flow_json_path_effective): raise FileNotFoundError(f"Flow JSON not found for loading: '{flow_json_path_effective}'.")
            print(f"{LOG_PREFIX_MAIN}   Loading existing flow data from: {flow_json_path_effective}")
            with open(flow_json_path_effective, 'r') as f:
                loaded_data = json.load(f); generated_flow_data = loaded_data.get("flow_lines"); beat_info_for_synthesis = loaded_data.get("beat_info") 
                if not generated_flow_data or not beat_info_for_synthesis: raise ValueError(f"Flow JSON '{flow_json_path_effective}' missing 'flow_lines' or 'beat_info'.")
            print(f"{LOG_PREFIX_MAIN}   Loaded {len(generated_flow_data)} flow lines and beat info.")
            instrumental_audio_for_synth = audio_processor.load_audio(args.instrumental_path, target_sr=args.synth_sample_rate, mono=True)
            if not instrumental_audio_for_synth or instrumental_audio_for_synth[0].size == 0:
                print(f"{LOG_PREFIX_MAIN}     Warning: Could not load instrumental."); instrumental_audio_for_synth = (np.zeros(int(args.synth_sample_rate * 1.0)), args.synth_sample_rate)
        else: 
            song_beat_features_temp, instrumental_audio_for_synth_temp, beat_info_for_synthesis_temp = _process_instrumental(args, audio_processor, beat_extractor)
            instrumental_audio_for_synth = instrumental_audio_for_synth_temp; beat_info_for_synthesis = beat_info_for_synthesis_temp
            if not beat_info_for_synthesis: raise RuntimeError("Beat info could not be determined.")
            if not song_beat_features_temp : raise RuntimeError("Song beat features not determined.")
            generated_flow_data = _generate_rap_flow(args, song_beat_features_temp, beat_info_for_synthesis)

        if not generated_flow_data: raise RuntimeError("Flow data is missing.")
        if not beat_info_for_synthesis: raise RuntimeError("Beat info is missing.")
        current_bpm = beat_info_for_synthesis.get("bpm")

        final_lyric_lines_for_tts = _get_lyrics_and_prepare_for_tts(
            args, generated_flow_data, lyric_agent_main_scope, current_bpm
        )
        
        if not final_lyric_lines_for_tts: raise RuntimeError("Final lyric lines for TTS are missing.")
        if len(final_lyric_lines_for_tts) != len(generated_flow_data): raise RuntimeError(f"Mismatch: Final lyric lines ({len(final_lyric_lines_for_tts)}) vs flow data ({len(generated_flow_data)}).")

        if instrumental_audio_for_synth is None: 
            instrumental_audio_for_synth = audio_processor.load_audio(args.instrumental_path, target_sr=args.synth_sample_rate, mono=True)
            if instrumental_audio_for_synth is None or instrumental_audio_for_synth[0].size == 0:
                 print(f"{LOG_PREFIX_MAIN}     Warning: Could not load instrumental."); instrumental_audio_for_synth = (np.zeros(int(args.synth_sample_rate * 1.0)), args.synth_sample_rate)

        _synthesize_rap_audio(args, final_lyric_lines_for_tts, generated_flow_data, 
                              beat_info_for_synthesis, instrumental_audio_for_synth, rap_synthesizer)

        total_time = time.time() - script_start_time
        print(f"\n{LOG_PREFIX_MAIN} --- AI Rap Battle Response Generation COMPLETE ---")
        print(f"{LOG_PREFIX_MAIN} Total processing time: {total_time:.2f} seconds.")
        print(f"{LOG_PREFIX_MAIN} Final output saved to: {os.path.join(args.output_dir, args.output_filename)}")
        if args.save_intermediate_vocals: print(f"{LOG_PREFIX_MAIN} Intermediate vocals saved to: {os.path.join(args.output_dir, DEFAULT_INTERMEDIATE_VOCALS_FILENAME)}")
        if args.save_lyrics_for_tts: print(f"{LOG_PREFIX_MAIN} Final lyric text for TTS saved to: {os.path.join(args.output_dir, DEFAULT_LYRICS_FOR_TTS_FILENAME)}")

    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"{LOG_PREFIX_MAIN} ERROR: {type(e).__name__} - {e}")
        if os.path.exists(os.path.join(args.output_dir, DEFAULT_GENERATED_FLOW_FILENAME)): print(f"{LOG_PREFIX_MAIN}   Intermediate flow data may be at: {os.path.join(args.output_dir, DEFAULT_GENERATED_FLOW_FILENAME)}")
        if os.path.exists(os.path.join(args.output_dir, DEFAULT_LYRICS_FOR_TTS_FILENAME)): print(f"{LOG_PREFIX_MAIN}   Final lyric text for TTS may be at: {os.path.join(args.output_dir, DEFAULT_LYRICS_FOR_TTS_FILENAME)}")
        sys.exit(1)
    except Exception as e: 
        print(f"{LOG_PREFIX_MAIN} UNEXPECTED ERROR: {type(e).__name__} - {e}")
        import traceback; traceback.print_exc()
        if os.path.exists(os.path.join(args.output_dir, DEFAULT_GENERATED_FLOW_FILENAME)): print(f"{LOG_PREFIX_MAIN}   Intermediate flow data may be at: {os.path.join(args.output_dir, DEFAULT_GENERATED_FLOW_FILENAME)}")
        if os.path.exists(os.path.join(args.output_dir, DEFAULT_LYRICS_FOR_TTS_FILENAME)): print(f"{LOG_PREFIX_MAIN}   Final lyric text for TTS may be at: {os.path.join(args.output_dir, DEFAULT_LYRICS_FOR_TTS_FILENAME)}")
        sys.exit(1)

if __name__ == "__main__":
    main()