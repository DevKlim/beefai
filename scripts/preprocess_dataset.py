import os
import sys
import argparse
import torch
import json 
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Ensure the beefai package is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from beefai.data_processing.beat_feature_extractor import BeatFeatureExtractor
from beefai.data_processing.flow_data_extractor import FlowDataExtractor 
from beefai.utils.data_types import SongBeatFeatures, FlowData, TrainingInstance # Ensure TrainingInstance matches actual use

# --- Configuration ---
DEFAULT_DATASET_BASE_DIR = os.path.join(project_root, "data")
DEFAULT_INSTRUMENTALS_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "instrumentals") 
DEFAULT_ALIGNMENTS_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "alignments_json") 
DEFAULT_PROCESSED_OUTPUT_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "processed_for_transformer")
DEFAULT_OUTPUT_PROCESSED_FILENAME = "processed_training_data.pt" # Renamed for clarity

# Default base directory for pre-separated stems
DEFAULT_PRE_SEPARATED_STEMS_BASE_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "temp_demucs_separated")
DEFAULT_SEPARATOR_TOOL_USED = "demucs" 
DEFAULT_DEMUCS_MODEL_NAME = "htdemucs_ft" # Example model name if demucs is used

BEAT_FEATURES_CACHE_SUBDIR = "beat_features_cache"
FLOW_DATA_CACHE_SUBDIR = "flow_data_cache"

def ensure_dir(directory_path: str):
    os.makedirs(directory_path, exist_ok=True)

def get_song_specific_stems_dir(
    base_stems_dir_for_tool: str, # e.g., "data/temp_demucs_separated/htdemucs_ft" or "data/temp_audio_sep_output"
    song_id: str, 
    separator_tool: str
    # demucs_model_name is now implicitly part of base_stems_dir_for_tool if separator_tool is 'demucs'
) -> str:
    """
    Constructs the path to the directory containing stems for a specific song.
    - For 'demucs', base_stems_dir_for_tool is expected to be like 'data/temp_demucs_separated/MODEL_NAME'.
      The song_id is then appended (e.g., 'data/temp_demucs_separated/MODEL_NAME/SONG_ID').
    - For 'audio_separator', base_stems_dir_for_tool is the top-level output (e.g., 'data/temp_audio_sep_output').
      The tool typically creates a subdirectory named after the song_id (e.g., 'data/temp_audio_sep_output/SONG_ID').
    """
    # Both demucs (when outputting to a model-specific dir) and audio-separator
    # often create a subdirectory named after the song (without extension).
    return os.path.join(base_stems_dir_for_tool, song_id)


def process_song(
    song_id: str,
    instrumental_audio_file_path: str, 
    alignment_file_path: Optional[str],
    beat_extractor: BeatFeatureExtractor,
    flow_extractor: FlowDataExtractor,
    cache_base_dir: str, # e.g. data/processed_for_transformer/
    stems_dir_for_this_song: Optional[str], # Specific path like "data/temp_demucs_separated/htdemucs_ft/Alright"
    force_reprocess: bool = False
) -> Optional[TrainingInstance]: # Return type updated
    print(f"\nProcessing song: {song_id} (Instrumental: {os.path.basename(instrumental_audio_file_path)})")

    beat_features_cache_file = os.path.join(cache_base_dir, BEAT_FEATURES_CACHE_SUBDIR, f"{song_id}_beat_features.pt")
    flow_data_cache_file = os.path.join(cache_base_dir, FLOW_DATA_CACHE_SUBDIR, f"{song_id}_flow_data.pt")
    
    ensure_dir(os.path.join(cache_base_dir, BEAT_FEATURES_CACHE_SUBDIR))
    ensure_dir(os.path.join(cache_base_dir, FLOW_DATA_CACHE_SUBDIR))

    if stems_dir_for_this_song:
        print(f"  Expecting pre-separated stems for '{song_id}' in: {stems_dir_for_this_song}")
        if not os.path.isdir(stems_dir_for_this_song):
            print(f"  Warning: Expected stems directory not found: {stems_dir_for_this_song}. BeatFeatureExtractor might use fallback.")
            # Stems might not be strictly required if BeatFeatureExtractor can simulate, but quality may differ.
    else:
        print(f"  No pre-separated stems directory provided for '{song_id}'. BeatFeatureExtractor will use fallback.")


    song_beat_features: Optional[SongBeatFeatures] = None
    if not force_reprocess and os.path.exists(beat_features_cache_file):
        print(f"  Loading cached beat features for {song_id}...")
        try: 
            song_beat_features = torch.load(beat_features_cache_file, weights_only=False)
        except Exception as e: 
            print(f"  Error loading cached beat features for {song_id}: {e}. Reprocessing.")
            song_beat_features = None # Ensure reprocessing
    
    if song_beat_features is None:
        if not os.path.exists(instrumental_audio_file_path):
            print(f"  ERROR: Instrumental audio file not found for {song_id} at '{instrumental_audio_file_path}'. Cannot extract beat features.")
            return None
        print(f"  Extracting beat features for {song_id}...")
        song_beat_features = beat_extractor.extract_song_beat_features(
            audio_path=instrumental_audio_file_path, 
            stems_input_dir=stems_dir_for_this_song 
        )
        if song_beat_features: 
            torch.save(song_beat_features, beat_features_cache_file)
            print(f"  Saved beat features for {song_id} to cache.")
        else: 
            print(f"  ERROR: Failed to extract beat features for {song_id}. Skipping song.")
            return None
            
    if not song_beat_features: # Should be caught by above, but defensive
        print(f"  ERROR: No beat features available for {song_id} after attempting extraction/cache. Skipping song.")
        return None

    song_flow_data: Optional[FlowData] = None
    if not alignment_file_path or not os.path.exists(alignment_file_path):
        print(f"  Warning: Alignment file not found for {song_id} at '{alignment_file_path}'. Cannot extract flow data. Proceeding with beat features only.")
        # We can still create a TrainingInstance with only beat_features if flow_data is optional for some use cases
        # However, for training the flow model, flow_targets are essential.
        # Let's assume for now that if alignment is missing, we skip the song for flow model training data.
        print(f"  Skipping {song_id} for processed_training_data.pt due to missing alignment for flow targets.")
        return None 
            
    if not force_reprocess and os.path.exists(flow_data_cache_file):
        print(f"  Loading cached flow data for {song_id}...")
        try: 
            song_flow_data = torch.load(flow_data_cache_file, weights_only=False)
        except Exception as e: 
            print(f"  Error loading cached flow data for {song_id}: {e}. Reprocessing.")
            song_flow_data = None # Ensure reprocessing
                
    if song_flow_data is None:
        print(f"  Extracting flow data for {song_id} using alignment: {os.path.basename(alignment_file_path)}...")
        song_flow_data = flow_extractor.extract_flow_for_song(
            alignment_data_path=alignment_file_path, 
            song_beat_features=song_beat_features # Pass beat features for context
        )
        if song_flow_data: 
            torch.save(song_flow_data, flow_data_cache_file)
            print(f"  Saved flow data for {song_id} to cache.")
        else: 
            print(f"  ERROR: Failed to extract flow data for {song_id}. Skipping song.")
            return None # Flow data is crucial for training targets
            
    if not song_flow_data: # Should be caught by above
        print(f"  ERROR: No flow data available for {song_id} after attempting extraction/cache. Skipping song.")
        return None

    # Construct the TrainingInstance dictionary
    # Ensure keys match what 05a_tokenize_data_lite.py / 05b_tokenize_data_full.py expect
    training_instance: TrainingInstance = {
        "song_name": song_id, # Changed from "song_id" to "song_name" if tokenization scripts use that
        "beat_features": song_beat_features,
        "flow_data": song_flow_data # Changed from "flow_targets" to "flow_data"
    }
    return training_instance


def main(args):
    ensure_dir(args.processed_output_dir)

    # Check main input directories
    if not os.path.isdir(args.instrumentals_dir) or not os.listdir(args.instrumentals_dir):
        print(f"ERROR: Instrumentals directory '{args.instrumentals_dir}' is missing or empty.")
        print("Please ensure it contains instrumental audio files.")
        return
    
    if not os.path.isdir(args.alignments_dir) or not os.listdir(args.alignments_dir):
        print(f"ERROR: Alignments directory '{args.alignments_dir}' is missing or empty.")
        print("Please ensure it contains alignment JSON files (e.g., from whisper-timestamped).")
        return
    
    # Construct the base directory for stems based on the separation tool and model (if demucs)
    actual_pre_separated_stems_base_dir: str
    if args.separator_tool_used == "demucs":
        if not args.demucs_model_name:
            print("ERROR: If separator_tool_used is 'demucs', --demucs_model_name must be provided.")
            return
        actual_pre_separated_stems_base_dir = os.path.join(args.pre_separated_stems_root_dir, args.demucs_model_name)
    else: # e.g., "audio_separator"
        actual_pre_separated_stems_base_dir = args.pre_separated_stems_root_dir
        
    if not os.path.isdir(actual_pre_separated_stems_base_dir):
        print(f"Warning: Pre-separated stems base directory for the tool ('{actual_pre_separated_stems_base_dir}') not found.")
        print("BeatFeatureExtractor may not find specific stems and might use fallbacks (e.g., simulate from full mix or use empty features for drums/bass).")


    beat_extractor = BeatFeatureExtractor(sample_rate=args.sample_rate) 
    flow_extractor = FlowDataExtractor(sample_rate_for_acapella=args.sample_rate) # Assuming acapellas also at this SR
    
    all_processed_song_data: List[TrainingInstance] = []
    audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg') # Added .ogg
    song_files_to_process = []

    for filename in os.listdir(args.instrumentals_dir):
        if filename.lower().endswith(audio_extensions):
            song_id = os.path.splitext(filename)[0]
            instrumental_audio_file_path = os.path.join(args.instrumentals_dir, filename)
            alignment_file_path = os.path.join(args.alignments_dir, f"{song_id}.json") 
            
            # Determine the specific directory for this song's stems
            stems_dir_for_song = get_song_specific_stems_dir(
                actual_pre_separated_stems_base_dir, song_id, args.separator_tool_used
            )

            song_files_to_process.append({
                "song_id": song_id, 
                "instrumental_audio_path": instrumental_audio_file_path,
                "alignment_path": alignment_file_path,
                "stems_dir_for_song": stems_dir_for_song
            })
    
    if not song_files_to_process: 
        print(f"No audio files with extensions {audio_extensions} found in the instrumentals directory: {args.instrumentals_dir}. Exiting.")
        return
    print(f"Found {len(song_files_to_process)} instrumental audio files to process.")

    for song_info in tqdm(song_files_to_process, desc="Preprocessing Songs"):
        processed_data = process_song(
            song_id=song_info["song_id"], 
            instrumental_audio_file_path=song_info["instrumental_audio_path"],
            alignment_file_path=song_info["alignment_path"],
            beat_extractor=beat_extractor, 
            flow_extractor=flow_extractor, 
            cache_base_dir=args.processed_output_dir, # Cache goes into processed_output_dir/beat_features_cache etc.
            stems_dir_for_this_song=song_info["stems_dir_for_song"],
            force_reprocess=args.force_reprocess
        )
        if processed_data: 
            all_processed_song_data.append(processed_data)
    
    output_file_path = os.path.join(args.processed_output_dir, DEFAULT_OUTPUT_PROCESSED_FILENAME)
    if all_processed_song_data:
        print(f"\nSuccessfully processed {len(all_processed_song_data)} songs.")
        try:
            torch.save(all_processed_song_data, output_file_path)
            print(f"Processed training data saved to: {output_file_path}")
        except Exception as e:
            print(f"ERROR: Failed to save processed training data to {output_file_path}: {e}")
    else:
        print("\nNo songs were successfully processed. Output file not created.")
        if os.path.exists(output_file_path):
            print(f"Note: An older version of {output_file_path} might still exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess rap dataset: Extract Beat & Flow features and save as TrainingInstances.")
    parser.add_argument("--instrumentals_dir", type=str, default=DEFAULT_INSTRUMENTALS_DIR,
                        help=f"Directory containing instrumental audio files. Default: {DEFAULT_INSTRUMENTALS_DIR}")
    parser.add_argument("--alignments_dir", type=str, default=DEFAULT_ALIGNMENTS_DIR, 
                        help=f"Directory containing alignment files (e.g., .json from whisper-timestamped). Default: {DEFAULT_ALIGNMENTS_DIR}")
    parser.add_argument("--processed_output_dir", type=str, default=DEFAULT_PROCESSED_OUTPUT_DIR,
                        help=f"Directory to save processed features, caches, and the output '{DEFAULT_OUTPUT_PROCESSED_FILENAME}'. Default: {DEFAULT_PROCESSED_OUTPUT_DIR}")
    
    # Arguments for locating pre-separated stems
    parser.add_argument("--pre_separated_stems_root_dir", type=str, 
                        default=DEFAULT_PRE_SEPARATED_STEMS_BASE_DIR,
                        help="Root directory where pre-separated stems are stored (e.g., 'data/temp_demucs_separated' or 'data/temp_audio_sep_output'). Default: %(default)s")
    parser.add_argument("--separator_tool_used", type=str, choices=["demucs", "audio_separator"], 
                        default=DEFAULT_SEPARATOR_TOOL_USED,
                        help="Which tool was used to generate the stems. This affects path construction under pre_separated_stems_root_dir. Default: %(default)s")
    parser.add_argument("--demucs_model_name", type=str, 
                        default=DEFAULT_DEMUCS_MODEL_NAME, 
                        help="If separator_tool_used is 'demucs', specify the Demucs model name (e.g., 'htdemucs_ft'). This forms part of the path: pre_separated_stems_root_dir/DEMUCS_MODEL_NAME/SONG_ID/. Default: %(default)s")

    parser.add_argument("--sample_rate", type=int, default=44100, help="Target sample rate for audio processing. Default: %(default)s")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of features, ignore all caches.")
    
    args = parser.parse_args()

    print("Starting dataset preprocessing (scripts/preprocess_dataset.py) with effective arguments:")
    for arg, value in vars(args).items(): print(f"  {arg}: {value}")
    
    main(args)