import os
import sys
import argparse
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Ensure the project root is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from beefai.utils.data_types import TrainingInstance, SongBeatFeatures, FlowData
except ImportError as e_import:
    print(f"CRITICAL IMPORT ERROR in aggregate_cached_features.py: {e_import}", flush=True)
    sys.exit(1)

# Default paths relative to project root
DEFAULT_PROCESSED_OUTPUT_DIR = os.path.join(project_root, "data", "processed_for_transformer")
DEFAULT_OUTPUT_FILENAME = "processed_training_data.pt"
BEAT_FEATURES_CACHE_SUBDIR = "beat_features_cache"
FLOW_DATA_CACHE_SUBDIR = "flow_data_cache"

def aggregate_cached_features(args):
    print("--- Aggregating Cached Features into Training Data ---", flush=True)

    processed_output_dir = args.processed_output_dir
    output_filename = args.output_filename

    beat_features_cache_dir = os.path.join(processed_output_dir, BEAT_FEATURES_CACHE_SUBDIR)
    flow_data_cache_dir = os.path.join(processed_output_dir, FLOW_DATA_CACHE_SUBDIR)

    if not os.path.isdir(beat_features_cache_dir):
        print(f"ERROR: Beat features cache directory not found: {beat_features_cache_dir}", flush=True)
        return
    if not os.path.isdir(flow_data_cache_dir):
        print(f"ERROR: Flow data cache directory not found: {flow_data_cache_dir}", flush=True)
        return

    print(f"Reading beat features from: {beat_features_cache_dir}", flush=True)
    print(f"Reading flow data from: {flow_data_cache_dir}", flush=True)

    aggregated_training_instances: List[TrainingInstance] = []
    successfully_aggregated_count = 0
    failed_aggregation_count = 0

    beat_feature_files = [f for f in os.listdir(beat_features_cache_dir) if f.endswith("_beat_features.pt")]
    print(f"Found {len(beat_feature_files)} beat feature cache files.", flush=True)

    for bf_filename in tqdm(beat_feature_files, desc="Aggregating cached data"):
        song_id_match = bf_filename.removesuffix("_beat_features.pt")
        if not song_id_match or song_id_match == bf_filename: # Check if suffix was actually removed
            print(f"Warning: Could not derive song_id from beat feature file: {bf_filename}. Skipping.", flush=True)
            failed_aggregation_count += 1
            continue
        
        song_id = song_id_match

        beat_features_path = os.path.join(beat_features_cache_dir, bf_filename)
        flow_data_filename = f"{song_id}_flow_data.pt"
        flow_data_path = os.path.join(flow_data_cache_dir, flow_data_filename)

        if not os.path.exists(flow_data_path):
            print(f"Warning: Flow data cache file not found for song_id '{song_id}' ({flow_data_path}). Skipping.", flush=True)
            failed_aggregation_count += 1
            continue

        try:
            # print(f"DEBUG: Loading beat features for {song_id} from {beat_features_path}", flush=True)
            beat_features: Optional[SongBeatFeatures] = torch.load(beat_features_path, weights_only=False)
            # print(f"DEBUG: Loading flow data for {song_id} from {flow_data_path}", flush=True)
            flow_data: Optional[FlowData] = torch.load(flow_data_path, weights_only=False)

            if not beat_features:
                print(f"Warning: Loaded beat features for song_id '{song_id}' are empty or None. Skipping.", flush=True)
                failed_aggregation_count += 1
                continue
            if not flow_data:
                print(f"Warning: Loaded flow data for song_id '{song_id}' are empty or None. Skipping.", flush=True)
                failed_aggregation_count += 1
                continue
            
            # Type check after loading (optional but good for robustness)
            if not isinstance(beat_features, list) or (beat_features and not isinstance(beat_features[0], dict)):
                print(f"Warning: Loaded beat_features for {song_id} is not in expected format (List[Dict]). Skipping.", flush=True)
                failed_aggregation_count += 1
                continue
            if not isinstance(flow_data, list) or (flow_data and not isinstance(flow_data[0], dict)):
                print(f"Warning: Loaded flow_data for {song_id} is not in expected format (List[Dict]). Skipping.", flush=True)
                failed_aggregation_count += 1
                continue


            training_instance: TrainingInstance = {
                "song_name": song_id,
                "beat_features": beat_features,
                "flow_data": flow_data
            }
            aggregated_training_instances.append(training_instance)
            successfully_aggregated_count += 1

        except Exception as e:
            print(f"Error loading or processing cached data for song_id '{song_id}': {e}. Skipping.", flush=True)
            failed_aggregation_count += 1
            continue

    print(f"\n--- Aggregation Summary ---", flush=True)
    print(f"Successfully aggregated instances: {successfully_aggregated_count}", flush=True)
    print(f"Failed/Skipped instances: {failed_aggregation_count}", flush=True)

    if aggregated_training_instances:
        output_file_path = os.path.join(processed_output_dir, output_filename)
        ensure_dir(os.path.dirname(output_file_path)) # Ensure output directory exists

        try:
            torch.save(aggregated_training_instances, output_file_path)
            print(f"Aggregated training data ({len(aggregated_training_instances)} instances) saved to: {output_file_path}", flush=True)
        except Exception as e_save:
            print(f"ERROR: Failed to save aggregated training data to {output_file_path}: {e_save}", flush=True)
    else:
        print(f"Warning: No training instances were successfully aggregated. Output file will not be created.", flush=True)

def ensure_dir(directory_path: str):
    os.makedirs(directory_path, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate cached beat and flow features into a single training data file.")
    parser.add_argument("--processed_output_dir", type=str, default=DEFAULT_PROCESSED_OUTPUT_DIR,
                        help=f"Directory containing '{BEAT_FEATURES_CACHE_SUBDIR}/' and '{FLOW_DATA_CACHE_SUBDIR}/', and where the output file will be saved. Default: %(default)s")
    parser.add_argument("--output_filename", type=str, default=DEFAULT_OUTPUT_FILENAME,
                        help=f"Filename for the aggregated training data .pt file. Default: %(default)s")
    
    args = parser.parse_args()
    
    print(f"Running script with arguments:", flush=True)
    print(f"  Processed Output Directory: {args.processed_output_dir}", flush=True)
    print(f"  Output Filename: {args.output_filename}", flush=True)
    
    aggregate_cached_features(args)
    print("--- Script Finished ---", flush=True)