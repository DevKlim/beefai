print("DEBUG: preprocess_dataset.py - Script execution started (top level)", flush=True) # VERY FIRST LINE

import os
import sys
import argparse
import torch
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import multiprocessing
from functools import partial
import re 
import traceback

# --- PROFILING IMPORTS (conditionally used) ---
import cProfile, pstats, io

print("DEBUG: preprocess_dataset.py - Basic imports done", flush=True)

# Ensure the beefai package is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"DEBUG: preprocess_dataset.py - Project root '{project_root}' added to sys.path if not present.", flush=True)

# Default paths (constants)
DEFAULT_DATASET_BASE_DIR = os.path.join(project_root, "data")
DEFAULT_INSTRUMENTALS_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "instrumentals") 
DEFAULT_ALIGNMENTS_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "alignments_json") 
DEFAULT_PROCESSED_OUTPUT_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "processed_for_transformer")
DEFAULT_OUTPUT_PROCESSED_FILENAME = "processed_training_data.pt" 

DEFAULT_PRE_SEPARATED_STEMS_ROOT_DIR_DEMUCS = os.path.join(DEFAULT_DATASET_BASE_DIR, "stems_cache") 
DEFAULT_PRE_SEPARATED_STEMS_ROOT_DIR_AUDIO_SEP = os.path.join(DEFAULT_DATASET_BASE_DIR, "stems_cache_audio_sep") 

DEFAULT_SEPARATOR_TOOL_USED = "demucs" 
DEFAULT_DEMUCS_MODEL_NAME = "htdemucs_ft" 

BEAT_FEATURES_CACHE_SUBDIR = "beat_features_cache"
FLOW_DATA_CACHE_SUBDIR = "flow_data_cache"

print("DEBUG: preprocess_dataset.py - Constants defined", flush=True)

# Attempt to import project modules now
try:
    from beefai.data_processing.beat_feature_extractor import BeatFeatureExtractor
    from beefai.data_processing.flow_data_extractor import FlowDataExtractor 
    from beefai.utils.data_types import SongBeatFeatures, FlowData, TrainingInstance 
    from beefai.flow_model.tokenizer import FlowTokenizer # Added for the tokenizer path argument
    print("DEBUG: preprocess_dataset.py - beefai module imports successful", flush=True)
except ImportError as e_import:
    print(f"DEBUG: preprocess_dataset.py - ERROR during beefai module import: {e_import}", flush=True)
    print(f"DEBUG: sys.path is currently: {sys.path}", flush=True)
    sys.exit(f"CRITICAL IMPORT ERROR in preprocess_dataset.py: {e_import}")


def ensure_dir(directory_path: str):
    os.makedirs(directory_path, exist_ok=True)

def get_song_specific_stems_dir(
    base_stems_dir_for_tool_model: str, 
    song_id: str
) -> str:
    return os.path.join(base_stems_dir_for_tool_model, song_id)


def process_song_worker(
    song_info: Dict[str, Any], 
    beat_extractor_config: Dict[str, Any], 
    flow_extractor_init_args: Dict[str, Any], 
    cache_base_dir_arg: str, 
    force_reprocess_arg: bool,
    worker_id: Optional[int] = None 
) -> Optional[TrainingInstance]:
    
    current_pid = os.getpid()
    log_prefix = f"[WorkerPID:{current_pid}]"
    song_id = song_info["song_id"] 

    # print(f"{log_prefix} [{song_id}] Worker processing started.", flush=True) # Less verbose for speed

    try: 
        # print(f"{log_prefix} [{song_id}] Initializing BeatFeatureExtractor...", flush=True)
        current_beat_extractor = BeatFeatureExtractor(sample_rate=beat_extractor_config["sample_rate"])
        # print(f"{log_prefix} [{song_id}] BeatFeatureExtractor initialized.", flush=True)
        
        # print(f"{log_prefix} [{song_id}] Initializing FlowDataExtractor...", flush=True)
        tokenizer_for_fde: FlowTokenizer
        if isinstance(flow_extractor_init_args.get("tokenizer"), FlowTokenizer):
            tokenizer_for_fde = flow_extractor_init_args["tokenizer"]
        elif isinstance(flow_extractor_init_args.get("tokenizer_path"), str):
            tokenizer_path = flow_extractor_init_args["tokenizer_path"]
            if not os.path.exists(tokenizer_path):
                print(f"{log_prefix} [{song_id}] ERROR: Tokenizer path '{tokenizer_path}' for FDE not found. Cannot proceed.")
                return None
            tokenizer_for_fde = FlowTokenizer(config_path=tokenizer_path)
        else:
            print(f"{log_prefix} [{song_id}] ERROR: Tokenizer or tokenizer_path not provided to FDE. Cannot proceed.")
            return None

        current_flow_extractor = FlowDataExtractor(
            tokenizer=tokenizer_for_fde, 
            sample_rate_for_acapella=flow_extractor_init_args["sample_rate"],
            subdivisions_per_bar=flow_extractor_init_args["subdivisions_per_bar"] 
        )
        # print(f"{log_prefix} [{song_id}] FlowDataExtractor initialized with tokenizer.", flush=True)

        instrumental_audio_file_path = song_info["instrumental_audio_path"]
        alignment_file_path = song_info["alignment_path"]
        stems_dir_for_this_song = song_info["stems_dir_for_song"] 

        beat_features_cache_file = os.path.join(cache_base_dir_arg, BEAT_FEATURES_CACHE_SUBDIR, f"{song_id}_beat_features.pt")
        flow_data_cache_file = os.path.join(cache_base_dir_arg, FLOW_DATA_CACHE_SUBDIR, f"{song_id}_flow_data.pt")
        
        ensure_dir(os.path.join(cache_base_dir_arg, BEAT_FEATURES_CACHE_SUBDIR))
        ensure_dir(os.path.join(cache_base_dir_arg, FLOW_DATA_CACHE_SUBDIR))

        if stems_dir_for_this_song and not os.path.isdir(stems_dir_for_this_song):
            # print(f"{log_prefix} [{song_id}] Stems directory '{stems_dir_for_this_song}' not found, will use full mix for BFE.", flush=True)
            stems_dir_for_this_song = None 

        # print(f"{log_prefix} [{song_id}] Starting Beat Feature Extraction (Stage 2)...", flush=True)
        song_beat_features: Optional[SongBeatFeatures] = None
        if not force_reprocess_arg and os.path.exists(beat_features_cache_file):
            # print(f"{log_prefix} [{song_id}] Attempting to load beat features from cache: {beat_features_cache_file}", flush=True)
            try: 
                song_beat_features = torch.load(beat_features_cache_file, weights_only=False)
                # print(f"{log_prefix} [{song_id}] Beat features loaded from cache.", flush=True)
            except Exception as e_load_bf:
                print(f"{log_prefix} [{song_id}] Beat Features: Warning - Failed to load from cache ({beat_features_cache_file}): {e_load_bf}. Reprocessing.", flush=True)
                song_beat_features = None 
        
        if song_beat_features is None: 
            if not os.path.exists(instrumental_audio_file_path):
                print(f"{log_prefix} [{song_id}] Beat Features: ERROR - Instrumental audio not found at '{instrumental_audio_file_path}'. Cannot extract beat features.", flush=True)
                return None 
            # print(f"{log_prefix} [{song_id}] Extracting beat features from: {instrumental_audio_file_path}", flush=True)
            song_beat_features = current_beat_extractor.extract_features_for_song(
                audio_path=instrumental_audio_file_path, 
                stems_input_dir=stems_dir_for_this_song 
            )
            if song_beat_features: 
                # print(f"{log_prefix} [{song_id}] Beat features extracted. Saving to cache: {beat_features_cache_file}", flush=True)
                try: torch.save(song_beat_features, beat_features_cache_file)
                except Exception as e_save_bf: print(f"{log_prefix} [{song_id}] Beat Features: Warning - Failed to save cache: {e_save_bf}", flush=True)
            # else:
                # print(f"{log_prefix} [{song_id}] Beat feature extraction returned no features.", flush=True)

        if not song_beat_features: 
            # print(f"{log_prefix} [{song_id}] Beat Features: ERROR - No beat features obtained after attempt. Skipping song.", flush=True)
            return None
        # print(f"{log_prefix} [{song_id}] Beat Feature Extraction (Stage 2) complete. Found {len(song_beat_features)} bars.", flush=True)

        # print(f"{log_prefix} [{song_id}] Starting Flow Data Extraction (Stage 3)...", flush=True)
        song_flow_data: Optional[FlowData] = None
        if not alignment_file_path or not os.path.exists(alignment_file_path):
            # print(f"{log_prefix} [{song_id}] Flow Data: Warning - Alignment file not found at '{alignment_file_path}'. Cannot extract flow data.", flush=True)
            # If no alignment, we can't get flow data. If beat features are still useful alone, decide policy.
            # For now, assume flow data is essential for a TrainingInstance.
            return None 
                
        if not force_reprocess_arg and os.path.exists(flow_data_cache_file):
            # print(f"{log_prefix} [{song_id}] Attempting to load flow data from cache: {flow_data_cache_file}", flush=True)
            try: 
                song_flow_data = torch.load(flow_data_cache_file, weights_only=False)
                # print(f"{log_prefix} [{song_id}] Flow data loaded from cache.", flush=True)
            except Exception as e_load_fd:
                print(f"{log_prefix} [{song_id}] Flow Data: Warning - Failed to load from cache ({flow_data_cache_file}): {e_load_fd}. Reprocessing.", flush=True)
                song_flow_data = None 
                    
        if song_flow_data is None: 
            # print(f"{log_prefix} [{song_id}] Extracting flow data using alignment: {alignment_file_path}", flush=True)
            song_flow_data = current_flow_extractor.extract_flow_for_song(
                alignment_data_path=alignment_file_path, 
                song_beat_features=song_beat_features
            )
            if song_flow_data: 
                # print(f"{log_prefix} [{song_id}] Flow data extracted. Saving to cache: {flow_data_cache_file}", flush=True)
                try: torch.save(song_flow_data, flow_data_cache_file)
                except Exception as e_save_fd: print(f"{log_prefix} [{song_id}] Flow Data: Warning - Failed to save cache: {e_save_fd}", flush=True)
            # else:
                 # print(f"{log_prefix} [{song_id}] Flow data extraction returned no data.", flush=True)

        if not song_flow_data:
            # print(f"{log_prefix} [{song_id}] Flow Data: ERROR - No flow data obtained after attempt. Skipping song.", flush=True)
            return None
        # print(f"{log_prefix} [{song_id}] Flow Data Extraction (Stage 3) complete. Found {len(song_flow_data)} flow entries.", flush=True)

        training_instance: TrainingInstance = {
            "song_name": song_id, 
            "beat_features": song_beat_features,
            "flow_data": song_flow_data 
        }
        # print(f"{log_prefix} [{song_id}] Worker processing finished successfully.", flush=True)
        return training_instance

    except Exception as e_worker:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        print(f"{log_prefix} [{song_id}] CRITICAL ERROR in worker process: {e_worker}", flush=True)
        print(f"{log_prefix} [{song_id}] Traceback:\n{traceback.format_exc()}", flush=True)
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        return None


def main(args):
    print("DEBUG: preprocess_dataset.py - main() function started", flush=True)
    ensure_dir(args.processed_output_dir)

    if not os.path.isdir(args.instrumentals_dir) or not os.listdir(args.instrumentals_dir):
        print(f"ERROR: Instrumentals directory '{args.instrumentals_dir}' is missing or empty. Cannot proceed.", flush=True)
        return
    
    if not os.path.isdir(args.alignments_dir): 
        print(f"Warning: Alignments directory '{args.alignments_dir}' is missing. Flow data extraction will fail for all songs if corresponding JSONs are absent.", flush=True)
    
    # Reminder about debug flags
    print("INFO: For optimal performance, ensure DEBUG_FLOW_EXTRACTOR in flow_data_extractor.py is False.", flush=True)

    print(f"DEBUG: preprocess_dataset.py - Determining actual_stems_path_for_bfe...", flush=True)
    actual_stems_path_for_bfe: str
    if args.separator_tool_used_for_stems == "demucs":
        if not args.stem_provider_model_name:
            print("ERROR: Demucs specified as separator tool but --stem_provider_model_name is missing.", flush=True)
            return
        actual_stems_path_for_bfe = os.path.join(args.pre_separated_stems_root_dir, args.stem_provider_model_name)
    elif args.separator_tool_used_for_stems == "audio_separator":
        if args.stem_provider_model_name: # audio_separator might or might not have a model-specific subdir
            actual_stems_path_for_bfe = os.path.join(args.pre_separated_stems_root_dir, args.stem_provider_model_name)
        else: 
             actual_stems_path_for_bfe = args.pre_separated_stems_root_dir
    else: # "none" or other
        print(f"Info: --separator_tool_used_for_stems is '{args.separator_tool_used_for_stems}'. Stems will not be actively searched for by BFE.", flush=True)
        actual_stems_path_for_bfe = "" # Explicitly empty
        
    if actual_stems_path_for_bfe and not os.path.isdir(actual_stems_path_for_bfe): 
        print(f"Warning: Constructed base stems path for BFE ('{actual_stems_path_for_bfe}') not found. BFE will likely fallback to full mix for all songs.", flush=True)

    song_files_to_process_info: List[Dict[str, Any]] = []
    audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg') 

    print(f"DEBUG: preprocess_dataset.py - Scanning instrumentals directory: {args.instrumentals_dir}", flush=True)
    for filename in os.listdir(args.instrumentals_dir):
        if filename.lower().endswith(audio_extensions):
            song_id = os.path.splitext(filename)[0]
            instrumental_audio_file_path = os.path.join(args.instrumentals_dir, filename)
            # Try to find alignment JSON (more flexible naming)
            possible_alignment_filenames = [
                f"{song_id}.json", 
                f"{filename}.json", # if instrumental filename was 'song.mp3.json'
                f"{song_id}.words.json", # common whisper-timestamped output
                f"{filename}.words.json" # if filename includes extension
            ]
            alignment_file_path = "" # Default to no alignment file found
            for p_align_fn in possible_alignment_filenames:
                cand_path = os.path.join(args.alignments_dir, p_align_fn)
                if os.path.exists(cand_path):
                    alignment_file_path = cand_path
                    break # Found one
            
            stems_dir_for_song_for_bfe = "" # Default
            if actual_stems_path_for_bfe: # Only try to construct if a base path is valid
                 stems_dir_for_song_for_bfe = get_song_specific_stems_dir(
                    actual_stems_path_for_bfe, song_id
                )
            
            song_files_to_process_info.append({
                "song_id": song_id, 
                "instrumental_audio_path": instrumental_audio_file_path,
                "alignment_path": alignment_file_path, 
                "stems_dir_for_song": stems_dir_for_song_for_bfe 
            })
    
    if not song_files_to_process_info: 
        print(f"ERROR: No audio files found in {args.instrumentals_dir}. Exiting.", flush=True)
        return
    print(f"DEBUG: preprocess_dataset.py - Found {len(song_files_to_process_info)} potential instrumental audio files to process.", flush=True)

    all_processed_song_data_results: List[Optional[TrainingInstance]] = []
    
    print("DEBUG: preprocess_dataset.py - Getting BeatFeatureExtractor config (subdivisions_per_bar)...", flush=True)
    try:
        # This is just to get the default subdivisions_per_bar, not for actual processing.
        temp_bfe_for_config = BeatFeatureExtractor(sample_rate=args.sample_rate)
        subdivisions_per_bar_for_config = temp_bfe_for_config.subdivisions_per_bar
        del temp_bfe_for_config 
        print(f"DEBUG: preprocess_dataset.py - Using subdivisions_per_bar: {subdivisions_per_bar_for_config} for FlowDataExtractor config.", flush=True)
    except Exception as e_bfe_init_main:
        print(f"ERROR: Failed to initialize BeatFeatureExtractor in main to get config: {e_bfe_init_main}", flush=True)
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return 

    # Prepare common configurations for workers
    beat_extractor_worker_config = {"sample_rate": args.sample_rate}
    if not os.path.exists(args.tokenizer_config_for_fde_path):
        print(f"ERROR: Tokenizer config for FlowDataExtractor not found at '{args.tokenizer_config_for_fde_path}'. Please specify with --tokenizer_config_for_fde_path.")
        return
    flow_extractor_worker_init_args = {
        "tokenizer_path": args.tokenizer_config_for_fde_path, 
        "sample_rate": args.sample_rate,
        "subdivisions_per_bar": subdivisions_per_bar_for_config
    }

    if args.debug_single_song_id:
        print(f"DEBUG MODE: Processing only song ID '{args.debug_single_song_id}'.", flush=True)
        target_song_info = next((s_info for s_info in song_files_to_process_info if s_info["song_id"] == args.debug_single_song_id), None)
        if target_song_info:
            print(f"[Main-Debug] Processing song: {args.debug_single_song_id} serially...", flush=True)
            
            pr = cProfile.Profile()
            pr.enable()
            
            result = process_song_worker(
                song_info=target_song_info,
                beat_extractor_config=beat_extractor_worker_config,
                flow_extractor_init_args=flow_extractor_worker_init_args,
                cache_base_dir_arg=args.processed_output_dir,
                force_reprocess_arg=args.force_reprocess, # Use True for worst-case profiling
                worker_id=0 
            )
            
            pr.disable()
            s_io = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE 
            ps = pstats.Stats(pr, stream=s_io).sort_stats(sortby)
            ps.print_stats(30) 
            print("\n--- cProfile Results (for single debug song) ---", flush=True)
            print(s_io.getvalue(), flush=True)
            print("--- End cProfile Results ---\n", flush=True)

            if result:
                 all_processed_song_data_results.append(result)
                 print(f"[Main-Debug] Finished debug song: {args.debug_single_song_id}. Result obtained.", flush=True)
            else:
                 print(f"[Main-Debug] Debugged song {args.debug_single_song_id} failed processing or returned no data.", flush=True)
        else:
            print(f"[Main-Debug] Debug song ID '{args.debug_single_song_id}' not found in the list of processable songs.", flush=True)
    else:
        num_workers_to_use = min(args.num_workers, os.cpu_count() if os.cpu_count() else 1) # Ensure not more than available
        num_workers_to_use = max(1, num_workers_to_use) # Ensure at least 1 worker
        
        print(f"DEBUG: preprocess_dataset.py - Starting parallel processing with {num_workers_to_use} worker(s).", flush=True)
        
        process_song_worker_partial = partial(process_song_worker,
                                              beat_extractor_config=beat_extractor_worker_config,
                                              flow_extractor_init_args=flow_extractor_worker_init_args,
                                              cache_base_dir_arg=args.processed_output_dir,
                                              force_reprocess_arg=args.force_reprocess,
                                              worker_id=None) # worker_id not strictly needed if not used by worker for logging
        
        print(f"DEBUG: preprocess_dataset.py - Dispatching {len(song_files_to_process_info)} songs to worker pool...", flush=True)
        try:
            # Using try-with-resources for the pool
            with multiprocessing.Pool(processes=num_workers_to_use) as pool:
                results_iterator = pool.imap_unordered(process_song_worker_partial, song_files_to_process_info)
                for result in tqdm(results_iterator, total=len(song_files_to_process_info), desc="Preprocessing Songs"):
                    if result: 
                        all_processed_song_data_results.append(result)
            print("DEBUG: preprocess_dataset.py - Worker pool finished and closed.", flush=True)
        except Exception as e_pool:
            print(f"ERROR: An error occurred during multiprocessing pool execution: {e_pool}", flush=True)
            print(f"Traceback: {traceback.format_exc()}", flush=True)
            # Depending on error, some results might be in all_processed_song_data_results
    
    final_training_instances: List[TrainingInstance] = [item for item in all_processed_song_data_results if item is not None]
    
    total_songs_considered = len(song_files_to_process_info) if not args.debug_single_song_id else 1
    successful_songs = len(final_training_instances)
    failed_songs = total_songs_considered - successful_songs

    output_file_path = os.path.join(args.processed_output_dir, DEFAULT_OUTPUT_PROCESSED_FILENAME)
    print(f"\n--- Preprocessing (Phases 2 & 3) Summary ---", flush=True)
    print(f"Total songs considered for processing: {total_songs_considered}", flush=True)
    print(f"Successfully processed songs: {successful_songs}", flush=True)
    print(f"Failed/Skipped songs: {failed_songs}", flush=True)

    if final_training_instances:
        try:
            torch.save(final_training_instances, output_file_path)
            print(f"Processed training data ({len(final_training_instances)} instances) saved to: {output_file_path}", flush=True)
        except Exception as e_save_main:
            print(f"ERROR: Failed to save processed training data to {output_file_path}: {e_save_main}", flush=True)
    else:
        print(f"Warning: No songs were successfully processed. Output file '{output_file_path}' will not be created or will be empty if it existed.", flush=True)

    print("DEBUG: preprocess_dataset.py - main() function finished", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support() # Good practice for multiprocessing, esp. on Windows

    print("DEBUG: preprocess_dataset.py - Script invoked as __main__", flush=True)
    parser = argparse.ArgumentParser(description="Preprocess rap dataset: Extract Beat & Flow features and save as TrainingInstances.")
    parser.add_argument("--instrumentals_dir", type=str, default=DEFAULT_INSTRUMENTALS_DIR,
                        help=f"Directory containing instrumental audio files. Default: {DEFAULT_INSTRUMENTALS_DIR}")
    parser.add_argument("--alignments_dir", type=str, default=DEFAULT_ALIGNMENTS_DIR, 
                        help=f"Directory containing alignment files (e.g., .json from whisper-timestamped). Default: {DEFAULT_ALIGNMENTS_DIR}")
    parser.add_argument("--processed_output_dir", type=str, default=DEFAULT_PROCESSED_OUTPUT_DIR,
                        help=f"Directory to save processed features, caches, and the output '{DEFAULT_OUTPUT_PROCESSED_FILENAME}'. Default: {DEFAULT_PROCESSED_OUTPUT_DIR}")
    
    parser.add_argument("--pre_separated_stems_root_dir", type=str, 
                        default=DEFAULT_PRE_SEPARATED_STEMS_ROOT_DIR_DEMUCS, # Defaulting to Demucs path convention
                        help="Root directory where BeatFeatureExtractor should look for detailed song stems. This path is combined with --stem_provider_model_name. Default: %(default)s")
    parser.add_argument("--separator_tool_used_for_stems", type=str, choices=["demucs", "audio_separator", "none"], 
                        default=DEFAULT_SEPARATOR_TOOL_USED,
                        help="Tool that provided detailed stems. 'none' means BFE uses full instrumentals for onset detection. Default: %(default)s")
    parser.add_argument("--stem_provider_model_name", type=str, 
                        default=DEFAULT_DEMUCS_MODEL_NAME, # Default for Demucs
                        help="Model name used by the --separator_tool_used_for_stems (e.g., 'htdemucs_ft', 'UVR-MDX-NET-Inst_HQ_3'). Forms part of the path to stems. Can be empty if not applicable. Default: %(default)s")

    parser.add_argument("--sample_rate", type=int, default=44100, help="Target sample rate for audio processing. Default: %(default)s")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing of features, ignore all caches.")
    
    default_num_workers = os.cpu_count()
    if default_num_workers is None: # os.cpu_count() can return None
        default_num_workers = 4 # Fallback if cpu_count is not determinable
    else:
        # Leave one core for system stability on multi-core, or use all for <=2 cores.
        default_num_workers = max(1, default_num_workers - 1 if default_num_workers > 2 else default_num_workers) 
    parser.add_argument("--num_workers", type=int, default=default_num_workers, 
                        help=f"Number of worker processes for parallel song processing. Default: ~{default_num_workers} (adaptive to CPU cores).")
    
    parser.add_argument("--debug_single_song_id", type=str, default=None, 
                        help="If set, process only this song ID serially for debugging (including cProfile output) and then exit early.")
    parser.add_argument("--tokenizer_config_for_fde_path", type=str, 
                            default=os.path.join(project_root, "beefai", "flow_model", "flow_tokenizer_config_v2.json"), # Default path
                            help="Path to the FlowTokenizer config JSON, needed by FlowDataExtractor for consistent binning. Default: %(default)s")
    
    args = parser.parse_args()
    print(f"DEBUG: preprocess_dataset.py - Parsed arguments: {args}", flush=True)

    print("DEBUG: preprocess_dataset.py - Starting dataset preprocessing with effective arguments:", flush=True)
    for arg_name, value in vars(args).items(): print(f"  {arg_name}: {value}", flush=True)
    
    if args.debug_single_song_id:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        print(f"!!! DEBUG MODE: ONLY PROCESSING SONG ID '{args.debug_single_song_id}' AND EXITING EARLY !!!", flush=True)
        print(f"!!! Profiling will be enabled for this single song run if not already cached or --force_reprocess. !!!", flush=True)
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", flush=True)

    try:
        main(args)
    except Exception as e_main_call:
        print(f"CRITICAL ERROR during main() call in preprocess_dataset.py: {e_main_call}", flush=True)
        print(f"Traceback for main() call error:\n{traceback.format_exc()}", flush=True)
    
    print("DEBUG: preprocess_dataset.py - Script execution finished.", flush=True)