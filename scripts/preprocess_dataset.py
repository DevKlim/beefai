import os
import sys
import argparse
import torch
import json # Not directly used for saving main data, but tokenizer might use it
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Ensure the beefai package is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from beefai.data_processing.beat_feature_extractor import BeatFeatureExtractor
from beefai.data_processing.flow_data_extractor import FlowDataExtractor
from beefai.flow_model.tokenizer import FlowTokenizer
from beefai.utils.data_types import SongBeatFeatures, FlowData # TrainingInstance defined implicitly by output dict

# --- Configuration ---
# Default paths relative to project root
DEFAULT_DATASET_BASE_DIR = os.path.join(project_root, "dataset_rap_music") # Main dataset folder

DEFAULT_RAW_SONGS_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "raw_songs")
DEFAULT_LYRICS_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "lyrics") 
DEFAULT_ALIGNMENTS_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "alignments_textgrid") # MFA TextGrid outputs

DEFAULT_PROCESSED_OUTPUT_DIR = os.path.join(DEFAULT_DATASET_BASE_DIR, "processed_for_transformer")
DEFAULT_TOKENIZER_CONFIG = os.path.join(project_root, "beefai", "flow_model", "flow_tokenizer_config_v2.json") # Consistent name
DEFAULT_OUTPUT_TOKENIZED_FILE = os.path.join(DEFAULT_PROCESSED_OUTPUT_DIR, "tokenized_flow_dataset.pt")

# Intermediate cache/output directories within the processed_output_dir
STEMS_CACHE_SUBDIR = "stems_cache" # For BFE's simulated/actual stems
BEAT_FEATURES_CACHE_SUBDIR = "beat_features_cache"
FLOW_DATA_CACHE_SUBDIR = "flow_data_cache"

def ensure_dir(directory_path: str):
    os.makedirs(directory_path, exist_ok=True)

def process_song(
    song_id: str,
    audio_file_path: str,
    # lyrics_file_path: Optional[str], # Lyrics file itself not directly used if alignment_file exists
    alignment_file_path: Optional[str], # TextGrid from MFA is primary for flow
    beat_extractor: BeatFeatureExtractor,
    flow_extractor: FlowDataExtractor,
    tokenizer: FlowTokenizer,
    base_processed_dir: str, # Base for all caches and stem outputs for this song
    force_reprocess: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Processes a single song: extracts beat features, flow data, and tokenizes.
    Returns a dictionary with 'song_id', 'token_ids', 'segment_ids', 'intra_line_pos_ids'.
    """
    print(f"\nProcessing song: {song_id} (Audio: {os.path.basename(audio_file_path)})")

    # Define cache paths relative to base_processed_dir
    beat_features_cache_file = os.path.join(base_processed_dir, BEAT_FEATURES_CACHE_SUBDIR, f"{song_id}_beat_features.pt")
    flow_data_cache_file = os.path.join(base_processed_dir, FLOW_DATA_CACHE_SUBDIR, f"{song_id}_flow_data.pt")
    stems_output_dir_for_song = os.path.join(base_processed_dir, STEMS_CACHE_SUBDIR) # BFE will create song-specific subdirs in here

    ensure_dir(os.path.join(base_processed_dir, BEAT_FEATURES_CACHE_SUBDIR))
    ensure_dir(os.path.join(base_processed_dir, FLOW_DATA_CACHE_SUBDIR))
    ensure_dir(stems_output_dir_for_song)


    # --- 1. Beat Feature Extraction ---
    song_beat_features: Optional[SongBeatFeatures] = None
    if not force_reprocess and os.path.exists(beat_features_cache_file):
        print(f"  Loading cached beat features for {song_id}...")
        try: song_beat_features = torch.load(beat_features_cache_file)
        except Exception as e: print(f"  Error loading cached beat features for {song_id}: {e}. Reprocessing.")
    
    if song_beat_features is None:
        print(f"  Extracting beat features for {song_id}...")
        song_beat_features = beat_extractor.extract_song_beat_features(
            audio_path=audio_file_path, stems_output_dir=stems_output_dir_for_song
        )
        if song_beat_features: torch.save(song_beat_features, beat_features_cache_file); print(f"  Saved beat features for {song_id}.")
        else: print(f"  Failed to extract beat features for {song_id}. Skipping."); return None
    if not song_beat_features: print(f"  No beat features for {song_id}. Skipping."); return None


    # --- 2. Flow Data Extraction ---
    song_flow_data: Optional[FlowData] = None
    if not alignment_file_path or not os.path.exists(alignment_file_path):
        print(f"  Alignment file not found for {song_id} at '{alignment_file_path}'. Cannot extract flow data. This song will not be used for flow model training.")
        return None # Essential for flow model targets
        
    if not force_reprocess and os.path.exists(flow_data_cache_file):
        print(f"  Loading cached flow data for {song_id}...")
        try: song_flow_data = torch.load(flow_data_cache_file)
        except Exception as e: print(f"  Error loading cached flow data for {song_id}: {e}. Reprocessing.")
            
    if song_flow_data is None:
        print(f"  Extracting flow data for {song_id} using alignment: {os.path.basename(alignment_file_path)}")
        song_flow_data = flow_extractor.extract_flow_for_song(
            alignment_data_path=alignment_file_path, song_beat_features=song_beat_features
        )
        if song_flow_data: torch.save(song_flow_data, flow_data_cache_file); print(f"  Saved flow data for {song_id}.")
        else: print(f"  Failed to extract flow data for {song_id}. Skipping."); return None
    if not song_flow_data: print(f"  No flow data for {song_id}. Skipping."); return None


    # --- 3. Tokenization ---
    print(f"  Tokenizing {song_id}...")
    try:
        token_ids, segment_ids, intra_line_pos_ids = tokenizer.encode_song_instance(
            song_beat_features, song_flow_data
        )
        if not token_ids: # encoding might return empty if data is unsuitable
            print(f"  Tokenization resulted in empty sequence for {song_id}. Skipping.")
            return None
        return {
            "song_id": song_id,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "intra_line_pos_ids": torch.tensor(intra_line_pos_ids, dtype=torch.long)
        }
    except Exception as e:
        print(f"  Error during tokenization for {song_id}: {e}"); return None


def create_dummy_dataset_files(base_dir, num_songs=2):
    """Creates dummy audio, lyrics, and TextGrid files for testing the script."""
    print(f"Creating dummy dataset files in {base_dir}...")
    raw_songs_d = os.path.join(base_dir, "raw_songs")
    lyrics_d = os.path.join(base_dir, "lyrics")
    alignments_d = os.path.join(base_dir, "alignments_textgrid")
    ensure_dir(raw_songs_d); ensure_dir(lyrics_d); ensure_dir(alignments_d)

    sr = 44100
    duration = 10 # seconds
    import soundfile as sf
    import numpy as np

    for i in range(1, num_songs + 1):
        song_id = f"dummy_song_{i}"
        # Audio
        audio_p = os.path.join(raw_songs_d, f"{song_id}.wav")
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wav_data = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.randn(len(t))
        sf.write(audio_p, wav_data.astype(np.float32), sr)
        # Lyrics
        lyrics_p = os.path.join(lyrics_d, f"{song_id}.txt")
        with open(lyrics_p, "w") as f:
            f.write(f"yo this is line one for {song_id}\nthis is another line quite fun for {song_id}\nshort third line\nand the last line is done for {song_id}")
        # Alignment (TextGrid) - very simplified
        tg_p = os.path.join(alignments_d, f"{song_id}.TextGrid")
        # Use content similar to flow_data_extractor's test, adjust xmax
        word_intervals = ""
        current_time = 0.5
        lyrics_lines = (open(lyrics_p).read()).split('\n')
        words_flat = [word for line in lyrics_lines for word in line.split()]
        
        for idx, word_text in enumerate(words_flat):
            start_t, end_t = current_time, current_time + np.random.uniform(0.2, 0.6)
            word_intervals += f'        intervals [{idx+1}]:\n            xmin = {start_t:.2f}\n            xmax = {end_t:.2f}\n            text = "{word_text}"\n'
            current_time = end_t + np.random.uniform(0.05, 0.2)
            if current_time > duration - 0.5: break # Stop if near end

        tg_content = f"""File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0 
xmax = {duration:.2f} 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "words" 
        xmin = 0 
        xmax = {duration:.2f}
        intervals: size = {idx+1 if words_flat else 0} 
{word_intervals}"""
        with open(tg_p, "w") as f: f.write(tg_content)
    print(f"Dummy dataset created with {num_songs} songs.")


def main(args):
    ensure_dir(args.processed_output_dir)

    # Check if dataset dirs exist, offer to create dummy if not
    if not os.path.exists(args.raw_songs_dir) or \
       not os.path.exists(args.lyrics_dir) or \
       not os.path.exists(args.alignments_dir):
        print("One or more dataset input directories are missing.")
        if input(f"Create dummy dataset in '{args.dataset_base_dir}' for testing? (y/n): ").lower() == 'y':
            create_dummy_dataset_files(args.dataset_base_dir)
        else:
            print("Exiting. Please provide valid dataset directories.")
            return

    beat_extractor = BeatFeatureExtractor(sample_rate=args.sample_rate) 
    flow_extractor = FlowDataExtractor(sample_rate_for_acapella=args.sample_rate)
    
    if not os.path.exists(args.tokenizer_config):
        print(f"Tokenizer config '{args.tokenizer_config}' not found. Building default and saving.")
        tokenizer = FlowTokenizer()
        tokenizer.save_vocab(args.tokenizer_config)
    else:
        tokenizer = FlowTokenizer(config_path=args.tokenizer_config)

    all_tokenized_song_data: List[Dict[str, Any]] = []
    audio_extensions = ('.wav', '.mp3', '.flac', '.m4a')
    song_files_to_process = []

    for filename in os.listdir(args.raw_songs_dir):
        if filename.lower().endswith(audio_extensions):
            song_id = os.path.splitext(filename)[0]
            audio_file_path = os.path.join(args.raw_songs_dir, filename)
            lyrics_file_path = os.path.join(args.lyrics_dir, f"{song_id}.txt") # For reference/MFA
            alignment_file_path = os.path.join(args.alignments_dir, f"{song_id}.TextGrid")
            song_files_to_process.append({
                "song_id": song_id, "audio_path": audio_file_path,
                "lyrics_path": lyrics_file_path, "alignment_path": alignment_file_path,
            })
    
    if not song_files_to_process: print(f"No audio files found in {args.raw_songs_dir}. Exiting."); return
    print(f"Found {len(song_files_to_process)} audio files to process.")

    for song_info in tqdm(song_files_to_process, desc="Processing Songs"):
        processed_data = process_song(
            song_id=song_info["song_id"], audio_file_path=song_info["audio_path"],
            alignment_file_path=song_info["alignment_path"],
            beat_extractor=beat_extractor, flow_extractor=flow_extractor, tokenizer=tokenizer,
            base_processed_dir=args.processed_output_dir, force_reprocess=args.force_reprocess
        )
        if processed_data: all_tokenized_song_data.append(processed_data)
    
    if all_tokenized_song_data:
        print(f"\nSuccessfully processed and tokenized {len(all_tokenized_song_data)} songs.")
        ensure_dir(os.path.dirname(args.output_tokenized_file))
        torch.save(all_tokenized_song_data, args.output_tokenized_file)
        print(f"Final tokenized dataset saved to: {args.output_tokenized_file}")
    else:
        print("\nNo songs were successfully processed. Output file not created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess rap dataset for Flow Transformer model.")
    parser.add_argument("--dataset_base_dir", type=str, default=DEFAULT_DATASET_BASE_DIR,
                        help="Base directory for the dataset (contains raw_songs, lyrics, alignments_textgrid, processed_output_dir).")
    parser.add_argument("--raw_songs_dir", type=str, default=DEFAULT_RAW_SONGS_DIR)
    parser.add_argument("--lyrics_dir", type=str, default=DEFAULT_LYRICS_DIR)
    parser.add_argument("--alignments_dir", type=str, default=DEFAULT_ALIGNMENTS_DIR)
    parser.add_argument("--processed_output_dir", type=str, default=DEFAULT_PROCESSED_OUTPUT_DIR)
    parser.add_argument("--tokenizer_config", type=str, default=DEFAULT_TOKENIZER_CONFIG)
    parser.add_argument("--output_tokenized_file", type=str, default=DEFAULT_OUTPUT_TOKENIZED_FILE)
    parser.add_argument("--sample_rate", type=int, default=44100, help="Target sample rate for audio processing.")
    parser.add_argument("--force_reprocess", action="store_true", help="Force reprocessing, ignore caches.")
    
    args = parser.parse_args()
    # Update default paths if base_dir is changed by user
    if args.dataset_base_dir != DEFAULT_DATASET_BASE_DIR:
        args.raw_songs_dir = os.path.join(args.dataset_base_dir, "raw_songs")
        args.lyrics_dir = os.path.join(args.dataset_base_dir, "lyrics")
        args.alignments_dir = os.path.join(args.dataset_base_dir, "alignments_textgrid")
        args.processed_output_dir = os.path.join(args.dataset_base_dir, "processed_for_transformer")
        args.output_tokenized_file = os.path.join(args.processed_output_dir, "tokenized_flow_dataset.pt")


    print("Starting dataset preprocessing with effective arguments:")
    for arg, value in vars(args).items(): print(f"  {arg}: {value}")
    
    main(args)