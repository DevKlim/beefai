import torch
import os
import json
import random
from tqdm import tqdm
import yaml
import sys
from typing import List, Tuple, Dict, Any, Optional

# Ensure beefai modules can be imported
sys.path.append(os.getcwd())

from beefai.flow_model.tokenizer import FlowTokenizer
from beefai.utils.data_types import SongBeatFeatures, FlowData # For type hints

# --- Configuration Loading ---
def load_yaml_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

DATA_CONFIG_LITE_PATH = "lite_model_training/data_config_lite.yaml"
MODEL_CONFIG_LITE_PATH = "lite_model_training/model_config_lite.yaml" # Not strictly needed here, but for consistency

data_params_lite = load_yaml_config(DATA_CONFIG_LITE_PATH)
# model_params_lite = load_yaml_config(MODEL_CONFIG_LITE_PATH) # block_size not used directly here

# --- Parameters ---
TOKENIZER_PATH = data_params_lite["tokenizer_path"] # Reuse main tokenizer

# Source of processed data (caches from preprocess_dataset.py)
PROCESSED_DATA_SOURCE_DIR = data_params_lite.get("processed_data_source_dir", "data/processed/")
BEAT_FEATURES_CACHE_SUBDIR = "beat_features_cache" # Matching preprocess_dataset.py
FLOW_DATA_CACHE_SUBDIR = "flow_data_cache"         # Matching preprocess_dataset.py


# Target for lite tokenized data
LITE_TOKENIZED_DATA_DIR = os.path.dirname(data_params_lite["train_data_path"])
os.makedirs(LITE_TOKENIZED_DATA_DIR, exist_ok=True)

# Parameters for subsetting data for the lite version
MAX_SONGS_LITE = data_params_lite.get("max_songs_for_lite", 10) # Reduced for faster lite processing
VAL_SPLIT_RATIO = 0.2 # 20% for validation with smaller dataset


def create_lite_tokenized_dataset():
    """
    Creates a tokenized dataset for the lite model by selecting a subset of songs
    from the cached processed features and tokenizing them.
    The output is compatible with FlowDataset, meaning it saves full song sequences.
    """
    print("--- Creating Lite Tokenized Dataset from Cached Features---")
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer config not found: {TOKENIZER_PATH}. Run main tokenizer saving first.")
        return
    tokenizer = FlowTokenizer(config_path=TOKENIZER_PATH)

    beat_features_base_path = os.path.join(PROCESSED_DATA_SOURCE_DIR, BEAT_FEATURES_CACHE_SUBDIR)
    flow_data_base_path = os.path.join(PROCESSED_DATA_SOURCE_DIR, FLOW_DATA_CACHE_SUBDIR)

    if not os.path.isdir(beat_features_base_path) or not os.path.isdir(flow_data_base_path):
        print(f"Processed data cache directories not found in {PROCESSED_DATA_SOURCE_DIR}. ")
        print(f"Ensure '{BEAT_FEATURES_CACHE_SUBDIR}' and '{FLOW_DATA_CACHE_SUBDIR}' exist and contain .pt files from 'scripts/preprocess_dataset.py'.")
        return

    # Get song_ids from available cached beat features (assuming flow data will also exist for these)
    available_song_ids = []
    for f_name in os.listdir(beat_features_base_path):
        if f_name.endswith("_beat_features.pt"):
            song_id = f_name.replace("_beat_features.pt", "")
            # Check if corresponding flow data also exists
            if os.path.exists(os.path.join(flow_data_base_path, f"{song_id}_flow_data.pt")):
                available_song_ids.append(song_id)
            else:
                print(f"Warning: Beat features found for {song_id}, but no flow data. Skipping.")
    
    if not available_song_ids:
        print(f"No processed song feature pairs found in cache at {PROCESSED_DATA_SOURCE_DIR}.")
        return
    
    print(f"Found {len(available_song_ids)} songs with cached features. Selecting up to {MAX_SONGS_LITE} for lite dataset.")
    if len(available_song_ids) > MAX_SONGS_LITE:
        selected_song_ids = random.sample(available_song_ids, MAX_SONGS_LITE)
    else:
        selected_song_ids = available_song_ids
    print(f"Using {len(selected_song_ids)} songs for lite dataset.")

    all_tokenized_song_data_for_lite = []

    for song_id in tqdm(selected_song_ids, desc="Processing songs for lite dataset"):
        beat_features_file = os.path.join(beat_features_base_path, f"{song_id}_beat_features.pt")
        flow_data_file = os.path.join(flow_data_base_path, f"{song_id}_flow_data.pt")

        try:
            song_beat_features: Optional[SongBeatFeatures] = torch.load(beat_features_file)
            song_flow_data: Optional[FlowData] = torch.load(flow_data_file)
        except Exception as e:
            print(f"Skipping {song_id}: Error loading cached .pt files - {e}")
            continue

        if not song_beat_features or not song_flow_data:
            print(f"Skipping {song_id}: Missing beat or flow features after loading.")
            continue
        
        try:
            # Use the tokenizer's method to get all necessary ID sequences for the full song
            token_ids, segment_ids, intra_line_pos_ids = tokenizer.encode_song_instance(
                song_beat_features, song_flow_data
            )
            if not token_ids:
                print(f"Tokenization resulted in empty sequence for {song_id}. Skipping.")
                continue

            all_tokenized_song_data_for_lite.append({
                "song_id": song_id, # Keep for reference if needed
                "token_ids": torch.tensor(token_ids, dtype=torch.long),
                "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
                "intra_line_pos_ids": torch.tensor(intra_line_pos_ids, dtype=torch.long),
            })
        except Exception as e:
            print(f"Error tokenizing {song_id}: {e}. Skipping.")
            continue
    
    if not all_tokenized_song_data_for_lite:
        print("No items were tokenized for the lite dataset. Check source data and processing logic.")
        return

    # Split into train and validation
    random.shuffle(all_tokenized_song_data_for_lite)
    val_size = int(len(all_tokenized_song_data_for_lite) * VAL_SPLIT_RATIO)
    train_items = all_tokenized_song_data_for_lite[val_size:]
    val_items = all_tokenized_song_data_for_lite[:val_size]

    train_file_path = data_params_lite["train_data_path"]
    val_file_path = data_params_lite.get("val_data_path")

    torch.save(train_items, train_file_path)
    print(f"Saved {len(train_items)} lite training song items to {train_file_path}")
    if val_items and val_file_path:
        torch.save(val_items, val_file_path)
        print(f"Saved {len(val_items)} lite validation song items to {val_file_path}")
    elif val_items:
        print(f"Lite validation items generated ({len(val_items)}) but no val_file_path specified in data_config_lite.yaml.")

    print("Lite dataset tokenization from cached features complete.")

if __name__ == "__main__":
    create_lite_tokenized_dataset()