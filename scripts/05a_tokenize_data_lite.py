import torch
import os
import yaml
import random
from tqdm import tqdm
from typing import List, Dict, Any
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from beefai.flow_model.tokenizer import FlowTokenizer
from beefai.utils.data_types import TrainingInstance # Make sure this type hint matches your actual structure

# --- Configuration Loading ---
def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def tokenize_data_lite():
    DATA_CONFIG_LITE_PATH = "lite_model_training/data_config_lite.yaml"
    # Note: This script didn't previously load model_config_lite.yaml,
    # which is likely why the check used an old/default max_segment_types.
    # We will add loading for model_config_lite.yaml.
    MODEL_CONFIG_LITE_PATH = "lite_model_training/model_config_lite.yaml"


    print(f"--- Tokenizing Data for LITE Model ---")
    print(f"Loading LITE data config from: {DATA_CONFIG_LITE_PATH}")
    data_config = load_yaml_config(DATA_CONFIG_LITE_PATH)
    
    print(f"Loading LITE model config from: {MODEL_CONFIG_LITE_PATH}")
    model_config = load_yaml_config(MODEL_CONFIG_LITE_PATH) # LOAD MODEL CONFIG

    tokenizer_path = data_config["tokenizer_path"]
    processed_data_source_dir = data_config["processed_data_source_dir"]
    # Assuming the main processed file is named 'processed_training_data.pt' as per CONTEXT.md
    processed_data_file = os.path.join(processed_data_source_dir, "processed_training_data.pt") 
    
    tokenized_output_dir = data_config["tokenized_data_output_dir"]
    train_output_path = data_config["train_data_path"]
    val_output_path = data_config["val_data_path"]
    
    max_songs_for_lite = data_config.get("max_songs_for_lite", 10) # Default to 10 if not in config
    val_split_ratio_for_lite = data_config.get("val_split_ratio_for_lite", 0.2) # Default

    # Get max_segment_types and max_intra_line_positions from the loaded MODEL config
    # These values are what the FlowTransformerDecoder will be configured with.
    # The tokenizer generates segment/intra-line IDs based on song structure.
    # This script checks if these generated IDs are within the model's capacity.
    # These keys MUST exist in your model_config_lite.yaml after your Step 1 changes.
    config_max_segment_types = model_config["max_segment_types"]
    config_max_intra_line_positions = model_config["max_intra_line_positions"]

    print(f"Using limits for checks from model config: max_segment_types={config_max_segment_types}, max_intra_line_positions={config_max_intra_line_positions}")


    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer config not found at {tokenizer_path}. "
              f"Please run the data pipeline (Phases 0-3, e.g., scripts/preprocess_dataset.py) "
              f"which should generate/save this tokenizer config via FlowTokenizer.save_vocab().")
        return
    
    tokenizer = FlowTokenizer(config_path=tokenizer_path)
    print(f"FlowTokenizer loaded. Vocabulary size: {tokenizer.get_vocab_size()}")

    if not os.path.exists(processed_data_file):
        print(f"ERROR: Processed data file not found: {processed_data_file}. "
              f"Please run scripts/preprocess_dataset.py to generate this file.")
        return
    
    print(f"Loading processed data instances from: {processed_data_file}...")
    # The structure of items in 'processed_training_data.pt' should be:
    # List[Dict[str, Any]], where each dict is a TrainingInstance like:
    # {'song_name': str, 'beat_features': SongBeatFeatures, 'flow_data': FlowData}
    all_song_instances: List[Dict[str, Any]] = torch.load(processed_data_file, weights_only=False) # ensure non-tensor data like song_name is loaded
    print(f"Loaded {len(all_song_instances)} processed song instances.")

    if not all_song_instances:
        print(f"ERROR: No song instances found in {processed_data_file}. Cannot proceed.")
        return

    # Select a subset of songs for the "lite" dataset
    if max_songs_for_lite > 0 and max_songs_for_lite < len(all_song_instances):
        print(f"Randomly selecting {max_songs_for_lite} songs for the LITE dataset.")
        # For reproducibility, you might want to set random.seed() here
        # random.seed(42) 
        selected_instances = random.sample(all_song_instances, max_songs_for_lite)
    else:
        print(f"Using all {len(all_song_instances)} available songs for the LITE dataset.")
        selected_instances = all_song_instances
        # Shuffle all instances if all are used, to ensure train/val split is random
        # random.seed(42) # For reproducibility
        random.shuffle(selected_instances)

    tokenized_songs_for_dataset: List[Dict[str, torch.Tensor]] = []
    skipped_count = 0
    
    print(f"Tokenizing {len(selected_instances)} selected song instances for LITE dataset...")
    for i, song_instance_dict in enumerate(tqdm(selected_instances, desc="Tokenizing for Lite Dataset")):
        # song_instance_dict is expected to be a TrainingInstance
        song_name = song_instance_dict.get('song_name', f"UnknownSong_{i}")
        beat_features = song_instance_dict.get('beat_features') # SongBeatFeatures (List[BarBeatFeatures])
        flow_data = song_instance_dict.get('flow_data')         # FlowData (List[FlowDatum])

        if not beat_features and not flow_data:
            print(f"WARNING for {song_name}: Both beat_features and flow_data are missing or empty. Skipping.")
            skipped_count += 1
            continue
        
        # Ensure they are lists, even if empty, for tokenizer
        if beat_features is None: beat_features = []
        if flow_data is None: flow_data = []

        try:
            # Tokenize the song instance
            token_ids, segment_ids, intra_line_pos_ids = tokenizer.encode_song_instance(
                song_beat_features=beat_features, 
                song_flow_data=flow_data
            )
        except Exception as e:
            print(f"ERROR during tokenization of {song_name}: {e}. Skipping.")
            skipped_count += 1
            continue
        
        if not token_ids: # Should not happen if inputs were valid
            print(f"WARNING for {song_name}: Tokenization resulted in empty token_ids. Skipping.")
            skipped_count += 1
            continue

        # --- Perform the CRITICAL check against model's configured limits ---
        max_observed_segment_id = -1
        if segment_ids: # segment_ids is List[int]
            max_observed_segment_id = max(segment_ids)
        
        max_observed_intra_pos_id = -1
        if intra_line_pos_ids: # intra_line_pos_ids is List[int]
            max_observed_intra_pos_id = max(intra_line_pos_ids)

        song_is_valid = True
        if max_observed_segment_id >= config_max_segment_types:
            print(f"  ERROR in {song_name}: Max segment_id {max_observed_segment_id} >= model's max_segment_types {config_max_segment_types}. Skipping.")
            song_is_valid = False
        
        if max_observed_intra_pos_id >= config_max_intra_line_positions:
            print(f"  ERROR in {song_name}: Max intra_line_pos_id {max_observed_intra_pos_id} >= model's max_intra_line_positions {config_max_intra_line_positions}. Skipping.")
            song_is_valid = False
            
        if not song_is_valid:
            skipped_count += 1
            continue # Skip this song

        # If valid, add to list for dataset
        # The FlowDataset expects each item to be a dict of tensors:
        # {'token_ids': Tensor, 'segment_ids': Tensor, 'intra_line_pos_ids': Tensor}
        tokenized_songs_for_dataset.append({
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            'intra_line_pos_ids': torch.tensor(intra_line_pos_ids, dtype=torch.long)
            # 'song_name': song_name # Optional: for debugging, but not used by FlowDataset
        })

    print(f"Tokenization finished. Successfully prepared {len(tokenized_songs_for_dataset)} songs for the LITE dataset.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} songs due to errors or exceeding model limits.")

    if not tokenized_songs_for_dataset:
        print("ERROR: No songs were successfully tokenized and validated. Cannot create train/val splits.")
        return

    # Split into training and validation sets
    num_val = int(len(tokenized_songs_for_dataset) * val_split_ratio_for_lite)
    num_train = len(tokenized_songs_for_dataset) - num_val

    # Already shuffled if all songs were used, or random.sample was used.
    train_data_list = tokenized_songs_for_dataset[:num_train]
    val_data_list = tokenized_songs_for_dataset[num_train:]

    print(f"Number of training song sequences for LITE model: {len(train_data_list)}")
    print(f"Number of validation song sequences for LITE model: {len(val_data_list)}")

    # Create output directory if it doesn't exist
    os.makedirs(tokenized_output_dir, exist_ok=True)

    # Save the datasets
    if train_data_list:
        print(f"Saving LITE training data to: {train_output_path}")
        torch.save(train_data_list, train_output_path)
    else:
        print("Warning: No training data to save for LITE model.")

    if val_data_list:
        print(f"Saving LITE validation data to: {val_output_path}")
        torch.save(val_data_list, val_output_path)
    elif val_split_ratio_for_lite > 0: # Only warn if val set was expected
        print(f"Warning: No validation data to save for LITE model (val_split_ratio={val_split_ratio_for_lite}).")
    
    print("LITE data tokenization and saving process complete.")

if __name__ == "__main__":
    # Ensure random seed if strict reproducibility is needed for song selection/shuffling
    # random.seed(1234) 
    tokenize_data_lite()