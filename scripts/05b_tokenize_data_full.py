import torch
import os
import yaml
import random
from tqdm import tqdm
from typing import List, Dict, Any
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from beefai.flow_model.tokenizer import FlowTokenizer
from beefai.utils.data_types import TrainingInstance # Make sure this type hint matches your actual structure

# --- Configuration Loading ---
def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def tokenize_data_full():
    DATA_CONFIG_FULL_PATH = "lite_model_training/data_config_full.yaml"
    MODEL_CONFIG_FULL_PATH = "lite_model_training/model_config_full.yaml" # For model limits

    print(f"--- Tokenizing Data for FULL Model ---")
    print(f"Loading FULL data config from: {DATA_CONFIG_FULL_PATH}")
    if not os.path.exists(DATA_CONFIG_FULL_PATH):
        print(f"ERROR: Data config file not found: {DATA_CONFIG_FULL_PATH}")
        return
    data_config = load_yaml_config(DATA_CONFIG_FULL_PATH)
    
    print(f"Loading FULL model config from: {MODEL_CONFIG_FULL_PATH}")
    if not os.path.exists(MODEL_CONFIG_FULL_PATH):
        print(f"ERROR: Model config file not found: {MODEL_CONFIG_FULL_PATH}")
        return
    model_config = load_yaml_config(MODEL_CONFIG_FULL_PATH)

    tokenizer_path = data_config.get("tokenizer_path")
    if not tokenizer_path:
        print(f"ERROR: 'tokenizer_path' not specified in {DATA_CONFIG_FULL_PATH}")
        return

    processed_data_source_dir = data_config.get("processed_data_source_dir") 
    if not processed_data_source_dir:
        print(f"ERROR: 'processed_data_source_dir' not specified in {DATA_CONFIG_FULL_PATH}")
        return
    # Use a default filename if not specified, but encourage user to specify it in YAML
    processed_data_filename = data_config.get("processed_data_filename", "processed_training_data.pt")
    processed_data_file = os.path.join(processed_data_source_dir, processed_data_filename) 
    
    tokenized_output_dir = data_config.get("tokenized_data_output_dir")
    if not tokenized_output_dir:
        print(f"ERROR: 'tokenized_data_output_dir' not specified in {DATA_CONFIG_FULL_PATH}")
        return
    train_output_filename = data_config.get("train_data_filename", "train_full.pt")
    val_output_filename = data_config.get("val_data_filename", "val_full.pt")
    train_output_path = os.path.join(tokenized_output_dir, train_output_filename)
    val_output_path = os.path.join(tokenized_output_dir, val_output_filename)
    
    max_songs_for_full = data_config.get("max_songs_for_full", -1) 
    val_split_ratio_for_full = data_config.get("val_split_ratio_for_full", 0.1)

    config_max_segment_types = model_config.get("max_segment_types")
    config_max_intra_line_positions = model_config.get("max_intra_line_positions")
    if config_max_segment_types is None or config_max_intra_line_positions is None:
        print(f"ERROR: 'max_segment_types' or 'max_intra_line_positions' not found in {MODEL_CONFIG_FULL_PATH}")
        return

    print(f"Using limits for checks from model config: max_segment_types={config_max_segment_types}, max_intra_line_positions={config_max_intra_line_positions}")

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer config not found at {tokenizer_path}. ")
        print(f"Please ensure it exists (e.g., created by FlowTokenizer if run standalone, or by a prior pipeline step).")
        return
    
    tokenizer = FlowTokenizer(config_path=tokenizer_path)
    print(f"FlowTokenizer loaded from {tokenizer_path}. Vocabulary size: {tokenizer.get_vocab_size()}")

    if not os.path.exists(processed_data_file):
        print(f"ERROR: Processed data file not found: {processed_data_file}. ")
        print(f"Please run scripts/preprocess_dataset.py (or ensure 'processed_data_source_dir' and 'processed_data_filename' in {DATA_CONFIG_FULL_PATH} are correct).")
        return
    
    print(f"Loading processed data instances from: {processed_data_file}...")
    try:
        # Ensure weights_only=False if your .pt file contains non-tensor data (like list of dicts)
        all_song_instances: List[TrainingInstance] = torch.load(processed_data_file, weights_only=False)
    except Exception as e:
        print(f"ERROR loading {processed_data_file}: {e}")
        return
        
    print(f"Loaded {len(all_song_instances)} processed song instances.")

    if not all_song_instances:
        print(f"ERROR: No song instances found in {processed_data_file}. Cannot proceed.")
        return

    if max_songs_for_full > 0 and len(all_song_instances) > max_songs_for_full :
        print(f"Randomly selecting {max_songs_for_full} songs for the FULL dataset from {len(all_song_instances)} available.")
        selected_instances = random.sample(all_song_instances, max_songs_for_full)
    else:
        print(f"Using all {len(all_song_instances)} available songs for the FULL dataset.")
        selected_instances = all_song_instances

    tokenized_songs_for_dataset: List[Dict[str, torch.Tensor]] = []
    skipped_count = 0
    
    print(f"Tokenizing {len(selected_instances)} selected song instances for FULL dataset...")
    for i, song_instance_dict in enumerate(tqdm(selected_instances, desc="Tokenizing for Full Dataset")):
        song_name = song_instance_dict.get('song_name', f"UnknownSong_{i}") 
        beat_features = song_instance_dict.get('beat_features')
        flow_data = song_instance_dict.get('flow_data')

        if not beat_features and not flow_data:
            print(f"WARNING for {song_name}: Both beat_features and flow_data are missing or empty. Skipping.")
            skipped_count += 1
            continue
        
        current_beat_features = beat_features if beat_features is not None else []
        current_flow_data = flow_data if flow_data is not None else []

        try:
            token_ids, segment_ids, intra_line_pos_ids = tokenizer.encode_song_instance(
                song_beat_features=current_beat_features, 
                song_flow_data=current_flow_data
            )
        except Exception as e:
            print(f"ERROR during tokenization of {song_name}: {e}. Skipping.")
            skipped_count += 1
            continue
        
        if not token_ids: 
            print(f"WARNING for {song_name}: Tokenization resulted in empty token_ids. Skipping.")
            skipped_count += 1
            continue

        max_observed_segment_id = max(segment_ids) if segment_ids else -1
        max_observed_intra_pos_id = max(intra_line_pos_ids) if intra_line_pos_ids else -1

        song_is_valid = True
        if max_observed_segment_id >= config_max_segment_types:
            print(f"  ERROR in {song_name}: Max observed segment_id {max_observed_segment_id} >= model's max_segment_types {config_max_segment_types}. Skipping.")
            song_is_valid = False
        
        if max_observed_intra_pos_id >= config_max_intra_line_positions:
            print(f"  ERROR in {song_name}: Max observed intra_line_pos_id {max_observed_intra_pos_id} >= model's max_intra_line_positions {config_max_intra_line_positions}. Skipping.")
            song_is_valid = False
            
        if not song_is_valid:
            skipped_count += 1
            continue

        tokenized_songs_for_dataset.append({
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            'intra_line_pos_ids': torch.tensor(intra_line_pos_ids, dtype=torch.long),
            'song_name': song_name 
        })

    print(f"Tokenization finished. Successfully prepared {len(tokenized_songs_for_dataset)} songs for the FULL dataset.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} songs due to errors, missing data, or exceeding model context limits.")

    if not tokenized_songs_for_dataset:
        print("ERROR: No songs were successfully tokenized and validated. Cannot create train/val splits for FULL dataset.")
        return

    if not (max_songs_for_full > 0 and len(all_song_instances) > max_songs_for_full): 
        random.shuffle(tokenized_songs_for_dataset)


    num_val = int(len(tokenized_songs_for_dataset) * val_split_ratio_for_full)
    num_train = len(tokenized_songs_for_dataset) - num_val

    train_data_list = tokenized_songs_for_dataset[:num_train]
    val_data_list = tokenized_songs_for_dataset[num_train:]

    print(f"Number of training song sequences for FULL model: {len(train_data_list)}")
    print(f"Number of validation song sequences for FULL model: {len(val_data_list)}")

    os.makedirs(tokenized_output_dir, exist_ok=True)

    if train_data_list:
        print(f"Saving FULL training data to: {train_output_path}")
        torch.save(train_data_list, train_output_path)
    else:
        print("Warning: No training data to save for FULL model (list is empty).")

    if val_data_list:
        print(f"Saving FULL validation data to: {val_output_path}")
        torch.save(val_data_list, val_output_path)
    elif val_split_ratio_for_full > 0: 
        print(f"Warning: No validation data to save for FULL model (list is empty, val_split_ratio={val_split_ratio_for_full}).")
    
    print("FULL data tokenization and saving process complete.")

if __name__ == "__main__":
    tokenize_data_full()