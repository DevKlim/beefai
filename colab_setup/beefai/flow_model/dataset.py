import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union
import os
import sys

class FlowDataset(Dataset):
    def __init__(self, 
                 tokenizer_pad_id: int, 
                 block_size: int,
                 data_file_path: Optional[str] = None, 
                 direct_data: Optional[List[Dict[str, Any]]] = None 
                ):
        self.pad_token_id = tokenizer_pad_id
        self.block_size = block_size 
        
        if data_file_path and direct_data:
            raise ValueError("Please provide either data_file_path or direct_data, not both.")
        if not data_file_path and not direct_data:
            raise ValueError("Either data_file_path or direct_data must be provided.")

        self.song_items: List[Dict[str, Any]] = []
        if data_file_path:
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found: {data_file_path}")
            print(f"Loading pre-tokenized data from {data_file_path}...")
            try:
                # Ensure weights_only=False if your .pt file contains non-tensor data like list of dicts
                loaded_data = torch.load(data_file_path, weights_only=False) 
                if isinstance(loaded_data, list):
                    self.song_items = loaded_data
                else:
                    # Handle older format if it was just a single tensor or dict.
                    # This part might need adjustment based on actual old formats.
                    print(f"Warning: Loaded data from {data_file_path} is not a list. Type: {type(loaded_data)}. Attempting to adapt if possible.")
                    if isinstance(loaded_data, dict) and all(k in loaded_data for k in ['token_ids', 'segment_ids', 'intra_line_pos_ids']):
                        self.song_items = [loaded_data] # Wrap it in a list
                    else:
                        raise ValueError(f"Unsupported data format in {data_file_path}. Expected list of dicts.")
            except Exception as e:
                raise ValueError(f"Error loading or parsing data from {data_file_path}: {e}")

        elif direct_data:
            print(f"Loading pre-tokenized data directly ({len(direct_data)} items)...")
            self.song_items = direct_data
        
        if not self.song_items:
            print("Warning: No song items loaded into FlowDataset.")
            self.examples = []
            return

        print(f"Loaded {len(self.song_items)} song items.")

        self.examples = []
        for song_idx, song_item in enumerate(self.song_items):
            if not isinstance(song_item, dict):
                print(f"Warning: Song item at index {song_idx} is not a dictionary (type: {type(song_item)}). Skipping.")
                continue

            token_ids_raw = song_item.get('token_ids')
            segment_ids_raw = song_item.get('segment_ids')
            intra_line_pos_ids_raw = song_item.get('intra_line_pos_ids')

            if token_ids_raw is None or segment_ids_raw is None or intra_line_pos_ids_raw is None:
                song_name_for_log = song_item.get('song_name', f'index {song_idx}')
                print(f"Warning: Song item '{song_name_for_log}' is missing one or more required keys ('token_ids', 'segment_ids', 'intra_line_pos_ids'). Skipping.")
                continue
            
            token_ids = torch.as_tensor(token_ids_raw, dtype=torch.long)
            segment_ids = torch.as_tensor(segment_ids_raw, dtype=torch.long)
            intra_line_pos_ids = torch.as_tensor(intra_line_pos_ids_raw, dtype=torch.long)

            if not (len(token_ids) == len(segment_ids) == len(intra_line_pos_ids)):
                song_name_for_log = song_item.get('song_name', f'index {song_idx}')
                print(f"Warning: Length mismatch in token sequences for song '{song_name_for_log}'. "
                      f"Tokens: {len(token_ids)}, Segments: {len(segment_ids)}, IntraPos: {len(intra_line_pos_ids)}. Skipping.")
                continue
            
            if len(token_ids) < self.block_size + 1 : # Song too short to make even one example
                 # print(f"Info: Song {song_idx} too short ({len(token_ids)} tokens) for block_size {self.block_size}. Skipping.")
                 continue


            seq_len_for_item = self.block_size + 1
            # Stride can be adjusted. self.block_size means no overlap.
            # self.block_size // 2 for 50% overlap.
            stride = self.block_size # Non-overlapping blocks for typical GPT training
            
            for i in range(0, len(token_ids) - seq_len_for_item + 1, stride): 
                chunk_tokens = token_ids[i : i + seq_len_for_item]
                chunk_segments = segment_ids[i : i + seq_len_for_item]
                chunk_intra_pos = intra_line_pos_ids[i : i + seq_len_for_item]
                
                if len(chunk_tokens) == seq_len_for_item: 
                    self.examples.append({
                        'full_chunk_tokens': chunk_tokens,
                        'full_chunk_segments': chunk_segments,
                        'full_chunk_intra_pos': chunk_intra_pos
                    })
        print(f"Created {len(self.examples)} training examples of block_size {self.block_size} from song items.")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if not self.examples:
            # This should not happen if constructor logic is correct and data is present
            raise IndexError("Dataset is empty or not properly initialized. No examples to fetch.")
        
        example = self.examples[idx]
        
        full_chunk_tokens = example['full_chunk_tokens']
        full_chunk_segments = example['full_chunk_segments']
        full_chunk_intra_pos = example['full_chunk_intra_pos']

        input_ids = full_chunk_tokens[:-1]
        target_ids = full_chunk_tokens[1:]
        
        input_segment_ids = full_chunk_segments[:-1]
        input_intra_line_pos_ids = full_chunk_intra_pos[:-1]
            
        return {
            "input_ids": input_ids,               # Shape: (block_size)
            "target_ids": target_ids,             # Shape: (block_size)
            "segment_ids": input_segment_ids,       # Shape: (block_size)
            "intra_line_pos_ids": input_intra_line_pos_ids # Shape: (block_size)
        }

# Example usage (assuming a script like 05a_tokenize_data_lite.py has created 'processed_flow_data.pt')
if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from beefai.flow_model.tokenizer import FlowTokenizer
    
    # Use a real or well-defined dummy tokenizer config
    tokenizer_config_for_test = os.path.join(os.path.dirname(__file__), "flow_tokenizer_config_v2.json")
    if not os.path.exists(tokenizer_config_for_test):
        print(f"Warning: Test tokenizer config '{tokenizer_config_for_test}' not found. Creating a default one for testing.")
        temp_tokenizer = FlowTokenizer() # Builds default vocab
        os.makedirs(os.path.dirname(tokenizer_config_for_test), exist_ok=True)
        temp_tokenizer.save_vocab(tokenizer_config_for_test)
        
    tokenizer_instance = FlowTokenizer(config_path=tokenizer_config_for_test)
    pad_id = tokenizer_instance.pad_token_id
    vocab_size_for_dummy = tokenizer_instance.get_vocab_size()

    dummy_data_path = "dummy_flow_dataset_test_data.pt"
    if not os.path.exists(dummy_data_path):
        print(f"Creating dummy data file: {dummy_data_path}")
        # Ensure dummy data is long enough for block_size
        dummy_song_1_tokens = torch.randint(0, vocab_size_for_dummy, (2000,)) # Longer sequence
        dummy_song_1_segments = torch.randint(0, 8, (2000,)) 
        dummy_song_1_intra_pos = torch.randint(0, 4, (2000,))

        dummy_song_2_tokens = torch.randint(0, vocab_size_for_dummy, (1800,)) # Longer sequence
        dummy_song_2_segments = torch.randint(0, 8, (1800,))
        dummy_song_2_intra_pos = torch.randint(0, 4, (1800,))
        
        dummy_data_list = [
            {'token_ids': dummy_song_1_tokens, 'segment_ids': dummy_song_1_segments, 'intra_line_pos_ids': dummy_song_1_intra_pos, 'song_name': 'dummy1'},
            {'token_ids': dummy_song_2_tokens, 'segment_ids': dummy_song_2_segments, 'intra_line_pos_ids': dummy_song_2_intra_pos, 'song_name': 'dummy2'}
        ]
        torch.save(dummy_data_list, dummy_data_path)

    test_block_size = 128 
    print(f"\n--- Testing FlowDataset loading from file (block_size={test_block_size}) ---")
    dataset_from_file = FlowDataset(
        data_file_path=dummy_data_path,
        tokenizer_pad_id=pad_id,
        block_size=test_block_size
    )
    print(f"FlowDataset (from file) created with {len(dataset_from_file)} examples.")
    
    if len(dataset_from_file) > 0:
        sample_item = dataset_from_file[0]
        print("\nSample item from dataset (from file):")
        for key, value in sample_item.items():
            print(f"  {key} shape: {value.shape}, dtype: {value.dtype}")
            assert value.shape[0] == test_block_size, f"Shape mismatch for {key}"
    else:
        print("No examples generated from file dataset, check data and block_size.")


    print("\n--- Testing FlowDataset loading directly ---")
    loaded_dummy_data = torch.load(dummy_data_path, weights_only=False)
    dataset_direct = FlowDataset(
        direct_data=loaded_dummy_data,
        tokenizer_pad_id=pad_id,
        block_size=test_block_size
    )
    print(f"FlowDataset (direct) created with {len(dataset_direct)} examples.")
    if len(dataset_direct) > 0:
        sample_item_direct = dataset_direct[0]
        print("\nSample item from dataset (direct):")
        for key, value in sample_item_direct.items():
            print(f"  {key} shape: {value.shape}")
    else:
        print("No examples generated from direct dataset, check data and block_size.")
    
    if os.path.exists(dummy_data_path):
        os.remove(dummy_data_path)
        print(f"\nRemoved dummy data file: {dummy_data_path}")