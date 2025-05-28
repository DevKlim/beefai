import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import os

class FlowDataset(Dataset):
    def __init__(self, 
                 data_file_path: str, # Path to a .pt file containing list of tokenized sequences
                 tokenizer_pad_id: int, # Pass tokenizer.pad_token_id
                 block_size: int
                ):
        self.pad_token_id = tokenizer_pad_id
        self.block_size = block_size # Max sequence length for the model
        
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Data file not found: {data_file_path}")
        
        print(f"Loading pre-tokenized data from {data_file_path}...")
        # Data is expected to be a list of dictionaries, where each dict is a song instance:
        # Each song instance: {
        #    'token_ids': List[int], 
        #    'segment_ids': List[int], 
        #    'intra_line_pos_ids': List[int]
        # }
        # These lists are for the *entire song sequence*.
        self.song_items = torch.load(data_file_path)
        print(f"Loaded {len(self.song_items)} song items.")

        # We will further process these song_items into fixed-size blocks for training.
        # Each song can yield multiple training examples by sliding a window of `block_size`.
        self.examples = []
        for song_item in self.song_items:
            token_ids = song_item['token_ids']
            segment_ids = song_item['segment_ids']
            intra_line_pos_ids = song_item['intra_line_pos_ids']
            
            # Ensure they are tensors
            if not isinstance(token_ids, torch.Tensor): token_ids = torch.tensor(token_ids, dtype=torch.long)
            if not isinstance(segment_ids, torch.Tensor): segment_ids = torch.tensor(segment_ids, dtype=torch.long)
            if not isinstance(intra_line_pos_ids, torch.Tensor): intra_line_pos_ids = torch.tensor(intra_line_pos_ids, dtype=torch.long)

            # Create overlapping sequences of length `block_size + 1`
            # The `+1` is because input is seq[:-1] and target is seq[1:]
            seq_len_for_item = self.block_size + 1
            for i in range(0, len(token_ids) - seq_len_for_item + 1, seq_len_for_item // 2): # Overlap by 50%
                chunk_tokens = token_ids[i : i + seq_len_for_item]
                chunk_segments = segment_ids[i : i + seq_len_for_item]
                chunk_intra_pos = intra_line_pos_ids[i : i + seq_len_for_item]
                
                if len(chunk_tokens) == seq_len_for_item: # Ensure full chunk
                    self.examples.append({
                        'full_chunk_tokens': chunk_tokens,
                        'full_chunk_segments': chunk_segments,
                        'full_chunk_intra_pos': chunk_intra_pos
                    })
        print(f"Created {len(self.examples)} training examples of block_size {self.block_size} from song items.")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        full_chunk_tokens = example['full_chunk_tokens']
        full_chunk_segments = example['full_chunk_segments']
        full_chunk_intra_pos = example['full_chunk_intra_pos']

        # Input is seq[:-1], Target is seq[1:]
        input_ids = full_chunk_tokens[:-1]
        target_ids = full_chunk_tokens[1:]
        
        # Context IDs are for the input part
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
    # This part requires a dummy 'processed_flow_data.pt' to run
    # And a tokenizer to get pad_id.
    from beefai.flow_model.tokenizer import FlowTokenizer
    
    # Create a dummy tokenizer for pad_id
    dummy_tokenizer = FlowTokenizer() # Builds default vocab
    dummy_pad_id = dummy_tokenizer.pad_token_id

    # Create a dummy data file for testing FlowDataset
    dummy_data_path = "dummy_processed_flow_data.pt"
    if not os.path.exists(dummy_data_path):
        print(f"Creating dummy data file: {dummy_data_path}")
        # Simulate output of the (future) tokenization script
        # Each item is a song, with long sequences of token_ids, segment_ids, etc.
        # vocab_size = dummy_tokenizer.get_vocab_size()
        dummy_song_1_tokens = torch.randint(0, dummy_tokenizer.get_vocab_size(), (1000,))
        dummy_song_1_segments = torch.randint(0, 8, (1000,)) # Example max_segment_types
        dummy_song_1_intra_pos = torch.randint(0, 4, (1000,)) # Example max_intra_line_positions

        dummy_song_2_tokens = torch.randint(0, dummy_tokenizer.get_vocab_size(), (800,))
        dummy_song_2_segments = torch.randint(0, 8, (800,))
        dummy_song_2_intra_pos = torch.randint(0, 4, (800,))
        
        dummy_data_list = [
            {'token_ids': dummy_song_1_tokens, 'segment_ids': dummy_song_1_segments, 'intra_line_pos_ids': dummy_song_1_intra_pos},
            {'token_ids': dummy_song_2_tokens, 'segment_ids': dummy_song_2_segments, 'intra_line_pos_ids': dummy_song_2_intra_pos}
        ]
        torch.save(dummy_data_list, dummy_data_path)

    test_block_size = 128
    dataset = FlowDataset(
        data_file_path=dummy_data_path,
        tokenizer_pad_id=dummy_pad_id,
        block_size=test_block_size
    )
    print(f"\nFlowDataset created with {len(dataset)} examples.")
    
    if len(dataset) > 0:
        sample_item = dataset[0]
        print("\nSample item from dataset:")
        print(f"  Input IDs shape: {sample_item['input_ids'].shape}")
        print(f"  Target IDs shape: {sample_item['target_ids'].shape}")
        print(f"  Segment IDs shape: {sample_item['segment_ids'].shape}")
        print(f"  Intra-Line Pos IDs shape: {sample_item['intra_line_pos_ids'].shape}")

        # Verify no padding tokens in target_ids (or ensure loss function ignores them)
        # print(f"  Target IDs sample: {sample_item['target_ids'][:10]}")
        # print(f"  Pad token ID: {dummy_pad_id}")
        
        # Example DataLoader
        # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        # for batch in dataloader:
        #     print("\nBatch shapes:")
        #     print(f"  Input IDs: {batch['input_ids'].shape}")
        #     # ... and so on for other elements in batch
        #     break