import torch
import os
import argparse
import sys
from typing import List, Dict, Any, Optional # Added Optional

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Attempt to import FlowTokenizer, but make it optional for basic inspection
try:
    from beefai.flow_model.tokenizer import FlowTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: FlowTokenizer not found. Token decoding will not be available.")
    print("Ensure beefai package is in PYTHONPATH if you need token decoding.")


def inspect_pt_file(file_path: str, tokenizer_config_path: Optional[str] = None, num_sequences_to_show: int = 3, show_tokens: bool = False, max_tokens_to_decode: int = 50):
    """
    Inspects a .pt file containing tokenized song data.

    Args:
        file_path (str): Path to the .pt file (e.g., train_lite.pt).
        tokenizer_config_path (str, optional): Path to the FlowTokenizer config JSON.
                                              Required if show_tokens is True.
        num_sequences_to_show (int): Number of song sequences to display details for.
        show_tokens (bool): If True, attempts to decode and print the first few tokens.
        max_tokens_to_decode (int): Max number of tokens to decode and show per sequence.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"\n--- Inspecting: {file_path} ---")
    
    try:
        # Ensure weights_only=False for loading lists of dicts with tensors
        data: List[Any] = torch.load(file_path, weights_only=False) 
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"Error: Expected data to be a list of dictionaries, but got type {type(data)}.")
        if hasattr(data, 'keys'): # Check if it's a single dictionary (older format perhaps)
            print(f"  It appears to be a single dictionary with keys: {list(data.keys())}")
            print("  Assuming this single dictionary is the item to inspect.")
            data = [data] # Wrap it in a list to proceed
        else:
            return # Cannot proceed if not list or adaptable dict
    
    num_total_sequences = len(data)
    print(f"Total items (song sequences/elements) in the list: {num_total_sequences}")

    if num_total_sequences == 0:
        print("File contains an empty list.")
        return

    tokenizer_instance = None
    if show_tokens and TOKENIZER_AVAILABLE:
        if tokenizer_config_path:
            if os.path.exists(tokenizer_config_path):
                try:
                    tokenizer_instance = FlowTokenizer(config_path=tokenizer_config_path)
                    print(f"Tokenizer loaded from {tokenizer_config_path} for decoding.")
                except Exception as e:
                    print(f"Error loading tokenizer from {tokenizer_config_path}: {e}. Token decoding disabled.")
                    show_tokens = False
            else:
                print(f"Warning: Tokenizer config path '{tokenizer_config_path}' not found. Cannot decode tokens.")
                show_tokens = False
        else:
            print("Warning: Tokenizer config path not provided. Cannot decode tokens.")
            show_tokens = False


    for i in range(min(num_sequences_to_show, num_total_sequences)):
        print(f"\n--- Item {i+1} / {num_total_sequences} ---")
        sequence_data_item = data[i]
        
        print(f"  Type of item: {type(sequence_data_item)}")

        if isinstance(sequence_data_item, dict):
            print(f"  Keys in dictionary: {list(sequence_data_item.keys())}")
            
            song_name = sequence_data_item.get('song_name', 'N/A')
            print(f"    Song Name (if present): {song_name}")

            for key, value in sequence_data_item.items():
                if key == 'song_name': continue # Already printed

                print(f"    Key: '{key}'")
                print(f"      Value type: {type(value)}")
                if isinstance(value, torch.Tensor):
                    print(f"      Value shape: {value.shape}, dtype: {value.dtype}")
                    if value.numel() > 0: # Check if tensor is not empty
                         print(f"        Min: {value.min().item():.2f}, Max: {value.max().item():.2f}, Mean: {value.float().mean().item():.2f}")
                    else:
                        print("        Tensor is empty.")
                elif isinstance(value, list):
                    print(f"      Value is a list of length: {len(value)}")
                    if value:
                        print(f"        Type of first element in list: {type(value[0])}")
                elif isinstance(value, (str, int, float, bool)):
                     print(f"      Value: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                # Add more type checks if needed (e.g., nested dicts)

            # Specific checks for expected tokenization keys
            token_ids = sequence_data_item.get('token_ids')
            segment_ids = sequence_data_item.get('segment_ids')
            intra_line_pos_ids = sequence_data_item.get('intra_line_pos_ids')

            # Check token_ids
            if token_ids is not None:
                if isinstance(token_ids, torch.Tensor):
                    print(f"    Found 'token_ids': shape {token_ids.shape}, dtype {token_ids.dtype}, length {len(token_ids)}")
                    if show_tokens and tokenizer_instance and len(token_ids) > 0:
                        print(f"      First {min(len(token_ids), max_tokens_to_decode)} tokens (decoded):")
                        decoded_tokens_str_list = []
                        for k_idx in range(min(len(token_ids), max_tokens_to_decode)):
                            tok_id = token_ids[k_idx].item()
                            tok_str = tokenizer_instance.id_to_token.get(tok_id, f"[ID:{tok_id}]")
                            decoded_tokens_str_list.append(tok_str)
                        print(f"        {' '.join(decoded_tokens_str_list)}")
                else:
                    print(f"    Found 'token_ids', but it's not a Tensor. Type: {type(token_ids)}")
            else:
                print("    'token_ids': Not found as a key.")

            # Check segment_ids
            if segment_ids is not None:
                if isinstance(segment_ids, torch.Tensor):
                     max_seg_val_str = str(torch.max(segment_ids).item()) if len(segment_ids) > 0 else 'N/A'
                     print(f"    Found 'segment_ids': shape {segment_ids.shape}, dtype {segment_ids.dtype}, max_val {max_seg_val_str}")
                else:
                    print(f"    Found 'segment_ids', but it's not a Tensor. Type: {type(segment_ids)}")
            else:
                print("    'segment_ids': Not found as a key.")

            # Check intra_line_pos_ids
            if intra_line_pos_ids is not None:
                if isinstance(intra_line_pos_ids, torch.Tensor):
                    max_intra_val_str = str(torch.max(intra_line_pos_ids).item()) if len(intra_line_pos_ids) > 0 else 'N/A'
                    print(f"    Found 'intra_line_pos_ids': shape {intra_line_pos_ids.shape}, dtype {intra_line_pos_ids.dtype}, max_val {max_intra_val_str}")
                else:
                    print(f"    Found 'intra_line_pos_ids', but it's not a Tensor. Type: {type(intra_line_pos_ids)}")
            else:
                print("    'intra_line_pos_ids': Not found as a key.")
            
            # Length consistency check
            if all(x is not None and isinstance(x, torch.Tensor) for x in [token_ids, segment_ids, intra_line_pos_ids]):
                if not (len(token_ids) == len(segment_ids) == len(intra_line_pos_ids)):
                    print(f"      CRITICAL WARNING: Length mismatch! Tokens: {len(token_ids)}, Segments: {len(segment_ids)}, IntraPos: {len(intra_line_pos_ids)}")
        
        elif isinstance(sequence_data_item, torch.Tensor):
            print(f"  Item is a Tensor: shape {sequence_data_item.shape}, dtype {sequence_data_item.dtype}")
        else:
            print(f"  Item is of an unexpected type. Value (first 100 chars): {str(sequence_data_item)[:100]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect tokenized .pt data files (e.g., train_lite.pt, train_full.pt).")
    parser.add_argument("file_path", type=str, help="Path to the .pt file to inspect.")
    parser.add_argument("--tokenizer_config", type=str, default="beefai/flow_model/flow_tokenizer_config_v2.json",
                        help="Path to the FlowTokenizer config JSON file (for decoding tokens). Default: %(default)s")
    parser.add_argument("--num_seq", type=int, default=3, help="Number of items/sequences to display details for. Default: %(default)s")
    parser.add_argument("--show_tokens", action="store_true", help="Attempt to decode and show initial tokens if possible.")
    parser.add_argument("--max_decode", type=int, default=50, help="Max number of tokens to decode per sequence if --show_tokens is used. Default: %(default)s")

    args = parser.parse_args()

    inspect_pt_file(args.file_path, 
                    tokenizer_config_path=args.tokenizer_config, 
                    num_sequences_to_show=args.num_seq, 
                    show_tokens=args.show_tokens,
                    max_tokens_to_decode=args.max_decode)