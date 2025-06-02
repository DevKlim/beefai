import torch
import os
import argparse
import sys

# --- BEGIN TEMPORARY PATH FIX (for direct script execution) ---
# Get the absolute path of the project root directory
# This assumes this script is in the 'scripts/' directory,
# and the 'beefai' package is one level up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- END TEMPORARY PATH FIX ---

try:
    from beefai.flow_model.tokenizer import FlowTokenizer
except ImportError:
    FlowTokenizer = None
    print("Warning: FlowTokenizer could not be imported. Decoding will not be available.")
    print("Ensure 'beefai' is in PYTHONPATH or run from project root if using a package structure.")

def inspect_data(data_file_path, tokenizer_config_path=None, num_items_to_inspect=5, decode_tokens=False, max_tokens_to_decode=100):
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        return

    print(f"--- Inspecting: {data_file_path} ---")
    try:
        # Load data, ensuring it works whether it's a list of dicts or a single dict
        loaded_data = torch.load(data_file_path, weights_only=False) # Important for list of dicts
        
        if isinstance(loaded_data, dict): # Handle if it's a single dict (older format?)
            data_list = [loaded_data]
        elif isinstance(loaded_data, list):
            data_list = loaded_data
        else:
            print(f"Error: Loaded data is not a list or a dictionary. Type: {type(loaded_data)}")
            return
            
        if not data_list:
            print("The loaded data list is empty.")
            return

        print(f"Total items (song sequences/elements) in the list: {len(data_list)}\n")

        tokenizer_instance = None
        if decode_tokens:
            if FlowTokenizer is None:
                print("Cannot decode tokens: FlowTokenizer class not available.")
                decode_tokens = False
            elif tokenizer_config_path and os.path.exists(tokenizer_config_path):
                tokenizer_instance = FlowTokenizer(config_path=tokenizer_config_path)
                print(f"Tokenizer loaded from {tokenizer_config_path} for decoding. Vocab size: {tokenizer_instance.get_vocab_size()}")
            else:
                print(f"Warning: Tokenizer config path '{tokenizer_config_path}' not provided or not found. Cannot decode tokens.")
                decode_tokens = False
        
        items_to_show = min(num_items_to_inspect, len(data_list))

        for i in range(items_to_show):
            item = data_list[i]
            print(f"--- Item {i+1} / {len(data_list)} ---")
            if not isinstance(item, dict):
                print(f"  Item is not a dictionary. Type: {type(item)}")
                continue

            print(f"  Type of item: {type(item)}")
            print(f"  Keys in dictionary: {list(item.keys())}")

            song_name = item.get('song_name', 'N/A')
            print(f"    Song Name (if present): {song_name}")

            for key, value in item.items():
                if key == 'song_name': continue # Already printed

                print(f"    Key: '{key}'")
                if isinstance(value, torch.Tensor):
                    print(f"      Value type: {type(value)}")
                    print(f"      Value shape: {value.shape}, dtype: {value.dtype}")
                    if value.numel() > 0: # Check if tensor is not empty
                        try:
                            min_val = value.min().item()
                            max_val = value.max().item()
                            mean_val = value.float().mean().item()
                            print(f"        Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}")
                        except RuntimeError as e:
                            print(f"        Could not compute stats (min/max/mean): {e}")
                    else:
                        print("        Tensor is empty.")
                else:
                    print(f"      Value type: {type(value)}, Value (first 100 chars): {str(value)[:100]}")
            
            # Sanity checks
            token_ids = item.get('token_ids')
            segment_ids = item.get('segment_ids')
            intra_line_pos_ids = item.get('intra_line_pos_ids')

            if isinstance(token_ids, torch.Tensor):
                print(f"    Found 'token_ids': shape {token_ids.shape}, dtype {token_ids.dtype}, length {len(token_ids)}")
                if decode_tokens and tokenizer_instance:
                    print(f"      Decoded first {max_tokens_to_decode} tokens:")
                    decoded_sequence = [tokenizer_instance.id_to_token.get(tok_id.item(), f"[UNK_ID:{tok_id.item()}]") for tok_id in token_ids[:max_tokens_to_decode]]
                    
                    # Print with structure highlighting [SEP_INPUT_FLOW] and [LINE_START]
                    current_line_str = "        "
                    for token_str in decoded_sequence:
                        current_line_str += token_str + " "
                        if token_str == tokenizer_instance.id_to_token.get(tokenizer_instance.sep_input_flow_token_id) or \
                           token_str == tokenizer_instance.id_to_token.get(tokenizer_instance.line_start_token_id) or \
                           token_str == tokenizer_instance.id_to_token.get(tokenizer_instance.end_syllable_sequence_token_id) or \
                           token_str == tokenizer_instance.id_to_token.get(tokenizer_instance.bar_start_token_id):
                            print(current_line_str.rstrip())
                            current_line_str = "        "
                        elif len(current_line_str) > 120: # Max line length for console
                            print(current_line_str.rstrip())
                            current_line_str = "        "
                    if current_line_str.strip(): # Print any remaining part
                        print(current_line_str.rstrip())
                    print("      ---")


            if isinstance(segment_ids, torch.Tensor) and segment_ids.numel() > 0:
                print(f"    Found 'segment_ids': shape {segment_ids.shape}, dtype {segment_ids.dtype}, max_val {segment_ids.max().item()}")
            if isinstance(intra_line_pos_ids, torch.Tensor) and intra_line_pos_ids.numel() > 0:
                print(f"    Found 'intra_line_pos_ids': shape {intra_line_pos_ids.shape}, dtype {intra_line_pos_ids.dtype}, max_val {intra_line_pos_ids.max().item()}")

            # Length consistency check
            if isinstance(token_ids, torch.Tensor) and isinstance(segment_ids, torch.Tensor) and isinstance(intra_line_pos_ids, torch.Tensor):
                if not (len(token_ids) == len(segment_ids) == len(intra_line_pos_ids)):
                    print(f"    WARNING: Length mismatch! Tokens: {len(token_ids)}, Segments: {len(segment_ids)}, IntraPos: {len(intra_line_pos_ids)}")
            print("")


    except FileNotFoundError:
        print(f"Error: Data file not found: {data_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect tokenized data .pt files.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the .pt data file to inspect.")
    parser.add_argument("--tokenizer_config", type=str, default=None, help="Optional path to the FlowTokenizer config JSON for decoding tokens.")
    parser.add_argument("--num_items_to_inspect", type=int, default=3, help="Number of items (songs/sequences) from the list to inspect.")
    parser.add_argument("--decode_tokens", action='store_true', help="Flag to decode and print token sequences if tokenizer_config is provided.")
    parser.add_argument("--max_tokens_to_decode", type=int, default=150, help="Max number of tokens to decode and print per item.")
    
    args = parser.parse_args()
    
    inspect_data(args.data_file, args.tokenizer_config, args.num_items_to_inspect, args.decode_tokens, args.max_tokens_to_decode)