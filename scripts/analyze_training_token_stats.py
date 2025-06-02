import torch
    import os
    import sys
    from collections import Counter
    import yaml
    from tqdm import tqdm
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from beefai.flow_model.tokenizer import FlowTokenizer
    
    def load_yaml_config(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def analyze_token_stats(data_config_path: str, model_config_path: str, limit_songs: Optional[int] = None):
        print(f"Loading data config from: {data_config_path}")
        data_config = load_yaml_config(data_config_path)
        print(f"Loading model config from: {model_config_path}") # Not strictly needed for this script but good practice
        # model_config = load_yaml_config(model_config_path)
    
        tokenizer_path = data_config.get("tokenizer_path")
        tokenizer = FlowTokenizer(config_path=tokenizer_path)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
    
        train_data_file_path = data_config.get("train_data_path")
        if not os.path.exists(train_data_file_path):
            print(f"ERROR: Training data file not found at {train_data_file_path}")
            return
    
        print(f"Loading tokenized training data from: {train_data_file_path}...")
        # Each item in tokenized_song_sequences is a dict:
        # {'token_ids': tensor, 'segment_ids': tensor, 'intra_line_pos_ids': tensor, 'song_name': str}
        tokenized_song_sequences = torch.load(train_data_file_path, weights_only=False)
        if limit_songs and limit_songs > 0:
            print(f"Limiting analysis to first {limit_songs} songs.")
            tokenized_song_sequences = tokenized_song_sequences[:limit_songs]
        
        print(f"Analyzing {len(tokenized_song_sequences)} tokenized song sequences...")
    
        # Overall counts for per-syllable tokens
        overall_syllable_starts_counts = Counter()
        overall_syllable_duration_counts = Counter()
        overall_syllable_stress_counts = Counter()
        overall_syllables_token_counts = Counter()
    
        # Per-syllable token counts conditioned on [SYLLABLES_X]
        conditional_syl_starts = {i: Counter() for i in range(tokenizer.max_syllables + 1)}
        conditional_syl_durations = {i: Counter() for i in range(tokenizer.max_syllables + 1)}
    
        for song_data in tqdm(tokenized_song_sequences, desc="Processing songs"):
            tokens = song_data['token_ids'].tolist()
            
            ptr = 0
            while ptr < len(tokens):
                if tokens[ptr] == tokenizer.line_start_token_id:
                    line_start_ptr = ptr
                    ptr += 1
                    if ptr >= len(tokens): break
    
                    syllables_token_id = tokens[ptr]
                    syllables_token_str = tokenizer.id_to_token.get(syllables_token_id, "")
                    if not syllables_token_str.startswith("[SYLLABLES_"):
                        # Not a valid flow line sequence start, advance to next LINE_START or end
                        while ptr < len(tokens) and tokens[ptr] != tokenizer.line_start_token_id:
                            ptr += 1
                        continue 
                    
                    overall_syllables_token_counts[syllables_token_str] += 1
                    current_syl_count_for_line = int(syllables_token_str.split('_')[-1].replace(']', ''))
                    ptr += 1 # Consumed [SYLLABLES_X]
    
                    # Skip [OFFSET_BIN_X] and [DURATION_BIN_X]
                    if ptr + 1 < len(tokens): # Need at least two more for offset and duration
                        ptr += 2 
                    else: # Incomplete line header
                        while ptr < len(tokens) and tokens[ptr] != tokenizer.line_start_token_id: ptr+=1
                        continue
    
                    num_syl_events_in_this_line = 0
                    while ptr < len(tokens) and tokens[ptr] != tokenizer.end_syllable_sequence_token_id \
                          and tokens[ptr] != tokenizer.line_start_token_id \
                          and tokens[ptr] != tokenizer.bar_start_token_id \
                          and tokens[ptr] != tokenizer.eos_token_id:
                        
                        if ptr + 2 >= len(tokens): break # Need a full triplet
    
                        syl_start_tok_id = tokens[ptr]
                        syl_dur_tok_id = tokens[ptr+1]
                        syl_stress_tok_id = tokens[ptr+2]
    
                        syl_start_str = tokenizer.id_to_token.get(syl_start_tok_id, "")
                        syl_dur_str = tokenizer.id_to_token.get(syl_dur_tok_id, "")
                        syl_stress_str = tokenizer.id_to_token.get(syl_stress_tok_id, "")
    
                        if syl_start_str.startswith("[SYLLABLE_STARTS_SUBDIV_") and \
                           syl_dur_str.startswith("[SYLLABLE_DURATION_BIN_") and \
                           syl_stress_str.startswith("[SYLLABLE_STRESS_"):
                            
                            overall_syllable_starts_counts[syl_start_str] += 1
                            overall_syllable_duration_counts[syl_dur_str] += 1
                            overall_syllable_stress_counts[syl_stress_str] += 1
    
                            conditional_syl_starts[current_syl_count_for_line][syl_start_str] += 1
                            conditional_syl_durations[current_syl_count_for_line][syl_dur_str] += 1
                            
                            num_syl_events_in_this_line +=1
                            ptr += 3 # Consumed triplet
                        else: # Malformed triplet or unexpected token
                            ptr +=1 # Skip and hope to resync
                            break # Break from per-syllable parsing for this line
                    
                    if ptr < len(tokens) and tokens[ptr] == tokenizer.end_syllable_sequence_token_id:
                        ptr += 1 # Consumed [END_SYLLABLE_SEQUENCE]
                else:
                    ptr += 1
        
        print("\n--- Overall Token Counts ---")
        print("SYLLABLES_X counts:", sorted(overall_syllables_token_counts.items(), key=lambda x: int(x[0].split('_')[-1].replace(']',''))))
        print("\nSYLLABLE_STARTS_SUBDIV_X counts:", sorted(overall_syllable_starts_counts.items(), key=lambda x: int(x[0].split('_')[-1].replace(']','')) if x[0].split('_')[-1].replace(']','').isdigit() else -1))
        print("\nSYLLABLE_DURATION_BIN_X counts:", sorted(overall_syllable_duration_counts.items(), key=lambda x: int(x[0].split('_')[-1].replace(']','')) if x[0].split('_')[-1].replace(']','').isdigit() else -1))
        print("\nSYLLABLE_STRESS_X counts:", sorted(overall_syllable_stress_counts.items(), key=lambda x: int(x[0].split('_')[-1].replace(']','')) if x[0].split('_')[-1].replace(']','').isdigit() else -1))
    
        print("\n--- Conditional Per-Syllable Token Counts (Sample for High Syllable Counts) ---")
        for syl_count_key in sorted(conditional_syl_starts.keys(), reverse=True):
            if syl_count_key >= 16: # Analyze longer lines
                total_events_for_this_syl_count = sum(conditional_syl_starts[syl_count_key].values())
                if total_events_for_this_syl_count == 0: continue

                print(f"\nFor lines with [SYLLABLES_{syl_count_key}] (Total {overall_syllables_token_counts.get(f'[SYLLABLES_{syl_count_key}]',0)} such lines):")
                
                print(f"  Start Subdivision Dist (Top 5 for SYLLABLES_{syl_count_key}):")
                for token, count in sorted(conditional_syl_starts[syl_count_key].items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {token}: {count} ({(count/total_events_for_this_syl_count)*100:.1f}%)")
                
                print(f"  Duration Bin Dist (Top 5 for SYLLABLES_{syl_count_key}):")
                for token, count in sorted(conditional_syl_durations[syl_count_key].items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {token}: {count} ({(count/total_events_for_this_syl_count)*100:.1f}%)")
    
    if __name__ == "__main__":
        # For "full" model data
        analyze_token_stats(
            data_config_path="lite_model_training/data_config_full.yaml",
            model_config_path="lite_model_training/model_config_full.yaml",
            limit_songs=None # Analyze all songs in train_full.pt, or set a number for faster testing
        )
        # For "lite" model data (if you want to compare)
        # analyze_token_stats(
        #     data_config_path="lite_model_training/data_config_lite.yaml",
        #     model_config_path="lite_model_training/model_config_lite.yaml",
        #     limit_songs=10
        # )