import json
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import os

from beefai.utils.data_types import BarBeatFeatures, FlowDatum

class FlowTokenizer:
    def __init__(self, config_path: Optional[str] = None):
        self.max_syllables = 24 
        self.num_offset_bins = 16 
        self.num_duration_bins = 32 
        self.bpm_bins = [(0, 79), (80, 89), (90, 99), (100, 109), (110, 119), 
                         (120, 129), (130, 139), (140, 149), (150, 159), 
                         (160, 169), (170, 179), (180, 250)]
        self.max_subdivisions = 16 # For percussive events AND syllable start subdivisions
        self.flow_offset_max_beats = 4.0 
        self.flow_duration_max_beats = 8.0 

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        if config_path and os.path.exists(config_path):
            self.load_vocab(config_path)
            # Ensure all necessary tokens are present after loading, rebuild if extending
            initial_vocab_size = len(self.token_to_id)
            self._build_vocab(extend_existing=True) # Ensure new tokens are added if config is old
            if len(self.token_to_id) > initial_vocab_size:
                print(f"Tokenizer vocabulary extended with new tokens after loading. New size: {len(self.token_to_id)}")
                # Consider re-saving if vocab was extended
                # self.save_vocab(config_path) # Or prompt user
        else:
            if config_path: print(f"Tokenizer config {config_path} not found. Building vocab from defaults.")
            self._build_vocab()

    def _add_token(self, token: str, extend_existing: bool = False):
        if token not in self.token_to_id:
            token_id = len(self.token_to_id) # This might be problematic if extend_existing relies on specific ordering
            if extend_existing and self.id_to_token: # Find next available ID if extending
                 # This simple sequential add is fine as long as _build_vocab is called in a consistent order
                 pass # token_id will be correct if we only add new tokens at the end
            
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        # If token already exists and we are extending, do nothing.
        # If token already exists and not extending (fresh build), it's fine.

    def _build_vocab(self, extend_existing: bool = False):
        # If not extending, clear existing vocab to ensure clean build
        if not extend_existing:
            self.token_to_id = {}
            self.id_to_token = {}

        special_tokens = [
            "[PAD]", "[UNK]", "[BOS]", "[EOS]", 
            "[BAR_START]", "[LINE_START]",      
            "[SEP_INPUT_FLOW]",                 
            "[END_KICK_EVENTS]", "[END_SNARE_EVENTS]", "[END_HIHAT_EVENTS]", "[END_BASS_EVENTS]",
            "[NO_KICK_EVENTS]", "[NO_SNARE_EVENTS]", "[NO_HIHAT_EVENTS]", "[NO_BASS_EVENTS]",
            "[END_SYLLABLE_SUBDIVISIONS]" # New token for syllable landing patterns
        ]
        for token in special_tokens:
            self._add_token(token, extend_existing)

        for i, (low, high) in enumerate(self.bpm_bins):
            self._add_token(f"[BPM_{low}_{high}]", extend_existing)
        self._add_token("[BPM_UNKNOWN]", extend_existing) 

        self._add_token("[TIMESIG_4_4]", extend_existing)
        self._add_token("[TIMESIG_3_4]", extend_existing)
        self._add_token("[TIMESIG_OTHER]", extend_existing) 

        instruments = ["KICK", "SNARE", "HIHAT", "BASS"]
        for instrument_prefix in instruments:
            for i in range(self.max_subdivisions): 
                self._add_token(f"[{instrument_prefix}_AT_{i}]", extend_existing)

        for i in range(self.max_syllables + 1): 
            self._add_token(f"[SYLLABLES_{i}]", extend_existing)
        
        for i in range(self.num_offset_bins):
            self._add_token(f"[OFFSET_BIN_{i}]", extend_existing)
            
        for i in range(self.num_duration_bins):
            self._add_token(f"[DURATION_BIN_{i}]", extend_existing)
        
        # New tokens for syllable start subdivisions
        for i in range(self.max_subdivisions): # Assuming same number of subdivisions as percussive
            self._add_token(f"[SYLLABLE_STARTS_SUBDIV_{i}]", extend_existing)
        
        if "[UNK]" not in self.token_to_id : # Ensure UNK is always present
             self._add_token("[UNK]", extend_existing)


    def bpm_to_token(self, bpm: float) -> str:
        for low, high in self.bpm_bins:
            if low <= bpm <= high:
                return f"[BPM_{low}_{high}]"
        return "[BPM_UNKNOWN]"
        
    def time_signature_to_token(self, ts: Tuple[int, int]) -> str:
        if ts == (4, 4): return "[TIMESIG_4_4]"
        if ts == (3, 4): return "[TIMESIG_3_4]"
        return "[TIMESIG_OTHER]"

    def encode_bar_features(self, bar_features: BarBeatFeatures) -> List[int]:
        tokens_str: List[str] = ["[BAR_START]"]
        
        tokens_str.append(self.bpm_to_token(bar_features["bpm"]))
        tokens_str.append(self.time_signature_to_token(bar_features["time_signature"]))
        
        instrument_event_map = {
            "KICK": "kick_events", "SNARE": "snare_events",
            "HIHAT": "hihat_events", "BASS": "bass_events"
        }
        for instr_prefix, event_key in instrument_event_map.items():
            events = bar_features.get(event_key, [])
            if events:
                for subdivision_idx in events:
                    if 0 <= subdivision_idx < self.max_subdivisions:
                        tokens_str.append(f"[{instr_prefix}_AT_{subdivision_idx}]")
                    else:
                        # This case should ideally be handled by BeatFeatureExtractor's quantization.
                        # If it still occurs, it's an upstream data issue.
                        print(f"Warning: Subdivision index {subdivision_idx} for {instr_prefix} in bar {bar_features.get('bar_index','N/A')} is out of range [0, {self.max_subdivisions-1}]. Skipping token.")
            else: 
                tokens_str.append(f"[NO_{instr_prefix}_EVENTS]")
            tokens_str.append(f"[END_{instr_prefix}_EVENTS]") 
            
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens_str]

    def encode_flow_datum(self, flow_datum: FlowDatum) -> List[int]:
        tokens_str: List[str] = ["[LINE_START]"]
        
        syllables = min(max(0, flow_datum["syllables"]), self.max_syllables)
        tokens_str.append(f"[SYLLABLES_{syllables}]")
        
        offset_normalized = flow_datum["start_offset_beats"] / self.flow_offset_max_beats
        offset_bin_idx = int(np.clip(offset_normalized * self.num_offset_bins, 0, self.num_offset_bins - 1))
        tokens_str.append(f"[OFFSET_BIN_{offset_bin_idx}]")
        
        duration_normalized = flow_datum["duration_beats"] / self.flow_duration_max_beats
        duration_bin_idx = int(np.clip(duration_normalized * self.num_duration_bins, 0, self.num_duration_bins - 1))
        tokens_str.append(f"[DURATION_BIN_{duration_bin_idx}]")

        # Encode syllable start subdivisions
        syllable_subdivisions = flow_datum.get("syllable_start_subdivisions", [])
        for sub_idx in syllable_subdivisions:
            if 0 <= sub_idx < self.max_subdivisions:
                tokens_str.append(f"[SYLLABLE_STARTS_SUBDIV_{sub_idx}]")
            else:
                # This case should ideally be handled by FlowDataExtractor clipping, but as a safeguard:
                print(f"Warning: Syllable subdivision index {sub_idx} in bar {flow_datum.get('bar_index','N/A')}, line {flow_datum.get('line_index_in_bar','N/A')} out of range [0, {self.max_subdivisions-1}]. Skipping token.")
        tokens_str.append("[END_SYLLABLE_SUBDIVISIONS]") # Crucial delimiter
        
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens_str]

    def encode_song_instance(self, 
                             song_beat_features: List[BarBeatFeatures], 
                             song_flow_data: List[FlowDatum]
                            ) -> Tuple[List[int], List[int], List[int]]:
        full_token_ids: List[int] = [self.bos_token_id]
        segment_ids: List[int] = [0] 
        intra_line_pos_ids: List[int] = [0] 

        current_segment_idx_for_bar_features = 0 # Starts at 0 for BOS, then for first bar's features
        
        flow_by_bar: Dict[int, List[FlowDatum]] = {}
        for fd in song_flow_data:
            bar_idx = fd["bar_index"]
            if bar_idx not in flow_by_bar: flow_by_bar[bar_idx] = []
            flow_by_bar[bar_idx].append(fd)

        for bar_feature_idx, bar_features in enumerate(song_beat_features):
            bar_idx = bar_features["bar_index"] # Actual bar index from data
            
            # Segment for Beat Features for this bar
            # Segment ID calculation: current_bar_feature_block_index * 2
            # e.g. bar 0 features: seg_id 0. bar 1 features: seg_id 2.
            seg_id_for_this_bar_features = bar_feature_idx * 2
            
            bar_feature_tokens = self.encode_bar_features(bar_features)
            full_token_ids.extend(bar_feature_tokens)
            segment_ids.extend([seg_id_for_this_bar_features] * len(bar_feature_tokens))
            intra_line_pos_ids.extend(list(range(len(bar_feature_tokens)))) 
            
            # Segment for SEP token and subsequent Flow Lines for this bar
            # e.g. bar 0 flow: seg_id 1. bar 1 flow: seg_id 3.
            seg_id_for_this_bar_flow = seg_id_for_this_bar_features + 1
            
            full_token_ids.append(self.sep_input_flow_token_id)
            segment_ids.append(seg_id_for_this_bar_flow) 
            intra_line_pos_ids.append(0) # SEP is pos 0 of this new flow segment

            bar_flow_lines = flow_by_bar.get(bar_idx, [])
            for line_idx, flow_datum in enumerate(bar_flow_lines):
                flow_line_tokens = self.encode_flow_datum(flow_datum)
                full_token_ids.extend(flow_line_tokens)
                segment_ids.extend([seg_id_for_this_bar_flow] * len(flow_line_tokens))
                intra_line_pos_ids.extend(list(range(len(flow_line_tokens))))

        # After all bars and their flows, add EOS
        # EOS segment_id will be (num_beat_feature_blocks * 2)
        # e.g., if 2 bars processed (bar_feature_idx 0 and 1), num_beat_feature_blocks = 2.
        # Last flow segment was for bar 1, seg_id_for_this_bar_flow = 1*2+1 = 3.
        # EOS segment is (2 * 2) = 4. (Or just last segment_id + 1, but need to be careful if no flow lines for last bar)
        # A safer way: if segment_ids is not empty, take the last one and add 1, or handle BOS case.
        eos_segment_id = segment_ids[-1] + 1 if segment_ids else 0 # if only BOS, then seg 0. Else, increment.
        if full_token_ids == [self.bos_token_id]: # Only BOS so far (empty song)
            eos_segment_id = 0 # EOS shares segment with BOS if song is empty.
        
        full_token_ids.append(self.eos_token_id)
        segment_ids.append(eos_segment_id) 
        intra_line_pos_ids.append(0) 
        
        return full_token_ids, segment_ids, intra_line_pos_ids


    def decode_flow_tokens_to_datum(self, flow_tokens: List[int], bar_idx_context: int, line_idx_context: int) -> Optional[FlowDatum]:
        tokens_str = [self.id_to_token.get(t_id, "[UNK]") for t_id in flow_tokens]
        
        ptr = 0
        if not tokens_str or tokens_str[ptr] != "[LINE_START]":
            # print(f"Debug: Decode FlowDatum failed. Expected [LINE_START]. Got: {' '.join(tokens_str[:5])}...")
            return None 
        ptr += 1
        
        # Expected: [SYLLABLES_X], [OFFSET_BIN_Y], [DURATION_BIN_Z], then SYLLABLE_STARTS_SUBDIV*, END_SYLLABLE_SUBDIVISIONS
        if len(tokens_str) - ptr < 3: # Need at least SYL, OFF, DUR tokens
            # print(f"Debug: Decode FlowDatum failed. Too few tokens after [LINE_START]. Got: {' '.join(tokens_str)}")
            return None

        try:
            syll_token = tokens_str[ptr]; ptr +=1
            offset_token = tokens_str[ptr]; ptr +=1
            duration_token = tokens_str[ptr]; ptr +=1

            if not (syll_token.startswith("[SYLLABLES_") and \
                    offset_token.startswith("[OFFSET_BIN_") and \
                    duration_token.startswith("[DURATION_BIN_")):
                # print(f"Debug: Decode FlowDatum failed. Incorrect token types for SYL/OFF/DUR. Got: {syll_token}, {offset_token}, {duration_token}")
                return None

            syllables = int(syll_token.split('_')[-1].replace(']', ''))
            
            offset_bin_idx = int(offset_token.split('_')[-1].replace(']', ''))
            start_offset_beats = (offset_bin_idx / self.num_offset_bins) * self.flow_offset_max_beats
            
            duration_bin_idx = int(duration_token.split('_')[-1].replace(']', ''))
            duration_beats = (duration_bin_idx / self.num_duration_bins) * self.flow_duration_max_beats

            syllable_start_subdivisions: List[int] = []
            # Loop to gather [SYLLABLE_STARTS_SUBDIV_X] tokens
            while ptr < len(tokens_str) and tokens_str[ptr].startswith("[SYLLABLE_STARTS_SUBDIV_"):
                subdiv_token = tokens_str[ptr]
                sub_idx = int(subdiv_token.split('_')[-1].replace(']', ''))
                syllable_start_subdivisions.append(sub_idx)
                ptr += 1
            
            # After subdivision tokens, we expect [END_SYLLABLE_SUBDIVISIONS]
            if ptr < len(tokens_str) and tokens_str[ptr] == "[END_SYLLABLE_SUBDIVISIONS]":
                ptr += 1 # Consume it
            else:
                # print(f"Debug: Decode FlowDatum failed. Missing or misplaced [END_SYLLABLE_SUBDIVISIONS]. Last token before check: {tokens_str[ptr-1] if ptr > 0 else 'N/A'}. Current token: {tokens_str[ptr] if ptr < len(tokens_str) else 'EOS'}")
                return None # Strict: must end with this token if subdivisions were present or if it's just an empty list of them.
                            # Or relax this if an empty subdivision list doesn't always write END_SYLLABLE_SUBDIVISIONS (but it should)

            # Ensure no unexpected tokens remain if ptr hasn't reached end of flow_tokens
            if ptr != len(flow_tokens):
                # print(f"Debug: Decode FlowDatum warning. Unexpected trailing tokens: {' '.join(tokens_str[ptr:])}")
                # Depending on strictness, you might return None here or allow it.
                # For now, let's be strict and assume the decoded part up to END_SYLLABLE_SUBDIVISIONS is one full FlowDatum.
                pass # Or return None if strictness is desired

            return {
                "bar_index": bar_idx_context, 
                "line_index_in_bar": line_idx_context,
                "syllables": syllables,
                "start_offset_beats": round(start_offset_beats, 3),
                "duration_beats": round(duration_beats, 3),
                "syllable_start_subdivisions": syllable_start_subdivisions
            }
        except (ValueError, IndexError, AttributeError) as e:
            # print(f"Debug: Decode FlowDatum exception: {e}. Tokens: {' '.join(tokens_str)}")
            return None

    def get_vocab_size(self) -> int:
        return len(self.token_to_id)

    def save_vocab(self, filepath: str):
        # Ensure id_to_token keys are strings for JSON serialization
        id_to_token_serializable = {str(k): v for k, v in self.id_to_token.items()}

        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": id_to_token_serializable, # Use serializable version
            "config_params": {
                "max_syllables": self.max_syllables,
                "num_offset_bins": self.num_offset_bins,
                "num_duration_bins": self.num_duration_bins,
                "bpm_bins": self.bpm_bins,
                "max_subdivisions": self.max_subdivisions,
                "flow_offset_max_beats": self.flow_offset_max_beats,
                "flow_duration_max_beats": self.flow_duration_max_beats
            }
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        print(f"Vocabulary and config saved to {filepath}")

    def load_vocab(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data["token_to_id"]
        # Convert string keys from JSON back to int for id_to_token
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()} 
        
        config_params = vocab_data.get("config_params", {})
        self.max_syllables = config_params.get("max_syllables", self.max_syllables)
        self.num_offset_bins = config_params.get("num_offset_bins", self.num_offset_bins)
        self.num_duration_bins = config_params.get("num_duration_bins", self.num_duration_bins)
        self.bpm_bins = config_params.get("bpm_bins", self.bpm_bins)
        self.max_subdivisions = config_params.get("max_subdivisions", self.max_subdivisions)
        self.flow_offset_max_beats = config_params.get("flow_offset_max_beats", self.flow_offset_max_beats)
        self.flow_duration_max_beats = config_params.get("flow_duration_max_beats", self.flow_duration_max_beats)
        
        # Note: _build_vocab(extend_existing=True) is called after load in __init__
        # to add any new tokens not present in the loaded config.
        print(f"Vocabulary and config loaded from {filepath}. Initial loaded size: {len(self.token_to_id)}")


    @property
    def pad_token_id(self) -> int: return self.token_to_id["[PAD]"]
    @property
    def bos_token_id(self) -> int: return self.token_to_id["[BOS]"]
    @property
    def eos_token_id(self) -> int: return self.token_to_id["[EOS]"]
    @property
    def unk_token_id(self) -> int: return self.token_to_id.get("[UNK]") # Use .get for safety if UNK somehow missing
    @property
    def sep_input_flow_token_id(self) -> int: return self.token_to_id["[SEP_INPUT_FLOW]"]
    @property
    def bar_start_token_id(self) -> int: return self.token_to_id["[BAR_START]"]
    @property
    def line_start_token_id(self) -> int: return self.token_to_id["[LINE_START]"]
    @property
    def end_syllable_subdivisions_token_id(self) -> int: return self.token_to_id["[END_SYLLABLE_SUBDIVISIONS]"]


if __name__ == '__main__':
    # Test with the v2 config file name, but content will be updated if it's old.
    tokenizer_path = "beefai/flow_model/flow_tokenizer_config_v2.json" 
    print(f"Attempting to load/create tokenizer config at: {tokenizer_path}")
    
    # Create dummy dir if it doesn't exist for saving
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

    tokenizer = FlowTokenizer(config_path=tokenizer_path) 
    # Save it back, potentially with new tokens if the loaded one was old
    tokenizer.save_vocab(tokenizer_path) 
        
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")
    assert "[SYLLABLE_STARTS_SUBDIV_0]" in tokenizer.token_to_id
    assert "[END_SYLLABLE_SUBDIVISIONS]" in tokenizer.token_to_id
    print("Essential syllable subdivision tokens are present.")


    sample_bar_features_1: BarBeatFeatures = {
        "bar_index": 0, "bpm": 125.0, "time_signature": (4, 4),
        "kick_events": [0, 4, 8, 12], "snare_events": [4, 12],
        "hihat_events": [i for i in range(0,16,2)], "bass_events": []
    }
    sample_bar_features_2: BarBeatFeatures = {
        "bar_index": 1, "bpm": 95.0, "time_signature": (4, 4),
        "kick_events": [0, 8], "snare_events": [4], "hihat_events": [], "bass_events": [0,2,4,6]
    }
    
    sample_flow_datum_b0_l0: FlowDatum = { 
        "bar_index": 0, "line_index_in_bar": 0, "syllables": 3, 
        "start_offset_beats": 0.0, "duration_beats": 1.9,
        "syllable_start_subdivisions": [4, 8, 12] 
    }
    sample_flow_datum_b0_l1: FlowDatum = { 
        "bar_index": 0, "line_index_in_bar": 1, "syllables": 2, 
        "start_offset_beats": 2.0, "duration_beats": 1.5,
        "syllable_start_subdivisions": [0, 7] 
    }
    sample_flow_datum_b1_l0: FlowDatum = { 
        "bar_index": 1, "line_index_in_bar": 0, "syllables": 4,
        "start_offset_beats": 0.5, "duration_beats": 3.0,
        "syllable_start_subdivisions": [2, 6, 10, 14]
    }
    
    song_beats = [sample_bar_features_1, sample_bar_features_2]
    song_flows = [sample_flow_datum_b0_l0, sample_flow_datum_b0_l1, sample_flow_datum_b1_l0]

    token_ids, seg_ids, intra_pos_ids = tokenizer.encode_song_instance(song_beats, song_flows)
    
    print(f"\nEncoded song instance (length {len(token_ids)}):")
    
    print("\nTokenized sequence:")
    for i in range(len(token_ids)):
        tok_str = tokenizer.id_to_token.get(token_ids[i], "ERR_TOK")
        seg_str = f"S:{seg_ids[i]}"
        pos_str = f"P:{intra_pos_ids[i]}"
        print(f"{tok_str:<30} {seg_str:<5} {pos_str:<5}")
        if tok_str == "[SEP_INPUT_FLOW]": print("-" * 40)
        if tok_str == "[EOS]": break

    example_flow_line_tokens = tokenizer.encode_flow_datum(sample_flow_datum_b0_l0)
    print(f"\nExample encoded flow line tokens for datum b0_l0: {example_flow_line_tokens}")
    print(" ".join([tokenizer.id_to_token.get(tid, "UNK") for tid in example_flow_line_tokens]))
    decoded_fd = tokenizer.decode_flow_tokens_to_datum(example_flow_line_tokens, bar_idx_context=0, line_idx_context=0)
    print(f"Decoded FlowDatum for b0_l0: {decoded_fd}")
    assert decoded_fd is not None
    assert decoded_fd["syllable_start_subdivisions"] == [4,8,12]
    
    sample_flow_datum_empty_subdiv: FlowDatum = { 
        "bar_index": 0, "line_index_in_bar": 2, "syllables": 1,
        "start_offset_beats": 3.0, "duration_beats": 0.5,
        "syllable_start_subdivisions": [] 
    }
    encoded_empty_subdiv = tokenizer.encode_flow_datum(sample_flow_datum_empty_subdiv)
    print(f"\nEncoded flow line with empty subdivisions: {encoded_empty_subdiv}")
    print(" ".join([tokenizer.id_to_token.get(tid, "UNK") for tid in encoded_empty_subdiv]))
    decoded_empty_subdiv = tokenizer.decode_flow_tokens_to_datum(encoded_empty_subdiv, 0, 2)
    print(f"Decoded FlowDatum (empty subdivisions): {decoded_empty_subdiv}")
    assert decoded_empty_subdiv is not None
    assert decoded_empty_subdiv["syllable_start_subdivisions"] == []
    
    # Test decoding of a slightly malformed sequence (e.g. no END_SYLLABLE_SUBDIVISIONS)
    malformed_tokens = example_flow_line_tokens[:-1] # Remove END_SYLLABLE_SUBDIVISIONS
    decoded_malformed = tokenizer.decode_flow_tokens_to_datum(malformed_tokens, 0,0)
    print(f"Decoded malformed (missing END_SYLLABLE_SUBDIVISIONS): {decoded_malformed}")
    assert decoded_malformed is None # Expecting None due to strict parsing

    print("\nTokenizer tests passed.")