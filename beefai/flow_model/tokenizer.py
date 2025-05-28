import json
from typing import List, Dict, Tuple, Any, Optional, Union

from beefai.utils.data_types import BarBeatFeatures, FlowDatum

class FlowTokenizer:
    def __init__(self, config_path: Optional[str] = None):
        # Define vocabulary limits and quantization parameters
        # These should ideally come from a config file or be passed during init
        self.max_syllables = 24  # Tokens: [SYLLABLES_0] to [SYLLABLES_24]
        self.num_offset_bins = 16 # For start_offset_beats, e.g., 0 to 3.75 beats in 0.25 beat steps for a 4-beat bar
        self.num_duration_bins = 32 # For duration_beats, e.g., 0.25 to 8.0 beats in 0.25 beat steps
        
        # BPM quantization: (lower_bpm, upper_bpm)
        self.bpm_bins = [(0, 79), (80, 89), (90, 99), (100, 109), (110, 119), 
                         (120, 129), (130, 139), (140, 149), (150, 159), 
                         (160, 169), (170, 179), (180, 250)]
        self.max_subdivisions = 16 # For percussive events (0-15)

        # Quantization parameters for flow (start_offset_beats, duration_beats)
        self.flow_offset_max_beats = 4.0 # Max offset considered (e.g., for a 4/4 bar)
        self.flow_duration_max_beats = 8.0 # Max duration considered (e.g., 2 bars of 4/4)


        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        if config_path and os.path.exists(config_path):
            self.load_vocab(config_path)
        else:
            if config_path: print(f"Tokenizer config {config_path} not found. Building vocab from defaults.")
            self._build_vocab()

    def _add_token(self, token: str):
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def _build_vocab(self):
        # Special tokens
        special_tokens = [
            "[PAD]", "[UNK]", "[BOS]", "[EOS]", # Basic
            "[BAR_START]", "[LINE_START]",      # Structural markers
            "[SEP_INPUT_FLOW]",                 # Separator between beat input and flow target
            # Markers for end of percussive event sequences for a given instrument
            "[END_KICK_EVENTS]", "[END_SNARE_EVENTS]", "[END_HIHAT_EVENTS]", "[END_BASS_EVENTS]",
            # Markers for absence of events for an instrument in a bar
            "[NO_KICK_EVENTS]", "[NO_SNARE_EVENTS]", "[NO_HIHAT_EVENTS]", "[NO_BASS_EVENTS]"
        ]
        for token in special_tokens:
            self._add_token(token)

        # BPM tokens from bins
        for i, (low, high) in enumerate(self.bpm_bins):
            self._add_token(f"[BPM_{low}_{high}]")
        self._add_token("[BPM_UNKNOWN]") # Fallback for out-of-range BPM

        # Time signature tokens (can be extended)
        self._add_token("[TIMESIG_4_4]")
        self._add_token("[TIMESIG_3_4]")
        self._add_token("[TIMESIG_OTHER]") # For less common time signatures

        # Percussive event tokens: [INSTRUMENT_AT_SUBDIVISION_X]
        instruments = ["KICK", "SNARE", "HIHAT", "BASS"]
        for instrument_prefix in instruments:
            for i in range(self.max_subdivisions): # 0 to 15 for 16 subdivisions
                self._add_token(f"[{instrument_prefix}_AT_{i}]")

        # Flow target tokens
        # Syllables: [SYLLABLES_0] to [SYLLABLES_MAX]
        for i in range(self.max_syllables + 1): 
            self._add_token(f"[SYLLABLES_{i}]")
        
        # Start Offset Bins: [OFFSET_BIN_0] to [OFFSET_BIN_N-1]
        for i in range(self.num_offset_bins):
            self._add_token(f"[OFFSET_BIN_{i}]")
            
        # Duration Bins: [DURATION_BIN_0] to [DURATION_BIN_M-1]
        for i in range(self.num_duration_bins):
            self._add_token(f"[DURATION_BIN_{i}]")
        
        if "[UNK]" not in self.token_to_id: # Ensure UNK is present
             self._add_token("[UNK]")


    def bpm_to_token(self, bpm: float) -> str:
        for low, high in self.bpm_bins:
            if low <= bpm <= high:
                return f"[BPM_{low}_{high}]"
        return "[BPM_UNKNOWN]"
        
    def time_signature_to_token(self, ts: Tuple[int, int]) -> str:
        if ts == (4, 4): return "[TIMESIG_4_4]"
        if ts == (3, 4): return "[TIMESIG_3_4]"
        # Add more common time signatures if needed, e.g., (2,4), (6,8)
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
                    # Ensure subdivision_idx is within the expected range [0, max_subdivisions-1]
                    if 0 <= subdivision_idx < self.max_subdivisions:
                        tokens_str.append(f"[{instr_prefix}_AT_{subdivision_idx}]")
                    else:
                        print(f"Warning: Subdivision index {subdivision_idx} for {instr_prefix} is out of range [0, {self.max_subdivisions-1}]. Skipping.")
            else: # No events for this instrument in this bar
                tokens_str.append(f"[NO_{instr_prefix}_EVENTS]")
            tokens_str.append(f"[END_{instr_prefix}_EVENTS]") # Explicit end marker for each instrument's event list
            
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens_str]

    def encode_flow_datum(self, flow_datum: FlowDatum) -> List[int]:
        tokens_str: List[str] = ["[LINE_START]"]
        
        # Syllables
        syllables = min(max(0, flow_datum["syllables"]), self.max_syllables)
        tokens_str.append(f"[SYLLABLES_{syllables}]")
        
        # Start Offset
        # Quantize start_offset_beats: (value / max_value) * num_bins
        offset_normalized = flow_datum["start_offset_beats"] / self.flow_offset_max_beats
        offset_bin_idx = int(np.clip(offset_normalized * self.num_offset_bins, 0, self.num_offset_bins - 1))
        tokens_str.append(f"[OFFSET_BIN_{offset_bin_idx}]")
        
        # Duration
        # Quantize duration_beats: (value / max_value) * num_bins
        duration_normalized = flow_datum["duration_beats"] / self.flow_duration_max_beats
        duration_bin_idx = int(np.clip(duration_normalized * self.num_duration_bins, 0, self.num_duration_bins - 1))
        tokens_str.append(f"[DURATION_BIN_{duration_bin_idx}]")
        
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens_str]

    def encode_song_instance(self, 
                             song_beat_features: List[BarBeatFeatures], 
                             song_flow_data: List[FlowDatum]
                            ) -> Tuple[List[int], List[int], List[int]]:
        """
        Encodes a full song instance (beat features + flow targets) into a single token sequence
        and corresponding segment IDs and intra-line position IDs.
        
        Flow data is grouped by bar_index.
        """
        full_token_ids: List[int] = [self.bos_token_id]
        segment_ids: List[int] = [0] # BOS is part of segment 0 (beat features)
        intra_line_pos_ids: List[int] = [0] # BOS is pos 0

        current_segment_idx = 0 # Segment 0 for beat features of the first bar
        
        # Group flow_data by bar_index for easier lookup
        flow_by_bar: Dict[int, List[FlowDatum]] = {}
        for fd in song_flow_data:
            bar_idx = fd["bar_index"]
            if bar_idx not in flow_by_bar: flow_by_bar[bar_idx] = []
            flow_by_bar[bar_idx].append(fd)

        for bar_features in song_beat_features:
            bar_idx = bar_features["bar_index"]
            
            # --- Encode Bar Features ---
            bar_feature_tokens = self.encode_bar_features(bar_features)
            full_token_ids.extend(bar_feature_tokens)
            segment_ids.extend([current_segment_idx] * len(bar_feature_tokens))
            # Intra-line positions for bar features: 0 for [BAR_START], 1 for BPM, etc.
            intra_line_pos_ids.extend(list(range(len(bar_feature_tokens)))) 
            
            # --- Separator ---
            full_token_ids.append(self.sep_input_flow_token_id)
            current_segment_idx += 1 # Increment segment for the upcoming flow lines for this bar
            segment_ids.append(current_segment_idx) 
            intra_line_pos_ids.append(0) # SEP token is pos 0 of its new segment

            # --- Encode Flow Data for this Bar ---
            bar_flow_lines = flow_by_bar.get(bar_idx, [])
            for line_idx, flow_datum in enumerate(bar_flow_lines):
                flow_line_tokens = self.encode_flow_datum(flow_datum)
                full_token_ids.extend(flow_line_tokens)
                
                # Each line in a bar can be its own segment, or group all lines in a bar to one segment
                # For simplicity, let's give each line its own segment_id after the initial SEP segment
                # This means current_segment_idx increments for each line.
                # Or, all lines in a bar share the segment_idx that started after SEP.
                # Let's try: all lines in this bar use `current_segment_idx`.
                # If we want to distinguish lines *within* a bar for the segment embedding, then increment.
                # Let's use one segment for all flow lines belonging to one bar.
                segment_ids.extend([current_segment_idx] * len(flow_line_tokens))
                
                # Intra-line positions for flow: 0 for [LINE_START], 1 for SYLL, 2 for OFF, 3 for DUR
                intra_line_pos_ids.extend(list(range(len(flow_line_tokens))))

            current_segment_idx += 1 # Next bar's features will be a new segment
            # If no flow lines for this bar, current_segment_idx still increments.

        full_token_ids.append(self.eos_token_id)
        segment_ids.append(current_segment_idx) # EOS belongs to the last segment type
        intra_line_pos_ids.append(0) 
        
        return full_token_ids, segment_ids, intra_line_pos_ids


    def decode_flow_tokens_to_datum(self, flow_tokens: List[int], bar_idx_context: int, line_idx_context: int) -> Optional[FlowDatum]:
        """Decodes a sequence of 3-4 tokens representing a single FlowDatum."""
        # Expected: [LINE_START], [SYLLABLES_X], [OFFSET_BIN_Y], [DURATION_BIN_Z]
        # or just [SYLLABLES_X], [OFFSET_BIN_Y], [DURATION_BIN_Z] if [LINE_START] is consumed externally
        
        tokens_str = [self.id_to_token.get(t_id, "[UNK]") for t_id in flow_tokens]
        
        ptr = 0
        if tokens_str[ptr] == "[LINE_START]": ptr += 1
        
        if len(tokens_str) - ptr < 3: # Needs at least 3 tokens after optional LINE_START
            # print(f"Warning: Insufficient tokens to decode FlowDatum: {tokens_str}")
            return None

        try:
            syll_token = tokens_str[ptr]
            offset_token = tokens_str[ptr+1]
            duration_token = tokens_str[ptr+2]

            syllables = int(syll_token.split('_')[-1].replace(']', ''))
            
            offset_bin_idx = int(offset_token.split('_')[-1].replace(']', ''))
            start_offset_beats = (offset_bin_idx / self.num_offset_bins) * self.flow_offset_max_beats
            
            duration_bin_idx = int(duration_token.split('_')[-1].replace(']', ''))
            duration_beats = (duration_bin_idx / self.num_duration_bins) * self.flow_duration_max_beats
            
            return {
                "bar_index": bar_idx_context, 
                "line_index_in_bar": line_idx_context,
                "syllables": syllables,
                "start_offset_beats": round(start_offset_beats, 3),
                "duration_beats": round(duration_beats, 3)
            }
        except (ValueError, IndexError, AttributeError) as e:
            # print(f"Error decoding flow tokens {tokens_str}: {e}")
            return None

    def get_vocab_size(self) -> int:
        return len(self.token_to_id)

    def save_vocab(self, filepath: str):
        # Ensure keys in id_to_token are strings for JSON serialization if they are not already
        # self.id_to_token keys are already int from _add_token
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k):v for k,v in self.id_to_token.items()},
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
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()} # Ensure keys are int
        
        config_params = vocab_data.get("config_params", {})
        self.max_syllables = config_params.get("max_syllables", self.max_syllables)
        self.num_offset_bins = config_params.get("num_offset_bins", self.num_offset_bins)
        self.num_duration_bins = config_params.get("num_duration_bins", self.num_duration_bins)
        self.bpm_bins = config_params.get("bpm_bins", self.bpm_bins)
        self.max_subdivisions = config_params.get("max_subdivisions", self.max_subdivisions)
        self.flow_offset_max_beats = config_params.get("flow_offset_max_beats", self.flow_offset_max_beats)
        self.flow_duration_max_beats = config_params.get("flow_duration_max_beats", self.flow_duration_max_beats)
        print(f"Vocabulary and config loaded from {filepath}. Size: {len(self.token_to_id)}")

    @property
    def pad_token_id(self) -> int: return self.token_to_id["[PAD]"]
    @property
    def bos_token_id(self) -> int: return self.token_to_id["[BOS]"]
    @property
    def eos_token_id(self) -> int: return self.token_to_id["[EOS]"]
    @property
    def unk_token_id(self) -> int: return self.token_to_id["[UNK]"]
    @property
    def sep_input_flow_token_id(self) -> int: return self.token_to_id["[SEP_INPUT_FLOW]"]
    @property
    def bar_start_token_id(self) -> int: return self.token_to_id["[BAR_START]"]
    @property
    def line_start_token_id(self) -> int: return self.token_to_id["[LINE_START]"]

import os # Already imported

if __name__ == '__main__':
    tokenizer_path = "flow_tokenizer_config_v2.json"
    tokenizer = FlowTokenizer(config_path=tokenizer_path) # Will build if not found
    if not os.path.exists(tokenizer_path): tokenizer.save_vocab(tokenizer_path)
        
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    sample_bar_features_1: BarBeatFeatures = {
        "bar_index": 0, "bpm": 125.0, "time_signature": (4, 4),
        "kick_events": [0, 4, 8, 12], "snare_events": [4, 12],
        "hihat_events": [i for i in range(0,16,2)], "bass_events": []
    }
    sample_bar_features_2: BarBeatFeatures = {
        "bar_index": 1, "bpm": 95.0, "time_signature": (4, 4),
        "kick_events": [0, 8], "snare_events": [4], "hihat_events": [], "bass_events": [0,2,4,6]
    }
    
    sample_flow_datum_b0_l0: FlowDatum = { # Belongs to bar 0
        "bar_index": 0, "line_index_in_bar": 0, "syllables": 10,
        "start_offset_beats": 0.0, "duration_beats": 1.9 
    }
    sample_flow_datum_b0_l1: FlowDatum = { # Belongs to bar 0
        "bar_index": 0, "line_index_in_bar": 1, "syllables": 12,
        "start_offset_beats": 2.0, "duration_beats": 2.1
    }
    sample_flow_datum_b1_l0: FlowDatum = { # Belongs to bar 1
        "bar_index": 1, "line_index_in_bar": 0, "syllables": 8,
        "start_offset_beats": 0.5, "duration_beats": 3.0
    }
    
    song_beats = [sample_bar_features_1, sample_bar_features_2]
    song_flows = [sample_flow_datum_b0_l0, sample_flow_datum_b0_l1, sample_flow_datum_b1_l0]

    token_ids, seg_ids, intra_pos_ids = tokenizer.encode_song_instance(song_beats, song_flows)
    
    print(f"\nEncoded song instance (length {len(token_ids)}):")
    # print(token_ids)
    # print(seg_ids)
    # print(intra_pos_ids)
    
    print("\nTokenized sequence:")
    for i in range(len(token_ids)):
        tok_str = tokenizer.id_to_token.get(token_ids[i], "ERR_TOK")
        seg_str = f"S:{seg_ids[i]}"
        pos_str = f"P:{intra_pos_ids[i]}"
        print(f"{tok_str:<20} {seg_str:<5} {pos_str:<5}")
        if tok_str == "[SEP_INPUT_FLOW]": print("-" * 30)
        if tok_str == "[EOS]": break

    # Test decoding a single flow line
    example_flow_line_tokens = tokenizer.encode_flow_datum(sample_flow_datum_b0_l0)
    print(f"\nExample encoded flow line tokens: {example_flow_line_tokens}")
    print(" ".join([tokenizer.id_to_token.get(tid, "UNK") for tid in example_flow_line_tokens]))
    decoded_fd = tokenizer.decode_flow_tokens_to_datum(example_flow_line_tokens, bar_idx_context=0, line_idx_context=0)
    print(f"Decoded FlowDatum: {decoded_fd}")