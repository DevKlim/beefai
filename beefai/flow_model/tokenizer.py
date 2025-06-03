import json
import os
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np

from beefai.utils.data_types import BarBeatFeatures, FlowDatum, SongBeatFeatures, FlowData, TrainingInstance


class FlowTokenizer:
    def __init__(self, config_path: Optional[str] = None):
        # Default config if none provided (can be part of the class or loaded from a default file)
        default_config = {
            "token_to_id": {
                "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
                "[BAR_START]": 4, "[LINE_START]": 5, "[SEP_INPUT_FLOW]": 6,
                "[END_KICK_EVENTS]": 7, "[END_SNARE_EVENTS]": 8,
                "[END_HIHAT_EVENTS]": 9, "[END_BASS_EVENTS]": 10,
                "[NO_KICK_EVENTS]": 11, "[NO_SNARE_EVENTS]": 12,
                "[NO_HIHAT_EVENTS]": 13, "[NO_BASS_EVENTS]": 14,
                "[END_SYLLABLE_SEQUENCE]": 15, # Changed from END_SYLLABLE_SUBDIVISIONS
            },
            "config_params": {
                "max_syllables": 24, # Max syllables in a line
                "num_offset_bins": 16, # Bins for line start offset
                "num_duration_bins": 32, # Bins for line duration
                "bpm_bins": [ # Define BPM ranges and their corresponding tokens
                    (0, 79), (80, 89), (90, 99), (100, 109), (110, 119), (120, 129),
                    (130, 139), (140, 149), (150, 159), (160, 169), (170, 179), (180, 250)
                ],
                "max_subdivisions": 16, # e.g., 16th notes if bar is 4 beats
                "flow_offset_max_beats": 4.0, # Max offset for a line within a bar (in beats)
                "flow_duration_max_beats": 8.0, # Max duration for a line (in beats)
                # Duration of a single syllable in beats (e.g., 16th, 8th, quarter)
                "syllable_duration_bins_beats": [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0], # 12 bins from 1/32 to dotted half at 120BPM/4beats per bar
                "num_syllable_duration_bins": 13, # Includes a "very short/grace note" bin (index 0) + the 12 defined above
                "num_stress_levels": 3 # 0: none, 1: primary, 2: secondary
            }
        }

        if config_path and os.path.exists(config_path):
            # print(f"[FlowTokenizer] Loading configuration from: {config_path}")
            with open(config_path, 'r') as f:
                loaded_config_data = json.load(f)
            self.token_to_id = loaded_config_data.get("token_to_id", {})
            self.config_params = loaded_config_data.get("config_params", default_config["config_params"])
        else:
            if config_path:
                print(f"[FlowTokenizer] Warning: Config file '{config_path}' not found. Using default configuration and building vocab.")
            else:
                print("[FlowTokenizer] No config path provided. Using default configuration and building vocab.")
            self.token_to_id = dict(default_config["token_to_id"]) # Make a copy
            self.config_params = dict(default_config["config_params"]) # Make a copy
            self._build_vocab_from_params() # Build the rest of the vocab

        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        # Convenience attributes from config_params
        self.max_syllables = self.config_params.get("max_syllables", 24)
        self.num_offset_bins = self.config_params.get("num_offset_bins", 16)
        self.num_duration_bins = self.config_params.get("num_duration_bins", 32)
        self.max_subdivisions = self.config_params.get("max_subdivisions", 16)
        self.flow_offset_max_beats = self.config_params.get("flow_offset_max_beats", 4.0)
        self.flow_duration_max_beats = self.config_params.get("flow_duration_max_beats", 8.0)
        self.syllable_duration_bins_beats = self.config_params.get("syllable_duration_bins_beats", [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0])
        self.num_syllable_duration_bins = self.config_params.get("num_syllable_duration_bins", len(self.syllable_duration_bins_beats) +1) # +1 for the 0th bin
        self.num_stress_levels = self.config_params.get("num_stress_levels", 3)


        # Ensure all special tokens are in vocab and have dedicated IDs
        self.pad_token_id = self.token_to_id["[PAD]"]
        self.unk_token_id = self.token_to_id["[UNK]"]
        self.bos_token_id = self.token_to_id["[BOS]"]
        self.eos_token_id = self.token_to_id["[EOS]"]
        self.bar_start_token_id = self.token_to_id["[BAR_START]"]
        self.line_start_token_id = self.token_to_id["[LINE_START]"]
        self.sep_input_flow_token_id = self.token_to_id["[SEP_INPUT_FLOW]"]
        self.end_syllable_sequence_token_id = self.token_to_id["[END_SYLLABLE_SEQUENCE]"]


    def _add_token(self, token: str):
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)

    def _build_vocab_from_params(self):
        # BPM tokens
        bpm_bins = self.config_params.get("bpm_bins", [])
        for i, (low, high) in enumerate(bpm_bins):
            self._add_token(f"[BPM_{low}_{high}]")
        self._add_token("[BPM_UNKNOWN]")

        # Time signature tokens (simple for now)
        self._add_token("[TIMESIG_4_4]")
        self._add_token("[TIMESIG_3_4]")
        self._add_token("[TIMESIG_OTHER]")

        # Beat event tokens (kick, snare, hihat, bass)
        instruments = ["KICK", "SNARE", "HIHAT", "BASS"]
        max_subdiv = self.config_params.get("max_subdivisions", 16)
        for instr in instruments:
            for i in range(max_subdiv):
                self._add_token(f"[{instr}_AT_{i}]")
            # self._add_token(f"[END_{instr}_EVENTS]") # Already in default
            # self._add_token(f"[NO_{instr}_EVENTS]")   # Already in default

        # Flow related tokens
        # Syllable counts for a line
        for i in range(self.config_params.get("max_syllables", 24) + 1): # 0 to max_syllables
            self._add_token(f"[SYLLABLES_{i}]")

        # Line start offset bins
        for i in range(self.config_params.get("num_offset_bins", 16)):
            self._add_token(f"[OFFSET_BIN_{i}]")

        # Line duration bins
        for i in range(self.config_params.get("num_duration_bins", 32)):
            self._add_token(f"[DURATION_BIN_{i}]")
        
        # Per-syllable start subdivision tokens
        for i in range(max_subdiv): # 0 to max_subdivisions-1
            self._add_token(f"[SYLLABLE_STARTS_SUBDIV_{i}]")

        # Per-syllable duration bin tokens
        num_syl_dur_bins = self.config_params.get("num_syllable_duration_bins", 13)
        for i in range(num_syl_dur_bins):
            self._add_token(f"[SYLLABLE_DURATION_BIN_{i}]")

        # Per-syllable stress level tokens
        num_stress = self.config_params.get("num_stress_levels", 3)
        for i in range(num_stress): # 0, 1, 2
            self._add_token(f"[SYLLABLE_STRESS_{i}]")
        
        # Rebuild id_to_token after potentially adding new tokens
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}


    def get_vocab_size(self) -> int:
        return len(self.token_to_id)

    def save_vocab(self, file_path: str):
        """Saves the tokenizer's vocabulary and configuration to a JSON file."""
        # print(f"[FlowTokenizer] Saving vocabulary and configuration to: {file_path}")
        # Ensure id_to_token is up-to-date if vocab was built dynamically
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        config_to_save = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token, # Also save this for easier inspection
            "config_params": self.config_params
        }
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            # print(f"[FlowTokenizer] Successfully saved to {file_path}")
        except Exception as e:
            print(f"[FlowTokenizer] Error saving tokenizer config to {file_path}: {e}")

    # --- Encoding Beat Features ---
    def _encode_bpm(self, bpm: float) -> int:
        if bpm is None or bpm <= 0: return self.token_to_id["[BPM_UNKNOWN]"]
        for low, high in self.config_params["bpm_bins"]:
            if low <= bpm <= high:
                return self.token_to_id.get(f"[BPM_{low}_{high}]", self.unk_token_id)
        return self.token_to_id["[BPM_UNKNOWN]"] # Fallback for out of defined range

    def _encode_time_signature(self, ts: Tuple[int, int]) -> int:
        if ts == (4, 4): return self.token_to_id["[TIMESIG_4_4]"]
        if ts == (3, 4): return self.token_to_id["[TIMESIG_3_4]"]
        return self.token_to_id["[TIMESIG_OTHER]"]

    def _encode_instrument_events(self, events: List[int], instrument_name: str) -> List[int]:
        token_ids = []
        if not events:
            token_ids.append(self.token_to_id[f"[NO_{instrument_name.upper()}_EVENTS]"])
        else:
            for event_subdiv in sorted(list(set(events))): # Ensure unique and sorted
                if 0 <= event_subdiv < self.max_subdivisions:
                    token_ids.append(self.token_to_id[f"[{instrument_name.upper()}_AT_{event_subdiv}]"])
        token_ids.append(self.token_to_id[f"[END_{instrument_name.upper()}_EVENTS]"])
        return token_ids

    def encode_bar_features(self, bar_features: BarBeatFeatures) -> List[int]:
        tokens = [self.bar_start_token_id]
        tokens.append(self._encode_bpm(bar_features.get("bpm", 0)))
        tokens.append(self._encode_time_signature(bar_features.get("time_signature", (4,4))))
        
        tokens.extend(self._encode_instrument_events(bar_features.get("kick_events", []), "KICK"))
        tokens.extend(self._encode_instrument_events(bar_features.get("snare_events", []), "SNARE"))
        tokens.extend(self._encode_instrument_events(bar_features.get("hihat_events", []), "HIHAT"))
        tokens.extend(self._encode_instrument_events(bar_features.get("bass_events", []), "BASS"))
        return tokens

    # --- Encoding Flow Data ---
    def _quantize_flow_value(self, value: float, max_value: float, num_bins: int) -> int:
        if value is None: return num_bins -1 # Or a specific "unknown" bin if defined
        value = np.clip(value, 0, max_value)
        bin_index = int(np.floor(value / (max_value + 1e-6) * num_bins)) # Ensure value < max_value maps to < num_bins
        return min(bin_index, num_bins - 1)

    def quantize_syllable_duration_to_bin_index(self, duration_sec: float, bpm: float) -> int:
        if bpm <= 0: bpm = 120.0 # Avoid division by zero
        beat_duration_sec = 60.0 / bpm
        duration_beats = duration_sec / beat_duration_sec
        
        # Find the closest bin value
        closest_bin_idx = 0
        min_diff = float('inf')

        # Check against predefined beat values for bins 1 onwards
        for i, bin_beat_val in enumerate(self.syllable_duration_bins_beats):
            diff = abs(duration_beats - bin_beat_val)
            if diff < min_diff:
                min_diff = diff
                closest_bin_idx = i + 1 # +1 because bin 0 is for very short/grace notes
        
        # Heuristic for bin 0 (very short / grace note)
        # If duration is much shorter than the smallest defined bin, assign to bin 0
        if self.syllable_duration_bins_beats and duration_beats < self.syllable_duration_bins_beats[0] * 0.5:
             closest_bin_idx = 0
        
        return min(closest_bin_idx, self.num_syllable_duration_bins -1)


    def encode_flow_datum(self, flow_datum: FlowDatum) -> List[int]:
        tokens = [self.line_start_token_id]
        
        # Syllables
        syllables_val = min(flow_datum["syllables"], self.max_syllables)
        tokens.append(self.token_to_id.get(f"[SYLLABLES_{syllables_val}]", self.unk_token_id))

        # Start Offset
        offset_bin = self._quantize_flow_value(flow_datum["start_offset_beats"], self.flow_offset_max_beats, self.num_offset_bins)
        tokens.append(self.token_to_id.get(f"[OFFSET_BIN_{offset_bin}]", self.unk_token_id))

        # Duration
        duration_bin = self._quantize_flow_value(flow_datum["duration_beats"], self.flow_duration_max_beats, self.num_duration_bins)
        tokens.append(self.token_to_id.get(f"[DURATION_BIN_{duration_bin}]", self.unk_token_id))

        # Syllable-specific tokens
        num_syl_events_to_encode = flow_datum.get("syllables", 0) # Use syllable count from token
        
        syl_starts = flow_datum.get("syllable_start_subdivisions", [])
        syl_durs_quantized = flow_datum.get("syllable_durations_quantized", [])
        syl_stresses = flow_datum.get("syllable_stresses", [])

        for i in range(num_syl_events_to_encode):
            # Syllable Start Subdivision
            if i < len(syl_starts):
                subdiv = min(syl_starts[i], self.max_subdivisions - 1)
                tokens.append(self.token_to_id.get(f"[SYLLABLE_STARTS_SUBDIV_{subdiv}]", self.unk_token_id))
            else: # Should not happen if data is well-formed and num_syl_events_to_encode matches lengths
                tokens.append(self.token_to_id.get(f"[SYLLABLE_STARTS_SUBDIV_0]", self.unk_token_id)) # Fallback


            # Syllable Duration Bin
            if i < len(syl_durs_quantized):
                dur_bin = min(syl_durs_quantized[i], self.num_syllable_duration_bins -1)
                tokens.append(self.token_to_id.get(f"[SYLLABLE_DURATION_BIN_{dur_bin}]", self.unk_token_id))
            else:
                tokens.append(self.token_to_id.get(f"[SYLLABLE_DURATION_BIN_0]", self.unk_token_id)) # Fallback

            # Syllable Stress
            if i < len(syl_stresses):
                stress_val = min(syl_stresses[i], self.num_stress_levels -1 )
                tokens.append(self.token_to_id.get(f"[SYLLABLE_STRESS_{stress_val}]", self.unk_token_id))
            else:
                tokens.append(self.token_to_id.get(f"[SYLLABLE_STRESS_0]", self.unk_token_id)) # Fallback (unstressed)

        tokens.append(self.end_syllable_sequence_token_id)
        return tokens

    def encode_song_instance(self, 
                             song_beat_features: SongBeatFeatures, 
                             song_flow_data: FlowData
                            ) -> Tuple[List[int], List[int], List[int]]:
        """
        Encodes a full song (beat features and flow data) into token IDs, segment IDs, and intra-line position IDs.
        Segment IDs:
        - Beat features for bar X: 2*X
        - Flow lines for bar X: 2*X + 1
        Intra-line Position IDs: Positional index within the current segment component (beat feature block or flow line).
        """
        all_token_ids = [self.bos_token_id]
        all_segment_ids = [0] # BOS is part of the "global" segment or first bar's features
        all_intra_line_pos_ids = [0] # BOS is at position 0 of its conceptual segment

        # Group flow data by bar_index and then by line_index_in_bar
        flow_by_bar: Dict[int, Dict[int, FlowDatum]] = {}
        for fd in song_flow_data:
            bar_idx = fd["bar_index"]
            line_idx = fd["line_index_in_bar"]
            if bar_idx not in flow_by_bar:
                flow_by_bar[bar_idx] = {}
            flow_by_bar[bar_idx][line_idx] = fd
        
        # Sort flow lines within each bar by line_index_in_bar
        for bar_idx in flow_by_bar:
            flow_by_bar[bar_idx] = dict(sorted(flow_by_bar[bar_idx].items()))

        # Determine the maximum bar index from either beat features or flow data
        max_bar_idx_beats = max(bf["bar_index"] for bf in song_beat_features) if song_beat_features else -1
        max_bar_idx_flow = max(flow_by_bar.keys()) if flow_by_bar else -1
        total_bars_to_process = max(max_bar_idx_beats, max_bar_idx_flow) + 1

        for bar_idx in range(total_bars_to_process):
            # --- Beat Features ---
            # Find the beat features for the current bar_idx
            current_bar_beat_feature = next((bf for bf in song_beat_features if bf["bar_index"] == bar_idx), None)
            
            if current_bar_beat_feature:
                beat_feature_tokens = self.encode_bar_features(current_bar_beat_feature)
                segment_id_for_beat_features = 2 * bar_idx # Even segment ID for beat features
                for i, token_id in enumerate(beat_feature_tokens):
                    all_token_ids.append(token_id)
                    all_segment_ids.append(segment_id_for_beat_features)
                    all_intra_line_pos_ids.append(i) # Position within this beat feature block
            # Else: if no beat features for this bar, we might skip or use a placeholder. For now, just skip adding tokens.

            # Separator between beat features and flow lines for this bar
            all_token_ids.append(self.sep_input_flow_token_id)
            # SEP token belongs to the beat features segment or the upcoming flow segment.
            # Let's assign it to the beat features segment if features exist, otherwise to a "transition" segment
            # or the flow segment of the current bar. For simplicity, let's keep it with beat features' segment.
            all_segment_ids.append(segment_id_for_beat_features if current_bar_beat_feature else 2 * bar_idx)
            all_intra_line_pos_ids.append(len(beat_feature_tokens) if current_bar_beat_feature else 0)


            # --- Flow Lines ---
            segment_id_for_flow_lines = 2 * bar_idx + 1 # Odd segment ID for flow lines
            if bar_idx in flow_by_bar:
                bar_flow_lines = flow_by_bar[bar_idx]
                for line_idx_in_bar, flow_datum in bar_flow_lines.items():
                    flow_line_tokens = self.encode_flow_datum(flow_datum)
                    for i, token_id in enumerate(flow_line_tokens):
                        all_token_ids.append(token_id)
                        all_segment_ids.append(segment_id_for_flow_lines)
                        # Intra-line position should ideally reset for each distinct component (flow line)
                        # However, the model config has max_intra_line_positions.
                        # Let's make intra_line_pos_id relative to the start of the *current flow line*.
                        all_intra_line_pos_ids.append(i)
            # Else: if no flow lines for this bar, no flow tokens are added.
            # The BAR_START token for the *next* bar will handle the transition.

        all_token_ids.append(self.eos_token_id)
        all_segment_ids.append(all_segment_ids[-1] + 1) # EOS gets a new segment ID (or belongs to last flow segment)
        all_intra_line_pos_ids.append(0) # EOS is at position 0 of its segment

        return all_token_ids, all_segment_ids, all_intra_line_pos_ids


    # --- Decoding ---
    def _dequantize_flow_value(self, bin_index: int, max_value: float, num_bins: int) -> float:
        """Approximate dequantization. Returns midpoint of the bin."""
        if bin_index < 0 or bin_index >= num_bins: return 0.0 # Or handle error
        bin_width = max_value / num_bins
        return (bin_index + 0.5) * bin_width

    def dequantize_syllable_duration_bin(self, bin_index: int) -> float:
        """Dequantizes a syllable duration bin index back to an approximate duration in beats."""
        if bin_index == 0: # Special case for the "very short/grace note" bin
            # Return a small beat value, e.g., smaller than the first defined bin_beat_val
            return 0.0625 # e.g., half of a 32nd note beat (0.125 / 2)
        elif 1 <= bin_index < self.num_syllable_duration_bins:
            # Bins 1 onwards map to the syllable_duration_bins_beats array
            # The index in the array is bin_index - 1
            return self.syllable_duration_bins_beats[bin_index - 1]
        else: # Invalid bin_index
            return 0.25 # Fallback to a common short duration (e.g., 16th note beat)


    def decode_flow_tokens_to_datum(self, tokens: List[int], bar_idx: int, line_idx_in_bar: int) -> Optional[FlowDatum]:
        if not tokens or tokens[0] != self.line_start_token_id:
            # print(f"DEBUG: Decode fail - Tokens empty or no LINE_START. Tokens: {tokens[:5]}")
            return None

        datum: Dict[str, Any] = {
            "bar_index": bar_idx,
            "line_index_in_bar": line_idx_in_bar,
            "syllables": 0,
            "start_offset_beats": 0.0,
            "duration_beats": 0.0,
            "syllable_start_subdivisions": [],
            "syllable_durations_quantized": [],
            "syllable_stresses": []
        }
        
        # Expected order: LINE_START, SYLLABLES_X, OFFSET_BIN_Y, DURATION_BIN_Z, (SYL_START, SYL_DUR, SYL_STRESS)*, END_SYLLABLE_SEQUENCE
        current_token_idx = 1 # Skip LINE_START

        # 1. Decode Syllables count
        if current_token_idx >= len(tokens): return None # Not enough tokens
        token_str = self.id_to_token.get(tokens[current_token_idx])
        if token_str and token_str.startswith("[SYLLABLES_"):
            try:
                datum["syllables"] = int(token_str.split('_')[-1].strip('[]'))
            except ValueError: return None # Malformed token
        else: return None # Expected SYLLABLES token
        current_token_idx += 1
        
        # 2. Decode Offset Bin
        if current_token_idx >= len(tokens): return None
        token_str = self.id_to_token.get(tokens[current_token_idx])
        if token_str and token_str.startswith("[OFFSET_BIN_"):
            try:
                offset_bin = int(token_str.split('_')[-1].strip('[]'))
                datum["start_offset_beats"] = self._dequantize_flow_value(offset_bin, self.flow_offset_max_beats, self.num_offset_bins)
            except ValueError: return None
        else: return None
        current_token_idx += 1

        # 3. Decode Duration Bin
        if current_token_idx >= len(tokens): return None
        token_str = self.id_to_token.get(tokens[current_token_idx])
        if token_str and token_str.startswith("[DURATION_BIN_"):
            try:
                duration_bin = int(token_str.split('_')[-1].strip('[]'))
                datum["duration_beats"] = self._dequantize_flow_value(duration_bin, self.flow_duration_max_beats, self.num_duration_bins)
            except ValueError: return None
        else: return None
        current_token_idx += 1

        # 4. Decode Syllable sequence (triplets: start_subdiv, duration_bin, stress)
        expected_num_syllables = datum["syllables"]
        
        while current_token_idx < len(tokens) and tokens[current_token_idx] != self.end_syllable_sequence_token_id:
            # Check if we have enough tokens for a full triplet
            if current_token_idx + 2 >= len(tokens): 
                # print(f"DEBUG: Decode fail - Incomplete syllable triplet. Tokens left: {tokens[current_token_idx:]}")
                return None # Not enough tokens for a full triplet

            # Syllable Start Subdivision
            start_subdiv_token_str = self.id_to_token.get(tokens[current_token_idx])
            if not (start_subdiv_token_str and start_subdiv_token_str.startswith("[SYLLABLE_STARTS_SUBDIV_")): return None
            try:
                val = int(start_subdiv_token_str.split('_')[-1].strip('[]'))
                datum["syllable_start_subdivisions"].append(val)
            except ValueError: return None
            current_token_idx += 1

            # Syllable Duration Bin
            syl_dur_token_str = self.id_to_token.get(tokens[current_token_idx])
            if not (syl_dur_token_str and syl_dur_token_str.startswith("[SYLLABLE_DURATION_BIN_")): return None
            try:
                val = int(syl_dur_token_str.split('_')[-1].strip('[]'))
                datum["syllable_durations_quantized"].append(val)
            except ValueError: return None
            current_token_idx += 1
            
            # Syllable Stress
            syl_stress_token_str = self.id_to_token.get(tokens[current_token_idx])
            if not (syl_stress_token_str and syl_stress_token_str.startswith("[SYLLABLE_STRESS_")): return None
            try:
                val = int(syl_stress_token_str.split('_')[-1].strip('[]'))
                datum["syllable_stresses"].append(val)
            except ValueError: return None
            current_token_idx += 1

        # Check for END_SYLLABLE_SEQUENCE token
        if current_token_idx >= len(tokens) or tokens[current_token_idx] != self.end_syllable_sequence_token_id:
            # print(f"DEBUG: Decode fail - Missing END_SYLLABLE_SEQUENCE. Last token: {self.id_to_token.get(tokens[current_token_idx-1]) if current_token_idx > 0 else 'N/A'}")
            return None
        
        # Validate if the number of decoded syllable events matches the declared syllable count
        if len(datum["syllable_start_subdivisions"]) != expected_num_syllables:
            # print(f"DEBUG: Decode Warning/Fail - Mismatch: declared syllables {expected_num_syllables}, found {len(datum['syllable_start_subdivisions'])} events.")
            # Depending on strictness, either return None or allow it but perhaps adjust datum["syllables"]
            # For now, let's be strict for generation quality.
            # If we want to be lenient: datum["syllables"] = len(datum["syllable_start_subdivisions"])
            return None


        return datum # Type: ignore (because TypedDict can be stricter than Dict[str,Any])

    def get_value_from_special_token(self, token_str: str) -> Optional[int]:
        """
        Extracts the integer value from a special token string like "[SYLLABLES_10]".
        Returns None if parsing fails or token is not in the expected format.
        """
        if not isinstance(token_str, str) or not token_str.endswith(']'):
            return None
        
        # Remove brackets and split by underscore
        parts = token_str.strip('[]').split('_')
        if not parts:
            return None
            
        # The value is expected to be the last part
        try:
            return int(parts[-1])
        except ValueError:
            return None # Last part is not an integer

    def get_token_ids_for_category(self, category_prefix: str) -> List[int]:
        """Returns a list of token IDs that start with the given category prefix."""
        # Ensure token_to_id is populated
        if not hasattr(self, 'token_to_id') or not self.token_to_id:
            # This might happen if called before full init or if vocab is empty
            return [] 
            
        return [
            self.token_to_id[token]
            for token in self.token_to_id
            if token.startswith(category_prefix) and token in self.token_to_id
        ]


if __name__ == '__main__':
    # Test basic initialization and vocab building
    tokenizer = FlowTokenizer(config_path="beefai/flow_model/flow_tokenizer_config_v2.json") # Use existing if available
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Save if it was newly built or to ensure format is up-to-date
    # tokenizer.save_vocab("beefai/flow_model/flow_tokenizer_config_v2_test_output.json")
    # print("Test vocab saved to flow_tokenizer_config_v2_test_output.json")

    # Test helper
    print(f"Value for [SYLLABLES_12]: {tokenizer.get_value_from_special_token('[SYLLABLES_12]')}")
    print(f"Value for [OFFSET_BIN_3]: {tokenizer.get_value_from_special_token('[OFFSET_BIN_3]')}")
    print(f"Value for [INVALID_TOKEN]: {tokenizer.get_value_from_special_token('[INVALID_TOKEN]')}")
    print(f"Value for [STRESS_0]: {tokenizer.get_value_from_special_token('[SYLLABLE_STRESS_0]')}")


    # Test encoding a dummy BarBeatFeatures
    dummy_bar_feat: BarBeatFeatures = {
        "bar_index": 0, "bpm": 125.0, "time_signature": (4, 4),
        "kick_events": [0, 4, 8, 12], "snare_events": [4, 12],
        "hihat_events": [0, 2, 4, 6, 8, 10, 12, 14], "bass_events": [0, 8],
        "bar_start_time_sec": 0.0, "bar_duration_sec": 1.92
    }
    encoded_beat_tokens = tokenizer.encode_bar_features(dummy_bar_feat)
    # print(f"\nEncoded Beat Tokens for dummy bar: {encoded_beat_tokens}")
    # print(f"Decoded: {[tokenizer.id_to_token.get(t, 'UNK') for t in encoded_beat_tokens]}")

    # Test encoding a dummy FlowDatum
    dummy_flow_datum: FlowDatum = {
        "bar_index": 0, "line_index_in_bar": 0, "syllables": 5,
        "start_offset_beats": 0.5, "duration_beats": 3.0,
        "syllable_start_subdivisions": [2, 4, 6, 8, 10], # 0.5 beats = 2 subdivisions if 1 beat = 4 subdivs (16ths / 4 per beat)
        "syllable_durations_quantized": [tokenizer.quantize_syllable_duration_to_bin_index(0.25 * (60/125), 125), # 0.25 beats
                                         tokenizer.quantize_syllable_duration_to_bin_index(0.25 * (60/125), 125),
                                         tokenizer.quantize_syllable_duration_to_bin_index(0.5 * (60/125), 125), # 0.5 beats
                                         tokenizer.quantize_syllable_duration_to_bin_index(0.5 * (60/125), 125),
                                         tokenizer.quantize_syllable_duration_to_bin_index(1.0 * (60/125), 125) ], # 1.0 beat
        "syllable_stresses": [1, 0, 1, 0, 0]
    }
    encoded_flow_tokens = tokenizer.encode_flow_datum(dummy_flow_datum)
    # print(f"\nEncoded Flow Tokens for dummy datum: {encoded_flow_tokens}")
    # print(f"Decoded: {[tokenizer.id_to_token.get(t, 'UNK') for t in encoded_flow_tokens]}")

    # Test decoding the dummy FlowDatum
    decoded_datum = tokenizer.decode_flow_tokens_to_datum(encoded_flow_tokens, 0, 0)
    # print(f"\nRe-decoded Flow Datum: {decoded_datum}")
    if decoded_datum:
        assert decoded_datum["syllables"] == dummy_flow_datum["syllables"]
        assert len(decoded_datum["syllable_start_subdivisions"]) == dummy_flow_datum["syllables"]
    else:
        print("ERROR: Failed to re-decode dummy flow datum.")

    # Test encoding a full song instance
    song_beats: SongBeatFeatures = [dummy_bar_feat, {**dummy_bar_feat, "bar_index": 1, "kick_events": [0,8]}]
    song_flow: FlowData = [dummy_flow_datum, {**dummy_flow_datum, "bar_index": 1, "line_index_in_bar":0, "syllables": 3, 
                                             "syllable_start_subdivisions": [0,2,4], 
                                             "syllable_durations_quantized": [1,1,2], 
                                             "syllable_stresses": [1,0,1]}]
    
    # print("\n--- Testing Full Song Instance Encoding ---")
    all_tokens, all_segments, all_intra_pos = tokenizer.encode_song_instance(song_beats, song_flow)
    # print(f"Total tokens: {len(all_tokens)}")
    # print(f"Token IDs (first 30): {all_tokens[:30]}")
    # print(f"Decoded (first 30): {[tokenizer.id_to_token.get(t, 'UNK') for t in all_tokens[:30]]}")
    # print(f"Segment IDs (first 30): {all_segments[:30]}")
    # print(f"Intra-Line Pos IDs (first 30): {all_intra_pos[:30]}")

    # Verify that max segment ID and intra-line pos ID are within reasonable limits
    # (This is more relevant for the model config, but good to check here too)
    # print(f"Max Segment ID generated: {max(all_segments)}")
    # print(f"Max Intra-Line Pos ID generated: {max(all_intra_pos)}")

    # Test category fetching
    # print("\n--- Testing Category Token Fetching ---")
    # print(f"Syllable tokens: {tokenizer.get_token_ids_for_category('[SYLLABLES_')[:5]}")
    # print(f"Offset bin tokens: {tokenizer.get_token_ids_for_category('[OFFSET_BIN_')[:5]}")
    # print(f"Duration bin tokens: {tokenizer.get_token_ids_for_category('[DURATION_BIN_')[:5]}")
    # print(f"Syllable start subdiv tokens: {tokenizer.get_token_ids_for_category('[SYLLABLE_STARTS_SUBDIV_')[:5]}")
    # print(f"Syllable duration bin tokens: {tokenizer.get_token_ids_for_category('[SYLLABLE_DURATION_BIN_')[:5]}")
    # print(f"Syllable stress tokens: {tokenizer.get_token_ids_for_category('[SYLLABLE_STRESS_')[:5]}")