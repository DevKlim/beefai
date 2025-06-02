# beefai/flow_model/tokenizer.py
import json
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import os
import sys
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from beefai.utils.data_types import BarBeatFeatures, FlowDatum
# Ensure get_next_context_ids_for_token is available if used by other modules directly with tokenizer
# For tokenizer's internal use, it doesn't call this, but other parts of the system might.
try:
    from beefai.flow_model.transformer_model import get_next_context_ids_for_token 
except ImportError:
    # This might happen if tokenizer.py is imported before transformer_model.py in some contexts
    # or if there's a circular dependency issue during isolated testing.
    # For the tokenizer's own functionality, this import is not strictly needed within this file.
    # print("Warning: Could not import get_next_context_ids_for_token in tokenizer.py. This is okay if only using tokenizer for encoding/decoding.")
    pass


class FlowTokenizer:
    def __init__(self, config_path: Optional[str] = None):
        # --- Define CURRENT intended parameters directly ---
        self.max_syllables = 24 
        self.num_offset_bins = 16 
        self.num_duration_bins = 32 # For overall line duration
        self.bpm_bins = [(0, 79), (80, 89), (90, 99), (100, 109), (110, 119), 
                         (120, 129), (130, 139), (140, 149), (150, 159), 
                         (160, 169), (170, 179), (180, 250)]
        self.max_subdivisions = 16 
        self.flow_offset_max_beats = 4.0 
        self.flow_duration_max_beats = 8.0 

        # Per-syllable duration bins (in beats)
        self.syllable_duration_bins_beats = [
            0.125, 0.25,  0.375, 0.5,   0.625, 0.75, 
            1.0,   1.25,  1.5,   1.75,  2.0,   3.0 
        ] # 12 upper edges, meaning 13 bins (0 to 12)
        self.num_syllable_duration_bins = len(self.syllable_duration_bins_beats) + 1 # Should be 13

        self.num_stress_levels = 3 # 0: unstressed, 1: primary, 2: secondary

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # If a config_path is provided, load it. This populates token_to_id, id_to_token,
        # and potentially overrides some of the above parameters if explicitly handled in load_vocab.
        if config_path and os.path.exists(config_path):
            self.load_vocab(config_path) 
            # After loading, ensure critical derived parameters like num_syllable_duration_bins
            # are consistent with the loaded syllable_duration_bins_beats if present,
            # or default to the class definition.
            # The load_vocab method now handles this logic to ensure consistency.

        # Build/extend vocabulary using the class's current parameters.
        # If config_path was loaded, extend_existing will be True.
        self._build_vocab(extend_existing=bool(config_path and os.path.exists(config_path)))

    def _add_token(self, token: str): # Removed extend_existing, always managed by _build_vocab
        if token not in self.token_to_id:
            token_id = len(self.token_to_id) # Assign next available ID
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def _build_vocab(self, extend_existing: bool = False):
        if not extend_existing: 
            # If not extending, means we are building from scratch or after a reset.
            # Clear existing mappings.
            self.token_to_id = {}
            self.id_to_token = {}

        # Define all token types based on current class parameters
        special_tokens = [
            "[PAD]", "[UNK]", "[BOS]", "[EOS]", 
            "[BAR_START]", "[LINE_START]",      
            "[SEP_INPUT_FLOW]",                 
            "[END_KICK_EVENTS]", "[END_SNARE_EVENTS]", "[END_HIHAT_EVENTS]", "[END_BASS_EVENTS]",
            "[NO_KICK_EVENTS]", "[NO_SNARE_EVENTS]", "[NO_HIHAT_EVENTS]", "[NO_BASS_EVENTS]",
            "[END_SYLLABLE_SEQUENCE]" # Marks the end of a per-syllable detail sequence
        ]
        for token in special_tokens: self._add_token(token)

        for low, high in self.bpm_bins: self._add_token(f"[BPM_{low}_{high}]")
        self._add_token("[BPM_UNKNOWN]") 
        self._add_token("[TIMESIG_4_4]"); self._add_token("[TIMESIG_3_4]"); self._add_token("[TIMESIG_OTHER]") 

        for instr in ["KICK", "SNARE", "HIHAT", "BASS"]:
            for i in range(self.max_subdivisions): self._add_token(f"[{instr}_AT_{i}]")
        
        for i in range(self.max_syllables + 1): self._add_token(f"[SYLLABLES_{i}]")
        for i in range(self.num_offset_bins): self._add_token(f"[OFFSET_BIN_{i}]")
        for i in range(self.num_duration_bins): self._add_token(f"[DURATION_BIN_{i}]") # For overall line duration
        
        # Per-syllable attribute tokens
        for i in range(self.max_subdivisions): 
            self._add_token(f"[SYLLABLE_STARTS_SUBDIV_{i}]")
        
        # This loop now correctly uses self.num_syllable_duration_bins (which should be 13)
        for i in range(self.num_syllable_duration_bins): 
            self._add_token(f"[SYLLABLE_DURATION_BIN_{i}]") # For per-syllable duration
            
        for i in range(self.num_stress_levels): # 0, 1, 2
            self._add_token(f"[SYLLABLE_STRESS_{i}]")
        
        # Ensure UNK is present if somehow missed (should be in special_tokens)
        if "[UNK]" not in self.token_to_id : self._add_token("[UNK]")


    def bpm_to_token(self, bpm: float) -> str:
        for low, high in self.bpm_bins:
            if low <= bpm <= high:
                return f"[BPM_{low}_{high}]"
        return "[BPM_UNKNOWN]"
        
    def time_signature_to_token(self, ts: Tuple[int, int]) -> str:
        if ts == (4, 4): return "[TIMESIG_4_4]"
        if ts == (3, 4): return "[TIMESIG_3_4]"
        return "[TIMESIG_OTHER]"

    def quantize_syllable_duration_to_bin_index(self, duration_sec: float, bpm: float) -> int:
        """
        Quantizes a syllable's duration in seconds to a bin index, 
        relative to the current bar's BPM. Uses self.syllable_duration_bins_beats.
        This is for individual syllable durations, not the FlowDatum's overall duration_beats.
        """
        if bpm <= 0: bpm = 120.0 
        beat_duration_sec = 60.0 / bpm
        if beat_duration_sec <= 1e-6: return len(self.syllable_duration_bins_beats) # Avoid division by zero

        duration_beats = duration_sec / beat_duration_sec
        
        for idx, upper_edge_beats in enumerate(self.syllable_duration_bins_beats): 
            if duration_beats <= upper_edge_beats:
                return idx 
        # If duration_beats is greater than all upper_edge_beats, it falls into the last bin
        return len(self.syllable_duration_bins_beats) 

    def dequantize_syllable_duration_bin(self, bin_index: int) -> float:
        """
        Dequantizes a syllable duration bin index back to an approximate duration in beats.
        Returns the midpoint of the bin or an estimated value for edge/overflow bins.
        """
        if not (0 <= bin_index < self.num_syllable_duration_bins):
            # Fallback for invalid bin index, return a common short duration
            # print(f"Warning: Invalid syllable duration bin_index {bin_index}. Max is {self.num_syllable_duration_bins-1}. Defaulting.")
            return self.syllable_duration_bins_beats[1] if len(self.syllable_duration_bins_beats) > 1 else 0.25


        if bin_index == 0: # First bin (from 0 up to first edge)
            return self.syllable_duration_bins_beats[0] / 2.0 
        elif bin_index < len(self.syllable_duration_bins_beats): # Bins between edges
            # Midpoint between previous edge and current edge
            lower_edge = self.syllable_duration_bins_beats[bin_index-1]
            upper_edge = self.syllable_duration_bins_beats[bin_index]
            return (lower_edge + upper_edge) / 2.0
        else: # Last bin (overflow, duration > last edge)
            # Estimate as last edge + half of the width of the previous bin
            if len(self.syllable_duration_bins_beats) >= 2:
                last_edge = self.syllable_duration_bins_beats[-1]
                prev_bin_width = self.syllable_duration_bins_beats[-1] - self.syllable_duration_bins_beats[-2]
                return last_edge + prev_bin_width / 2.0
            elif self.syllable_duration_bins_beats: # Only one edge defined
                 return self.syllable_duration_bins_beats[-1] * 1.5 
            else: # No edges defined, should not happen with current setup
                return 0.25 # Fallback

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
            if events: # Ensure events is not None and has content
                for subdivision_idx in events:
                    if 0 <= subdivision_idx < self.max_subdivisions:
                        tokens_str.append(f"[{instr_prefix}_AT_{subdivision_idx}]")
                    # else: print(f"Warning: Invalid subdivision_idx {subdivision_idx} for {instr_prefix}")
            else: 
                tokens_str.append(f"[NO_{instr_prefix}_EVENTS]")
            tokens_str.append(f"[END_{instr_prefix}_EVENTS]") 
            
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens_str]

    def encode_flow_datum(self, flow_datum: FlowDatum, bar_bpm: float) -> List[int]:
        tokens_str: List[str] = ["[LINE_START]"]
        
        syllables_count_for_token = min(max(0, flow_datum["syllables"]), self.max_syllables)
        tokens_str.append(f"[SYLLABLES_{syllables_count_for_token}]")
        
        offset_normalized = flow_datum["start_offset_beats"] / self.flow_offset_max_beats
        offset_bin_idx = int(np.clip(offset_normalized * self.num_offset_bins, 0, self.num_offset_bins - 1))
        tokens_str.append(f"[OFFSET_BIN_{offset_bin_idx}]")
        
        duration_normalized = flow_datum["duration_beats"] / self.flow_duration_max_beats
        duration_bin_idx = int(np.clip(duration_normalized * self.num_duration_bins, 0, self.num_duration_bins - 1))
        tokens_str.append(f"[DURATION_BIN_{duration_bin_idx}]")

        # These are per-syllable details
        syllable_starts_subdiv = flow_datum.get("syllable_start_subdivisions", [])
        # This key "syllable_durations_quantized" comes from FlowDataExtractor's output, 
        # which uses SYLLABLE_DURATION_BINS_SEC_FDE (12 bins, 0-11 index, or 12 for overflow).
        # These are NOT the same bins as self.syllable_duration_bins_beats (13 bins, 0-12 index).
        # The FlowDataExtractor should ideally use the tokenizer's bins, or we need a mapping.
        # For now, assuming FlowDataExtractor's "syllable_durations_quantized" are indices 
        # compatible with `[SYLLABLE_DURATION_BIN_X]` where X is 0 to self.num_syllable_duration_bins-1 (i.e. 0-12).
        syllable_durations_quantized_indices = flow_datum.get("syllable_durations_quantized", []) 
        syllable_stresses = flow_datum.get("syllable_stresses", []) 

        num_actual_syllable_events_in_datum = len(syllable_starts_subdiv)
        
        # Ensure all per-syllable lists are of the same length as syllable_starts_subdiv for safety
        # though FlowDataExtractor should already ensure this.
        if len(syllable_durations_quantized_indices) != num_actual_syllable_events_in_datum:
            # Pad or truncate if necessary, though this indicates an upstream issue.
            # For simplicity, we'll use the shortest length if they mismatch.
            # print(f"Warning: Mismatch in per-syllable data lengths for flow_datum. Using min length.")
            min_len_of_related_syl_data = min(num_actual_syllable_events_in_datum, 
                                              len(syllable_durations_quantized_indices), 
                                              len(syllable_stresses))
        else:
            min_len_of_related_syl_data = num_actual_syllable_events_in_datum
        
        # Encode up to `syllables_count_for_token` or the number of actual events, whichever is smaller.
        num_syllable_events_to_encode = min(syllables_count_for_token, min_len_of_related_syl_data)

        for i in range(num_syllable_events_to_encode):
            sub_idx = syllable_starts_subdiv[i]
            # The `fde_dur_bin_idx` is assumed to be an index from 0 to N-1 where N is self.num_syllable_duration_bins
            fde_dur_bin_idx = syllable_durations_quantized_indices[i] 
            stress_val = syllable_stresses[i]

            # Syllable start subdivision token
            if 0 <= sub_idx < self.max_subdivisions:
                tokens_str.append(f"[SYLLABLE_STARTS_SUBDIV_{sub_idx}]")
            else: tokens_str.append(self.id_to_token[self.unk_token_id]) # Should not happen if data is clean
            
            # Per-syllable duration bin token
            # Here, fde_dur_bin_idx must be a valid index for the tokenizer's syllable duration bins.
            if 0 <= fde_dur_bin_idx < self.num_syllable_duration_bins:
                tokens_str.append(f"[SYLLABLE_DURATION_BIN_{fde_dur_bin_idx}]")
            else: 
                # print(f"Warning: Invalid syllable duration bin index {fde_dur_bin_idx} from FDE. Max is {self.num_syllable_duration_bins-1}. Using UNK.")
                tokens_str.append(self.id_to_token[self.unk_token_id])

            # Syllable stress token
            if 0 <= stress_val < self.num_stress_levels:
                tokens_str.append(f"[SYLLABLE_STRESS_{stress_val}]")
            else: tokens_str.append(self.id_to_token[self.unk_token_id]) # Default to unstressed or UNK if invalid

        tokens_str.append("[END_SYLLABLE_SEQUENCE]")
        
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens_str]

    def encode_song_instance(self, 
                             song_beat_features: List[BarBeatFeatures], 
                             song_flow_data: List[FlowDatum]
                            ) -> Tuple[List[int], List[int], List[int]]:
        full_token_ids: List[int] = [self.bos_token_id]
        segment_ids: List[int] = [0] 
        intra_line_pos_ids: List[int] = [0] 
        
        flow_by_bar: Dict[int, List[FlowDatum]] = {}
        for fd in song_flow_data:
            bar_idx = fd["bar_index"]
            if bar_idx not in flow_by_bar: flow_by_bar[bar_idx] = []
            flow_by_bar[bar_idx].append(fd)

        for bar_feature_idx, bar_features in enumerate(song_beat_features):
            bar_idx = bar_features["bar_index"] 
            bar_bpm = bar_features.get("bpm", 120.0) 
            
            # Segment ID for beat features of this bar
            # Each bar's features get a unique segment ID.
            # Using bar_feature_idx * 2 ensures beat features and flow data for the same bar
            # can have adjacent but distinct segment IDs.
            seg_id_for_this_bar_features = bar_feature_idx * 2 
            
            bar_feature_tokens = self.encode_bar_features(bar_features)
            full_token_ids.extend(bar_feature_tokens)
            segment_ids.extend([seg_id_for_this_bar_features] * len(bar_feature_tokens))
            # Intra-line positions reset for each new type of block (beat features, flow line)
            intra_line_pos_ids.extend(list(range(len(bar_feature_tokens)))) 
            
            # Segment ID for flow data of this bar
            seg_id_for_this_bar_flow = seg_id_for_this_bar_features + 1 
            
            full_token_ids.append(self.sep_input_flow_token_id)
            segment_ids.append(seg_id_for_this_bar_flow) 
            intra_line_pos_ids.append(0) # SEP token is pos 0 of the flow segment part

            bar_flow_lines = flow_by_bar.get(bar_idx, [])
            
            # This history is for calculating intra-line positions *within* the flow part of this bar's segment
            current_flow_segment_token_history_for_intra_pos = [self.sep_input_flow_token_id] 

            for line_idx, flow_datum in enumerate(bar_flow_lines):
                flow_line_tokens = self.encode_flow_datum(flow_datum, bar_bpm) 
                full_token_ids.extend(flow_line_tokens)
                segment_ids.extend([seg_id_for_this_bar_flow] * len(flow_line_tokens))
                
                # Calculate intra-line positions for this specific flow line
                line_intra_positions = []
                for tok_id in flow_line_tokens:
                    # Use a large max_segment_types and max_intra_line_positions for get_next_context_ids_for_token
                    # as we are only interested in the intra_id relative to the current_flow_segment_token_history_for_intra_pos.
                    # The segment_id returned here is not used for the final segment_ids list.
                    _, intra_id = get_next_context_ids_for_token(current_flow_segment_token_history_for_intra_pos, tok_id, self, 10000, 10000)
                    line_intra_positions.append(intra_id)
                    current_flow_segment_token_history_for_intra_pos.append(tok_id) 
                intra_line_pos_ids.extend(line_intra_positions)
        
        # EOS token
        # It should belong to a segment ID that's one greater than the last used segment ID.
        eos_segment_id = segment_ids[-1] + 1 if segment_ids and segment_ids[-1] is not None else 0 
        # If only BOS was added, segment_ids would be [0], so eos_segment_id becomes 1.
        if full_token_ids == [self.bos_token_id]: # Only BOS was in the sequence
             eos_segment_id = 0 # Or 1, depending on how EOS after only BOS should be handled. Let's make it 0 for simplicity here.
        
        full_token_ids.append(self.eos_token_id)
        segment_ids.append(eos_segment_id) 
        intra_line_pos_ids.append(0) # EOS is pos 0 of its segment
        
        return full_token_ids, segment_ids, intra_line_pos_ids

    def decode_flow_tokens_to_datum(self, flow_tokens: List[int], bar_idx_context: int, line_idx_context: int) -> Optional[FlowDatum]:
        tokens_str = [self.id_to_token.get(t_id, "[UNK]") for t_id in flow_tokens]
        
        ptr = 0
        if not tokens_str or tokens_str[ptr] != "[LINE_START]": 
            # print(f"Decoder: Expected [LINE_START], got {tokens_str[ptr] if tokens_str else 'None'}")
            return None 
        ptr += 1
        
        # Need at least: SYLLABLES, OFFSET, DURATION, END_SYLLABLE_SEQUENCE (4 tokens after LINE_START)
        if len(tokens_str) - ptr < 4: 
            # print(f"Decoder: Token sequence too short after [LINE_START]. Length: {len(tokens_str)-ptr}")
            return None 

        try:
            syll_token = tokens_str[ptr]; ptr +=1
            offset_token = tokens_str[ptr]; ptr +=1
            duration_token = tokens_str[ptr]; ptr +=1

            if not (syll_token.startswith("[SYLLABLES_") and \
                    offset_token.startswith("[OFFSET_BIN_") and \
                    duration_token.startswith("[DURATION_BIN_")):
                # print(f"Decoder: Missing or malformed SYLLABLES/OFFSET/DURATION tokens. Got: {syll_token}, {offset_token}, {duration_token}")
                return None

            syllables_target_count = int(syll_token.split('_')[-1].replace(']', ''))
            offset_bin_idx = int(offset_token.split('_')[-1].replace(']', ''))
            start_offset_beats = (offset_bin_idx / self.num_offset_bins) * self.flow_offset_max_beats
            duration_bin_idx = int(duration_token.split('_')[-1].replace(']', ''))
            duration_beats = (duration_bin_idx / self.num_duration_bins) * self.flow_duration_max_beats

            temp_syllable_starts: List[int] = []
            temp_syllable_durations: List[int] = [] 
            temp_syllable_stresses: List[int] = []

            actual_syl_events_found = 0
            # Loop to parse per-syllable (start_subdiv, duration_bin, stress) triplets
            while ptr < len(tokens_str) and tokens_str[ptr].startswith("[SYLLABLE_STARTS_SUBDIV_"):
                # Check if there are enough tokens for a full (start, duration, stress) triplet
                if ptr + 2 >= len(tokens_str): 
                    # print(f"Decoder: Incomplete syllable triplet at end of sequence. Current token: {tokens_str[ptr]}")
                    return None 

                subdiv_token_str = tokens_str[ptr]
                sub_idx = int(subdiv_token_str.split('_')[-1].replace(']', ''))
                temp_syllable_starts.append(sub_idx)
                ptr += 1
                
                if not tokens_str[ptr].startswith("[SYLLABLE_DURATION_BIN_"):
                    # print(f"Decoder: Expected SYLLABLE_DURATION_BIN, got {tokens_str[ptr]}")
                    return None 
                dur_bin_token_str = tokens_str[ptr]
                dur_bin_idx_for_token = int(dur_bin_token_str.split('_')[-1].replace(']', ''))
                temp_syllable_durations.append(dur_bin_idx_for_token)
                ptr += 1

                if not tokens_str[ptr].startswith("[SYLLABLE_STRESS_"):
                    # print(f"Decoder: Expected SYLLABLE_STRESS, got {tokens_str[ptr]}")
                    return None
                stress_token_str = tokens_str[ptr]
                stress_val = int(stress_token_str.split('_')[-1].replace(']', ''))
                temp_syllable_stresses.append(stress_val)
                ptr += 1
                actual_syl_events_found +=1
            
            if not (ptr < len(tokens_str) and tokens_str[ptr] == "[END_SYLLABLE_SEQUENCE]"):
                # print(f"Decoder: Expected [END_SYLLABLE_SEQUENCE], got {tokens_str[ptr] if ptr < len(tokens_str) else 'EOS'}")
                return None
            ptr += 1 # Consume [END_SYLLABLE_SEQUENCE]

            # Ensure no trailing tokens after [END_SYLLABLE_SEQUENCE] within this conceptual line
            if ptr != len(flow_tokens): 
                # print(f"Decoder: Trailing tokens found after [END_SYLLABLE_SEQUENCE]. Parsed {ptr}, total {len(flow_tokens)}")
                return None # Or handle this differently, e.g. ignore trailing

            # --- Syllable count enforcement ---
            final_syllable_starts = temp_syllable_starts
            final_syllable_durations = temp_syllable_durations
            final_syllable_stresses = temp_syllable_stresses

            if actual_syl_events_found > syllables_target_count:
                # If more events were decoded than the [SYLLABLES_X] token specified, truncate.
                final_syllable_starts = temp_syllable_starts[:syllables_target_count]
                final_syllable_durations = temp_syllable_durations[:syllables_target_count]
                final_syllable_stresses = temp_syllable_stresses[:syllables_target_count]
            elif actual_syl_events_found < syllables_target_count:
                # If fewer events were decoded, pad with default values.
                num_to_pad = syllables_target_count - actual_syl_events_found
                default_start_subdiv = 0
                # Try to continue pattern if some events exist
                if temp_syllable_starts: 
                    # A simple heuristic: place subsequent syllables roughly one beat (4 subdivisions) apart
                    default_start_subdiv = (temp_syllable_starts[-1] + (self.max_subdivisions // 4)) % self.max_subdivisions 
                
                default_duration_bin_idx = 1 # Corresponds to a short duration (e.g., 0.25 beats)
                if self.num_syllable_duration_bins > 1: # Check if SYLLABLE_DURATION_BIN_1 exists
                    if not f"[SYLLABLE_DURATION_BIN_{default_duration_bin_idx}]" in self.token_to_id:
                        default_duration_bin_idx = 0 # Fallback to bin 0
                else: default_duration_bin_idx = 0


                default_stress = 0 # Unstressed
                if self.num_stress_levels > 0: # Check if SYLLABLE_STRESS_0 exists
                     if not f"[SYLLABLE_STRESS_{default_stress}]" in self.token_to_id:
                         default_stress = 0 # Should always be valid if num_stress_levels > 0
                
                for _ in range(num_to_pad):
                    final_syllable_starts.append(default_start_subdiv)
                    final_syllable_durations.append(default_duration_bin_idx)
                    final_syllable_stresses.append(default_stress)
                    # Advance default_start_subdiv for the next padding, if any
                    default_start_subdiv = (default_start_subdiv + (self.max_subdivisions // 4)) % self.max_subdivisions


            return {
                "bar_index": bar_idx_context, 
                "line_index_in_bar": line_idx_context,
                "syllables": syllables_target_count, # The target count from the token
                "start_offset_beats": round(start_offset_beats, 3),
                "duration_beats": round(duration_beats, 3),
                "syllable_start_subdivisions": final_syllable_starts,
                "syllable_durations_quantized": final_syllable_durations, # These are indices for SYLLABLE_DURATION_BIN_X
                "syllable_stresses": final_syllable_stresses
            }
        except (ValueError, IndexError, AttributeError) as e:
            # print(f"Decoder: Error during token parsing: {e}")
            return None

    def save_vocab(self, filepath: str):
        # Ensure current class parameters are what's saved
        config_params_to_save = {
            "max_syllables": self.max_syllables,
            "num_offset_bins": self.num_offset_bins,
            "num_duration_bins": self.num_duration_bins, # For overall line duration
            "bpm_bins": self.bpm_bins,
            "max_subdivisions": self.max_subdivisions,
            "flow_offset_max_beats": self.flow_offset_max_beats,
            "flow_duration_max_beats": self.flow_duration_max_beats,
            "syllable_duration_bins_beats": self.syllable_duration_bins_beats, # Edges for per-syllable duration
            "num_syllable_duration_bins": self.num_syllable_duration_bins, # Number of bins for per-syllable duration
            "num_stress_levels": self.num_stress_levels
        }
        # id_to_token keys are integers, need to be strings for JSON
        id_to_token_serializable = {str(k): v for k, v in self.id_to_token.items()}
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": id_to_token_serializable, 
            "config_params": config_params_to_save 
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        # print(f"Tokenizer vocabulary and config saved to {filepath}")

    def load_vocab(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data["token_to_id"]
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()} 
        
        config_params_loaded = vocab_data.get("config_params", {})
        
        # Load parameters, prioritizing class defaults if a loaded param seems inconsistent 
        # (e.g. if tokenizer code was updated since config was saved).
        self.max_syllables = config_params_loaded.get("max_syllables", self.max_syllables)
        self.num_offset_bins = config_params_loaded.get("num_offset_bins", self.num_offset_bins)
        self.num_duration_bins = config_params_loaded.get("num_duration_bins", self.num_duration_bins)
        self.bpm_bins = config_params_loaded.get("bpm_bins", self.bpm_bins)
        self.max_subdivisions = config_params_loaded.get("max_subdivisions", self.max_subdivisions)
        self.flow_offset_max_beats = config_params_loaded.get("flow_offset_max_beats", self.flow_offset_max_beats)
        self.flow_duration_max_beats = config_params_loaded.get("flow_duration_max_beats", self.flow_duration_max_beats)
        self.num_stress_levels = config_params_loaded.get("num_stress_levels", self.num_stress_levels)

        # Special handling for syllable_duration_bins_beats and num_syllable_duration_bins
        # to ensure consistency if the class definition (source of truth) has changed.
        loaded_syl_dur_bins_beats = config_params_loaded.get("syllable_duration_bins_beats")
        if loaded_syl_dur_bins_beats is not None:
            self.syllable_duration_bins_beats = loaded_syl_dur_bins_beats
            # Recalculate num_syllable_duration_bins based on loaded edges
            self.num_syllable_duration_bins = len(self.syllable_duration_bins_beats) + 1
        # Else, the class defaults set in __init__ remain.

        # Verify loaded num_syllable_duration_bins if it exists in config, against derived value
        loaded_num_syl_dur_bins = config_params_loaded.get("num_syllable_duration_bins")
        if loaded_num_syl_dur_bins is not None and loaded_num_syl_dur_bins != self.num_syllable_duration_bins:
            # This implies an inconsistency; the number of bins derived from loaded edges (or class default edges)
            # is the source of truth.
            # print(f"Warning: 'num_syllable_duration_bins' in loaded config ({loaded_num_syl_dur_bins}) "
            #       f"differs from value derived from 'syllable_duration_bins_beats' ({self.num_syllable_duration_bins}). "
            #       f"Using derived value: {self.num_syllable_duration_bins}.")
            pass # self.num_syllable_duration_bins is already correctly set based on edges.

        # print(f"Tokenizer vocabulary and config loaded from {filepath}")


    def get_vocab_size(self) -> int: return len(self.token_to_id)
    @property
    def pad_token_id(self) -> int: return self.token_to_id["[PAD]"]
    @property
    def bos_token_id(self) -> int: return self.token_to_id["[BOS]"]
    @property
    def eos_token_id(self) -> int: return self.token_to_id["[EOS]"]
    @property
    def unk_token_id(self) -> int: return self.token_to_id.get("[UNK]", 1) # Default to 1 if UNK somehow missing
    @property
    def sep_input_flow_token_id(self) -> int: return self.token_to_id["[SEP_INPUT_FLOW]"]
    @property
    def bar_start_token_id(self) -> int: return self.token_to_id["[BAR_START]"]
    @property
    def line_start_token_id(self) -> int: return self.token_to_id["[LINE_START]"]
    @property
    def end_syllable_sequence_token_id(self) -> int: return self.token_to_id["[END_SYLLABLE_SEQUENCE]"]


if __name__ == '__main__':
    tokenizer_path = "beefai/flow_model/flow_tokenizer_config_v2.json" 
    print(f"Attempting to load/create tokenizer config at: {tokenizer_path}")
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    
    tokenizer = FlowTokenizer(config_path=tokenizer_path) 
    
    initial_vocab_size_before_save = tokenizer.get_vocab_size()
    tokenizer.save_vocab(tokenizer_path) 
    print(f"Tokenizer config saved to {tokenizer_path}. Vocab size: {tokenizer.get_vocab_size()}")
    
    if tokenizer.get_vocab_size() > initial_vocab_size_before_save:
        print(f"Note: Tokenizer vocabulary was extended from {initial_vocab_size_before_save} to {tokenizer.get_vocab_size()} during init/save.")
    elif initial_vocab_size_before_save > tokenizer.get_vocab_size() :
         print(f"Note: Tokenizer vocabulary SHRUNK from {initial_vocab_size_before_save} to {tokenizer.get_vocab_size()}. This can happen if tokens were removed from _build_vocab logic.")

    # Sanity checks for current parameters
    print(f"  Max Syllables: {tokenizer.max_syllables}")
    print(f"  Num Offset Bins: {tokenizer.num_offset_bins}")
    print(f"  Num Duration Bins (line): {tokenizer.num_duration_bins}")
    print(f"  Num Subdivisions: {tokenizer.max_subdivisions}")
    print(f"  Syllable Duration Bin Edges (beats): {tokenizer.syllable_duration_bins_beats}")
    print(f"  Num Syllable Duration Bins (per-syl): {tokenizer.num_syllable_duration_bins} (Expected: {len(tokenizer.syllable_duration_bins_beats) + 1})")
    assert tokenizer.num_syllable_duration_bins == len(tokenizer.syllable_duration_bins_beats) + 1
    assert f"[SYLLABLE_DURATION_BIN_{tokenizer.num_syllable_duration_bins - 1}]" in tokenizer.token_to_id, \
        f"Max syllable duration bin token '[SYLLABLE_DURATION_BIN_{tokenizer.num_syllable_duration_bins - 1}]' missing."
    
    print(f"  Num Stress Levels: {tokenizer.num_stress_levels} (Expected: 3)")
    assert tokenizer.num_stress_levels == 3
    assert "[SYLLABLE_STRESS_0]" in tokenizer.token_to_id
    assert "[SYLLABLE_STRESS_1]" in tokenizer.token_to_id
    assert "[SYLLABLE_STRESS_2]" in tokenizer.token_to_id
    
    print("\nTokenizer parameter check complete based on current class definition and saved/loaded config.")

    # Test decoding with syllable count enforcement and stress
    print("\n--- Testing decode_flow_tokens_to_datum (with stress & syllable count enforcement) ---")
    # Target SYLLABLES_2, but 3 events provided
    tokens_too_many_stress = [
        tokenizer.line_start_token_id,
        tokenizer.token_to_id['[SYLLABLES_2]'], 
        tokenizer.token_to_id['[OFFSET_BIN_1]'],
        tokenizer.token_to_id['[DURATION_BIN_5]'],
        tokenizer.token_to_id['[SYLLABLE_STARTS_SUBDIV_0]'], tokenizer.token_to_id['[SYLLABLE_DURATION_BIN_1]'], tokenizer.token_to_id['[SYLLABLE_STRESS_0]'],
        tokenizer.token_to_id['[SYLLABLE_STARTS_SUBDIV_4]'], tokenizer.token_to_id['[SYLLABLE_DURATION_BIN_2]'], tokenizer.token_to_id['[SYLLABLE_STRESS_1]'],
        tokenizer.token_to_id['[SYLLABLE_STARTS_SUBDIV_8]'], tokenizer.token_to_id['[SYLLABLE_DURATION_BIN_1]'], tokenizer.token_to_id['[SYLLABLE_STRESS_0]'], 
        tokenizer.end_syllable_sequence_token_id
    ]
    decoded_too_many_stress = tokenizer.decode_flow_tokens_to_datum(tokens_too_many_stress, 0, 0)
    print(f"Decoded (target 2 syls, 3 events provided): {decoded_too_many_stress}")
    assert decoded_too_many_stress is not None and decoded_too_many_stress['syllables'] == 2
    assert len(decoded_too_many_stress['syllable_start_subdivisions']) == 2
    assert len(decoded_too_many_stress['syllable_stresses']) == 2
    assert decoded_too_many_stress['syllable_stresses'] == [0, 1] # Check stress values

    # Target SYLLABLES_3, but 1 event provided
    tokens_too_few_stress = [
        tokenizer.line_start_token_id,
        tokenizer.token_to_id['[SYLLABLES_3]'], 
        tokenizer.token_to_id['[OFFSET_BIN_2]'],
        tokenizer.token_to_id['[DURATION_BIN_6]'],
        tokenizer.token_to_id['[SYLLABLE_STARTS_SUBDIV_2]'], tokenizer.token_to_id['[SYLLABLE_DURATION_BIN_3]'], tokenizer.token_to_id['[SYLLABLE_STRESS_1]'], 
        tokenizer.end_syllable_sequence_token_id
    ]
    decoded_too_few_stress = tokenizer.decode_flow_tokens_to_datum(tokens_too_few_stress, 0, 1)
    print(f"Decoded (target 3 syls, 1 event provided): {decoded_too_few_stress}")
    assert decoded_too_few_stress is not None and decoded_too_few_stress['syllables'] == 3
    assert len(decoded_too_few_stress['syllable_start_subdivisions']) == 3
    assert len(decoded_too_few_stress['syllable_stresses']) == 3
    assert decoded_too_few_stress['syllable_stresses'][0] == 1 # First one is from input
    assert decoded_too_few_stress['syllable_stresses'][1] == 0 # Padded default
    assert decoded_too_few_stress['syllable_stresses'][2] == 0 # Padded default
    
    print("\nTokenizer sanity checks and advanced decoding tests passed.")