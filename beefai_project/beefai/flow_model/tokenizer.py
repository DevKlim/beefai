import json
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
import os

from beefai.utils.data_types import BarBeatFeatures, FlowDatum

class FlowTokenizer:
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration parameters
        self.max_syllables = 24 
        self.num_offset_bins = 16 
        self.num_duration_bins = 32 
        self.bpm_bins = [(0, 79), (80, 89), (90, 99), (100, 109), (110, 119), 
                         (120, 129), (130, 139), (140, 149), (150, 159), 
                         (160, 169), (170, 179), (180, 250)]
        self.max_subdivisions = 16 
        self.flow_offset_max_beats = 4.0 
        self.flow_duration_max_beats = 8.0 

        self.syllable_duration_bins_beats = [
            0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0
        ] # upper edges in beats
        self.num_syllable_duration_bins = len(self.syllable_duration_bins_beats) + 1

        self.num_stress_levels = 3 # 0: none, 1: primary, 2: secondary

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        if config_path and os.path.exists(config_path):
            self.load_vocab(config_path)
            initial_vocab_size = len(self.token_to_id)
            self._build_vocab(extend_existing=True) 
            if len(self.token_to_id) > initial_vocab_size:
                print(f"Tokenizer vocabulary extended. New size: {len(self.token_to_id)}. Consider re-saving.")
        else:
            if config_path: print(f"Tokenizer config {config_path} not found. Building vocab from defaults.")
            self._build_vocab()

    def _add_token(self, token: str, extend_existing: bool = False):
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def _build_vocab(self, extend_existing: bool = False):
        if not extend_existing:
            self.token_to_id = {}
            self.id_to_token = {}

        special_tokens = [
            "[PAD]", "[UNK]", "[BOS]", "[EOS]", 
            "[BAR_START]", "[LINE_START]",      
            "[SEP_INPUT_FLOW]",                 
            "[END_KICK_EVENTS]", "[END_SNARE_EVENTS]", "[END_HIHAT_EVENTS]", "[END_BASS_EVENTS]",
            "[NO_KICK_EVENTS]", "[NO_SNARE_EVENTS]", "[NO_HIHAT_EVENTS]", "[NO_BASS_EVENTS]",
            "[END_SYLLABLE_SEQUENCE]" 
        ]
        for token in special_tokens: self._add_token(token, extend_existing)

        for i, (low, high) in enumerate(self.bpm_bins): self._add_token(f"[BPM_{low}_{high}]", extend_existing)
        self._add_token("[BPM_UNKNOWN]", extend_existing) 
        self._add_token("[TIMESIG_4_4]", extend_existing); self._add_token("[TIMESIG_3_4]", extend_existing); self._add_token("[TIMESIG_OTHER]", extend_existing) 

        for instr in ["KICK", "SNARE", "HIHAT", "BASS"]:
            for i in range(self.max_subdivisions): self._add_token(f"[{instr}_AT_{i}]", extend_existing)
        for i in range(self.max_syllables + 1): self._add_token(f"[SYLLABLES_{i}]", extend_existing)
        for i in range(self.num_offset_bins): self._add_token(f"[OFFSET_BIN_{i}]", extend_existing)
        for i in range(self.num_duration_bins): self._add_token(f"[DURATION_BIN_{i}]", extend_existing)
        
        for i in range(self.max_subdivisions): 
            self._add_token(f"[SYLLABLE_STARTS_SUBDIV_{i}]", extend_existing)
        
        for i in range(self.num_syllable_duration_bins): 
            self._add_token(f"[SYLLABLE_DURATION_BIN_{i}]", extend_existing)
            
        for i in range(self.num_stress_levels):
            self._add_token(f"[SYLLABLE_STRESS_{i}]", extend_existing)
        
        if "[UNK]" not in self.token_to_id : self._add_token("[UNK]", extend_existing)

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
        if bpm <= 0: bpm = 120.0 
        beat_duration_sec = 60.0 / bpm
        duration_beats = duration_sec / beat_duration_sec
        
        for idx, upper_edge_beats in enumerate(self.syllable_duration_bins_beats):
            if duration_beats <= upper_edge_beats:
                return idx
        return len(self.syllable_duration_bins_beats) 

    def dequantize_syllable_duration_bin(self, bin_index: int) -> float:
        """
        Converts a syllable duration bin index back to an approximate duration in beats.
        Returns the midpoint of the bin or the lower bound for the first/last bin.
        """
        if not (0 <= bin_index < self.num_syllable_duration_bins):
            # print(f"Warning: Syllable duration bin_index {bin_index} out of range. Returning 0.25 beats as default.")
            return 0.25 # Default to a 16th note duration in beats

        if bin_index == 0: # First bin (up to self.syllable_duration_bins_beats[0])
            return self.syllable_duration_bins_beats[0] / 2.0 
        elif bin_index < len(self.syllable_duration_bins_beats): # Intermediate bins
            lower_edge = self.syllable_duration_bins_beats[bin_index-1]
            upper_edge = self.syllable_duration_bins_beats[bin_index]
            return (lower_edge + upper_edge) / 2.0
        else: # Last bin (greater than the last defined edge)
            # Return the last edge + a bit, e.g., half of the previous bin's width
            if len(self.syllable_duration_bins_beats) > 1:
                last_edge = self.syllable_duration_bins_beats[-1]
                prev_bin_width = self.syllable_duration_bins_beats[-1] - self.syllable_duration_bins_beats[-2]
                return last_edge + prev_bin_width / 2.0
            else: # Only one edge defined, so this is > that edge
                return self.syllable_duration_bins_beats[-1] * 1.5


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
                        print(f"Warning: Subdivision index {subdivision_idx} for {instr_prefix} in bar {bar_features.get('bar_index','N/A')} is out of range [0, {self.max_subdivisions-1}]. Skipping token.")
            else: 
                tokens_str.append(f"[NO_{instr_prefix}_EVENTS]")
            tokens_str.append(f"[END_{instr_prefix}_EVENTS]") 
            
        return [self.token_to_id.get(t, self.unk_token_id) for t in tokens_str]

    def encode_flow_datum(self, flow_datum: FlowDatum, bar_bpm: float) -> List[int]:
        tokens_str: List[str] = ["[LINE_START]"]
        
        # This correctly caps the SYLLABLES_X token value.
        syllables_count_for_token = min(max(0, flow_datum["syllables"]), self.max_syllables)
        tokens_str.append(f"[SYLLABLES_{syllables_count_for_token}]")
        
        offset_normalized = flow_datum["start_offset_beats"] / self.flow_offset_max_beats
        offset_bin_idx = int(np.clip(offset_normalized * self.num_offset_bins, 0, self.num_offset_bins - 1))
        tokens_str.append(f"[OFFSET_BIN_{offset_bin_idx}]")
        
        duration_normalized = flow_datum["duration_beats"] / self.flow_duration_max_beats
        duration_bin_idx = int(np.clip(duration_normalized * self.num_duration_bins, 0, self.num_duration_bins - 1))
        tokens_str.append(f"[DURATION_BIN_{duration_bin_idx}]")

        syllable_starts = flow_datum.get("syllable_start_subdivisions", [])
        quantized_syllable_durations = flow_datum.get("syllable_durations_quantized", []) 
        syllable_stresses = flow_datum.get("syllable_stresses", []) 

        # --- Start of FIX ---
        # Determine the number of syllable events to actually encode,
        # capped by self.max_syllables.
        num_actual_syllable_events_in_datum = len(syllable_starts)
        num_syllable_events_to_encode = min(num_actual_syllable_events_in_datum, self.max_syllables)

        # For robustness, ensure we don't try to access beyond the shortest of the related lists.
        # This primarily protects against malformed FlowDatum, though FlowDataExtractor should be consistent.
        min_len_of_related_syl_data = num_actual_syllable_events_in_datum # Start with len(syllable_starts)
        if quantized_syllable_durations is not None:
            min_len_of_related_syl_data = min(min_len_of_related_syl_data, len(quantized_syllable_durations))
        if syllable_stresses is not None:
            min_len_of_related_syl_data = min(min_len_of_related_syl_data, len(syllable_stresses))
        
        num_syllable_events_to_encode = min(num_syllable_events_to_encode, min_len_of_related_syl_data)
        # --- End of FIX ---


        for i in range(num_syllable_events_to_encode): # Iterate up to the capped number
            sub_idx = syllable_starts[i]
            # Ensure dur_bin_idx and stress_val are safely accessed if lists could be short (already handled by min_len_of_related_syl_data)
            dur_bin_idx = quantized_syllable_durations[i] 
            stress_val = syllable_stresses[i]

            if 0 <= sub_idx < self.max_subdivisions:
                tokens_str.append(f"[SYLLABLE_STARTS_SUBDIV_{sub_idx}]")
            else: tokens_str.append(self.id_to_token[self.unk_token_id])
            
            if 0 <= dur_bin_idx < self.num_syllable_duration_bins:
                tokens_str.append(f"[SYLLABLE_DURATION_BIN_{dur_bin_idx}]")
            else: tokens_str.append(self.id_to_token[self.unk_token_id])

            if 0 <= stress_val < self.num_stress_levels:
                tokens_str.append(f"[SYLLABLE_STRESS_{stress_val}]")
            else: tokens_str.append(self.id_to_token[self.unk_token_id]) 

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
            
            seg_id_for_this_bar_features = bar_feature_idx * 2
            bar_feature_tokens = self.encode_bar_features(bar_features)
            full_token_ids.extend(bar_feature_tokens)
            segment_ids.extend([seg_id_for_this_bar_features] * len(bar_feature_tokens))
            intra_line_pos_ids.extend(list(range(len(bar_feature_tokens)))) 
            
            seg_id_for_this_bar_flow = seg_id_for_this_bar_features + 1
            full_token_ids.append(self.sep_input_flow_token_id)
            segment_ids.append(seg_id_for_this_bar_flow) 
            intra_line_pos_ids.append(0)

            bar_flow_lines = flow_by_bar.get(bar_idx, [])
            for line_idx, flow_datum in enumerate(bar_flow_lines):
                flow_line_tokens = self.encode_flow_datum(flow_datum, bar_bpm) 
                full_token_ids.extend(flow_line_tokens)
                segment_ids.extend([seg_id_for_this_bar_flow] * len(flow_line_tokens))
                intra_line_pos_ids.extend(list(range(len(flow_line_tokens))))
        
        eos_segment_id = segment_ids[-1] + 1 if segment_ids else 0 
        if full_token_ids == [self.bos_token_id]: eos_segment_id = 0 
        
        full_token_ids.append(self.eos_token_id)
        segment_ids.append(eos_segment_id) 
        intra_line_pos_ids.append(0) 
        
        return full_token_ids, segment_ids, intra_line_pos_ids

    def decode_flow_tokens_to_datum(self, flow_tokens: List[int], bar_idx_context: int, line_idx_context: int) -> Optional[FlowDatum]:
        tokens_str = [self.id_to_token.get(t_id, "[UNK]") for t_id in flow_tokens]
        
        ptr = 0
        if not tokens_str or tokens_str[ptr] != "[LINE_START]": return None 
        ptr += 1
        
        if len(tokens_str) - ptr < 3: return None # Need at least SYL, OFFSET, DUR tokens

        try:
            syll_token = tokens_str[ptr]; ptr +=1
            offset_token = tokens_str[ptr]; ptr +=1
            duration_token = tokens_str[ptr]; ptr +=1

            if not (syll_token.startswith("[SYLLABLES_") and \
                    offset_token.startswith("[OFFSET_BIN_") and \
                    duration_token.startswith("[DURATION_BIN_")):
                return None

            syllables_from_token = int(syll_token.split('_')[-1].replace(']', ''))
            offset_bin_idx = int(offset_token.split('_')[-1].replace(']', ''))
            start_offset_beats = (offset_bin_idx / self.num_offset_bins) * self.flow_offset_max_beats
            duration_bin_idx = int(duration_token.split('_')[-1].replace(']', ''))
            duration_beats = (duration_bin_idx / self.num_duration_bins) * self.flow_duration_max_beats

            syllable_start_subdivisions: List[int] = []
            syllable_durations_quantized: List[int] = [] 
            syllable_stresses: List[int] = []

            while ptr < len(tokens_str) and tokens_str[ptr].startswith("[SYLLABLE_STARTS_SUBDIV_"):
                subdiv_token_str = tokens_str[ptr]
                sub_idx = int(subdiv_token_str.split('_')[-1].replace(']', ''))
                syllable_start_subdivisions.append(sub_idx)
                ptr += 1
                
                if not (ptr < len(tokens_str) and tokens_str[ptr].startswith("[SYLLABLE_DURATION_BIN_")):
                    # print(f"DEBUG DECODE: Missing DURATION_BIN after STARTS_SUBDIV. Current token: {tokens_str[ptr] if ptr < len(tokens_str) else 'EOS'}")
                    return None 
                dur_bin_token_str = tokens_str[ptr]
                dur_bin_idx = int(dur_bin_token_str.split('_')[-1].replace(']', ''))
                syllable_durations_quantized.append(dur_bin_idx)
                ptr += 1

                if not (ptr < len(tokens_str) and tokens_str[ptr].startswith("[SYLLABLE_STRESS_")):
                    # print(f"DEBUG DECODE: Missing STRESS after DURATION_BIN. Current token: {tokens_str[ptr] if ptr < len(tokens_str) else 'EOS'}")
                    return None
                stress_token_str = tokens_str[ptr]
                stress_val = int(stress_token_str.split('_')[-1].replace(']', ''))
                syllable_stresses.append(stress_val)
                ptr += 1
                
            if not (ptr < len(tokens_str) and tokens_str[ptr] == "[END_SYLLABLE_SEQUENCE]"):
                # print(f"DEBUG DECODE: Missing END_SYLLABLE_SEQUENCE. Current token: {tokens_str[ptr] if ptr < len(tokens_str) else 'EOS'}")
                return None
            ptr += 1 

            # Check if there are any unexpected tokens after END_SYLLABLE_SEQUENCE
            if ptr != len(flow_tokens): # Or len(tokens_str)
                # print(f"DEBUG DECODE: Extra tokens found after END_SYLLABLE_SEQUENCE. Remaining: {tokens_str[ptr:]}")
                return None

            actual_decoded_syllables = len(syllable_start_subdivisions)
            # The SYLLABLES_X token is a hint; the actual number of syllable events decoded is what matters.
            # We can choose to use actual_decoded_syllables for the 'syllables' field in FlowDatum.
            # The original `syllables_from_token` might be useful for diagnostics or if the model
            # is expected to explicitly predict the count that matches the sequence length.

            # For now, let's use actual_decoded_syllables for consistency.
            # if syllables_from_token != actual_decoded_syllables:
            #    print(f"Warning: Decoded syllable count ({actual_decoded_syllables}) mismatch with token ({syllables_from_token}) "
            #          f"for bar {bar_idx_context}, line {line_idx_context}. Using actual decoded count.")


            return {
                "bar_index": bar_idx_context, 
                "line_index_in_bar": line_idx_context,
                "syllables": actual_decoded_syllables, # Use the count of decoded syllable events
                "start_offset_beats": round(start_offset_beats, 3),
                "duration_beats": round(duration_beats, 3),
                "syllable_start_subdivisions": syllable_start_subdivisions,
                "syllable_durations_quantized": syllable_durations_quantized,
                "syllable_stresses": syllable_stresses
            }
        except (ValueError, IndexError, AttributeError) as e:
            # print(f"DEBUG DECODE: Error during token parsing: {e}. Tokens: {tokens_str}")
            return None

    def get_vocab_size(self) -> int:
        return len(self.token_to_id)

    def save_vocab(self, filepath: str):
        id_to_token_serializable = {str(k): v for k, v in self.id_to_token.items()}
        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": id_to_token_serializable, 
            "config_params": {
                "max_syllables": self.max_syllables,
                "num_offset_bins": self.num_offset_bins,
                "num_duration_bins": self.num_duration_bins,
                "bpm_bins": self.bpm_bins,
                "max_subdivisions": self.max_subdivisions,
                "flow_offset_max_beats": self.flow_offset_max_beats,
                "flow_duration_max_beats": self.flow_duration_max_beats,
                "syllable_duration_bins_beats": self.syllable_duration_bins_beats, 
                "num_syllable_duration_bins": self.num_syllable_duration_bins,
                "num_stress_levels": self.num_stress_levels
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
        self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()} 
        
        config_params = vocab_data.get("config_params", {})
        self.max_syllables = config_params.get("max_syllables", self.max_syllables)
        self.num_offset_bins = config_params.get("num_offset_bins", self.num_offset_bins)
        self.num_duration_bins = config_params.get("num_duration_bins", self.num_duration_bins)
        self.bpm_bins = config_params.get("bpm_bins", self.bpm_bins)
        self.max_subdivisions = config_params.get("max_subdivisions", self.max_subdivisions)
        self.flow_offset_max_beats = config_params.get("flow_offset_max_beats", self.flow_offset_max_beats)
        self.flow_duration_max_beats = config_params.get("flow_duration_max_beats", self.flow_duration_max_beats)
        self.syllable_duration_bins_beats = config_params.get("syllable_duration_bins_beats", self.syllable_duration_bins_beats)
        self.num_syllable_duration_bins = config_params.get("num_syllable_duration_bins", len(self.syllable_duration_bins_beats) + 1)
        self.num_stress_levels = config_params.get("num_stress_levels", self.num_stress_levels)

        print(f"Vocabulary and config loaded from {filepath}. Initial loaded size: {len(self.token_to_id)}")

    @property
    def pad_token_id(self) -> int: return self.token_to_id["[PAD]"]
    @property
    def bos_token_id(self) -> int: return self.token_to_id["[BOS]"]
    @property
    def eos_token_id(self) -> int: return self.token_to_id["[EOS]"]
    @property
    def unk_token_id(self) -> int: return self.token_to_id.get("[UNK]", 1) 
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
    tokenizer.save_vocab(tokenizer_path) # This ensures extended vocab is saved
        
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")
    assert "[SYLLABLE_DURATION_BIN_0]" in tokenizer.token_to_id
    assert "[SYLLABLE_STRESS_0]" in tokenizer.token_to_id
    assert "[END_SYLLABLE_SEQUENCE]" in tokenizer.token_to_id
    print("Essential syllable duration and stress tokens are present.")

    # Test dequantize
    print("\nTesting dequantize_syllable_duration_bin:")
    for i in range(tokenizer.num_syllable_duration_bins):
        approx_beats = tokenizer.dequantize_syllable_duration_bin(i)
        print(f"  Bin {i} -> approx {approx_beats:.3f} beats")
    
    # Test with an out-of-range bin index
    print(f"  Bin OOR (e.g., 100) -> approx {tokenizer.dequantize_syllable_duration_bin(100):.3f} beats")


    sample_bar_features_1: BarBeatFeatures = {
        "bar_index": 0, "bpm": 125.0, "time_signature": (4, 4),
        "kick_events": [0, 4, 8, 12], "snare_events": [4, 12],
        "hihat_events": [i for i in range(0,16,2)], "bass_events": []
    }
    
    # Test FlowDatum with more syllables than tokenizer.max_syllables (24)
    # For this test, let's assume FlowDataExtractor produced this.
    num_test_syllables = 30 
    sample_flow_datum_long: FlowDatum = { 
        "bar_index": 0, "line_index_in_bar": 0, "syllables": num_test_syllables, 
        "start_offset_beats": 0.0, "duration_beats": 3.8, # (e.g. two bars long)
        "syllable_start_subdivisions": [i % tokenizer.max_subdivisions for i in range(num_test_syllables)], 
        "syllable_durations_quantized": [(i % tokenizer.num_syllable_duration_bins) for i in range(num_test_syllables)], 
        "syllable_stresses": [(i % tokenizer.num_stress_levels) for i in range(num_test_syllables)] 
    }
    
    print(f"\nEncoding FlowDatum with {num_test_syllables} syllables (tokenizer.max_syllables={tokenizer.max_syllables})...")
    example_flow_line_tokens_long = tokenizer.encode_flow_datum(sample_flow_datum_long, sample_bar_features_1["bpm"])
    print(f"Encoded flow line tokens (long): {example_flow_line_tokens_long}")
    # Count how many per-syllable token groups (start, dur, stress) were actually encoded
    syl_start_tokens_encoded = sum(1 for t_id in example_flow_line_tokens_long if tokenizer.id_to_token.get(t_id, "").startswith("[SYLLABLE_STARTS_SUBDIV_"))
    print(f"Number of SYLLABLE_STARTS_SUBDIV tokens encoded: {syl_start_tokens_encoded} (Expected to be {tokenizer.max_syllables})")
    assert syl_start_tokens_encoded == tokenizer.max_syllables, "Tokenizer did not cap syllable event tokens!"
    
    decoded_fd_long = tokenizer.decode_flow_tokens_to_datum(example_flow_line_tokens_long, bar_idx_context=0, line_idx_context=0)
    print(f"Decoded FlowDatum (long input): {decoded_fd_long}")
    assert decoded_fd_long is not None
    assert decoded_fd_long["syllables"] == tokenizer.max_syllables, "Decoded syllable count does not match tokenizer cap."


    # Original test case
    sample_flow_datum_b0_l0: FlowDatum = { 
        "bar_index": 0, "line_index_in_bar": 0, "syllables": 3, 
        "start_offset_beats": 0.0, "duration_beats": 1.9,
        "syllable_start_subdivisions": [0, 4, 8], 
        "syllable_durations_quantized": [1, 2, 1], 
        "syllable_stresses": [0, 1, 0] 
    }
    example_flow_line_tokens = tokenizer.encode_flow_datum(sample_flow_datum_b0_l0, sample_bar_features_1["bpm"])
    print(f"\nExample encoded flow line tokens (short): {example_flow_line_tokens}")
    # print(" ".join([tokenizer.id_to_token.get(tid, "UNK") for tid in example_flow_line_tokens]))
    
    decoded_fd = tokenizer.decode_flow_tokens_to_datum(example_flow_line_tokens, bar_idx_context=0, line_idx_context=0)
    print(f"Decoded FlowDatum (short): {decoded_fd}")
    assert decoded_fd is not None
    assert decoded_fd["syllables"] == 3
    assert decoded_fd["syllable_stresses"] == [0,1,0]


    print("\nTokenizer tests (including fix for syllable capping) passed.")