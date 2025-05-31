import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from beefai.utils.data_types import FlowDatum, FlowData, SongBeatFeatures, LyricsData, BarBeatFeatures, WordTiming
from beefai.data_processing.text_processor import TextProcessor
import os
import json
import librosa 

def parse_whisper_timestamped_json(json_file_path: str) -> Optional[LyricsData]:
    """
    Parses a JSON file output by whisper-timestamped to get word alignments.
    Filters out silence or short pause markers if present (though whisper-timestamped usually gives words).
    """
    if not os.path.exists(json_file_path):
        print(f"FlowDataExtractor: Whisper-timestamped JSON file not found: {json_file_path}")
        return None
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"FlowDataExtractor: Error reading whisper-timestamped JSON file {json_file_path}: {e}")
        return None

    word_alignments: LyricsData = []
    
    if "segments" not in data or not isinstance(data["segments"], list):
        print(f"FlowDataExtractor: JSON file {json_file_path} does not have the expected 'segments' list structure.")
        return None

    for segment in data["segments"]:
        if "words" not in segment or not isinstance(segment["words"], list):
            continue 

        for word_info in segment["words"]:
            if not all(k in word_info for k in ["text", "start", "end"]):
                continue

            word_text = word_info["text"].strip().lower()
            if word_text and word_text not in ["", "[Music]", "[ Silence ]", "(Music)", "(Silence)"]: 
                word_alignments.append({
                    "word": word_text,
                    "start_time": round(float(word_info["start"]), 4),
                    "end_time": round(float(word_info["end"]), 4)
                })
    
    if not word_alignments:
        print(f"FlowDataExtractor: No valid word alignments extracted from {json_file_path}.")
        return None
            
    return word_alignments


class FlowDataExtractor:
    def __init__(self, 
                 sample_rate_for_acapella: int = 44100, # Only if loading acapella audio directly
                 subdivisions_per_bar: int = 16 # For syllable landing pattern
                ): 
        self.sample_rate = sample_rate_for_acapella
        self.text_processor = TextProcessor() 
        self.subdivisions_per_bar = subdivisions_per_bar

    def _estimate_syllable_timings_for_word(self, word_data: WordTiming) -> List[Dict[str, Any]]:
        """
        Estimates start and end times for each syllable in a word.
        A simple approach: divides word duration equally among its syllables.
        More advanced: could use pyphen's syllable segmentation and distribute time by syllable length.
        """
        syllable_timings: List[Dict[str, Any]] = []
        word_text = word_data["word"]
        word_start_time = word_data["start_time"]
        word_end_time = word_data["end_time"]
        
        syllables_list = self.text_processor.get_syllables_from_word(word_text)
        num_syllables = len(syllables_list)

        if num_syllables == 0:
            return []

        word_duration = word_end_time - word_start_time
        if word_duration <= 0: # If word has no duration, assign all syllables to word_start_time
            for syl_text in syllables_list:
                syllable_timings.append({
                    "syllable_text": syl_text,
                    "start_time": word_start_time,
                    "end_time": word_start_time
                })
            return syllable_timings

        avg_syllable_duration = word_duration / num_syllables
        current_syllable_start_time = word_start_time

        for i, syl_text in enumerate(syllables_list):
            syl_start = current_syllable_start_time
            syl_end = current_syllable_start_time + avg_syllable_duration
            # Ensure last syllable ends exactly at word_end_time
            if i == num_syllables - 1:
                syl_end = word_end_time
            
            syllable_timings.append({
                "syllable_text": syl_text,
                "start_time": round(syl_start, 4),
                "end_time": round(syl_end, 4)
            })
            current_syllable_start_time = syl_end
            
        return syllable_timings

    def _segment_words_into_lines(self, 
                                  word_alignments: LyricsData,
                                  max_pause_sec_between_words_in_line: float = 0.35, 
                                  min_pause_sec_for_line_break: float = 0.5,       
                                  min_words_per_line: int = 2,
                                  max_words_per_line: int = 20 
                                 ) -> List[List[WordTiming]]: 
        if not word_alignments:
            return []

        lines_of_words: List[List[WordTiming]] = []
        current_line_words: List[WordTiming] = []
        
        for i, word_data in enumerate(word_alignments):
            if not current_line_words: 
                current_line_words.append(word_data)
            else:
                prev_word_end_time = current_line_words[-1]["end_time"]
                current_word_start_time = word_data["start_time"]
                pause_duration = current_word_start_time - prev_word_end_time

                if pause_duration >= min_pause_sec_for_line_break or \
                   len(current_line_words) >= max_words_per_line:
                    if len(current_line_words) >= min_words_per_line:
                        lines_of_words.append(list(current_line_words))
                    current_line_words = [word_data] 
                elif pause_duration < max_pause_sec_between_words_in_line:
                    current_line_words.append(word_data) 
                else: 
                    if len(current_line_words) >= min_words_per_line:
                        lines_of_words.append(list(current_line_words))
                    current_line_words = [word_data]
            
            if i == len(word_alignments) - 1:
                if current_line_words and (not lines_of_words or (lines_of_words and current_line_words != lines_of_words[-1])): 
                    if len(current_line_words) >= min_words_per_line or not lines_of_words : 
                        lines_of_words.append(list(current_line_words))
                    elif lines_of_words: 
                        lines_of_words[-1].extend(current_line_words)

        if current_line_words and (not lines_of_words or current_line_words != lines_of_words[-1]):
             if len(current_line_words) >= min_words_per_line :
                lines_of_words.append(list(current_line_words))
             elif lines_of_words: 
                lines_of_words[-1].extend(current_line_words)

        return lines_of_words

    def _create_flow_datum_from_line(self, 
                                     line_words: List[WordTiming], 
                                     bar_info_for_line: BarBeatFeatures, 
                                     all_song_beat_features: SongBeatFeatures, 
                                     line_idx_in_bar: int 
                                     ) -> Optional[FlowDatum]:
        if not line_words: return None
        
        line_actual_start_time_sec = line_words[0]["start_time"]
        line_actual_end_time_sec = line_words[-1]["end_time"]
        
        total_syllables_in_line = sum(self.text_processor.count_syllables_in_word(wd["word"]) for wd in line_words)
        if total_syllables_in_line == 0: return None 

        current_bar_abs_start_time_sec = 0.0
        found_bar_start = False
        _bar_starts_map = {} 

        if not _bar_starts_map: 
            temp_bar_time = 0.0
            for bf_idx, bf in enumerate(all_song_beat_features):
                _bar_starts_map[bf["bar_index"]] = temp_bar_time
                if bf["bpm"] > 0:
                    beat_dur = 60.0 / bf["bpm"]
                    bar_dur = bf["time_signature"][0] * beat_dur
                    temp_bar_time += bar_dur
                else: 
                    temp_bar_time += 2.0

        target_bar_index = bar_info_for_line["bar_index"]
        if target_bar_index in _bar_starts_map:
            current_bar_abs_start_time_sec = _bar_starts_map[target_bar_index]
            found_bar_start = True
        
        if not found_bar_start:
            print(f"Error: Could not determine absolute start time for bar_index {target_bar_index}. Skipping line.")
            return None

        bpm_of_bar = bar_info_for_line["bpm"]
        if bpm_of_bar <= 0: 
            print(f"Warning: Bar {target_bar_index} has BPM <=0. Cannot calculate beat-relative timings. Skipping.")
            return None
        
        beats_in_current_bar = bar_info_for_line["time_signature"][0]
        beat_duration_sec = 60.0 / bpm_of_bar
        bar_duration_sec = beats_in_current_bar * beat_duration_sec
        if bar_duration_sec <= 0.01: # Avoid division by zero if bar duration is negligible
            print(f"Warning: Bar {target_bar_index} has near-zero duration. Skipping line.")
            return None
        subdivision_duration_sec = bar_duration_sec / self.subdivisions_per_bar


        start_offset_beats = (line_actual_start_time_sec - current_bar_abs_start_time_sec) / beat_duration_sec
        duration_beats = (line_actual_end_time_sec - line_actual_start_time_sec) / beat_duration_sec
        
        start_offset_beats = np.clip(start_offset_beats, -1.0, beats_in_current_bar + 1.0) 
        duration_beats = max(0.1, duration_beats) 

        # Calculate syllable landing pattern
        syllable_start_subdivisions: List[int] = []
        for word_detail in line_words:
            syllables_in_word = self._estimate_syllable_timings_for_word(word_detail)
            for syl_timing in syllables_in_word:
                syl_abs_start_time = syl_timing["start_time"]
                # Time of syllable relative to the start of its assigned bar
                syl_time_in_bar = syl_abs_start_time - current_bar_abs_start_time_sec
                
                # Quantize to subdivision index
                # Allow syllables to land slightly before bar start (pickup) or slightly after bar end
                # by not strictly clipping syl_time_in_bar to [0, bar_duration_sec] before normalization.
                # The subdivision_index will be clipped later.
                normalized_time_in_bar = syl_time_in_bar / bar_duration_sec
                subdivision_index = int(normalized_time_in_bar * self.subdivisions_per_bar)
                
                # Clip to valid range [0, self.subdivisions_per_bar - 1]
                subdivision_index = max(0, min(subdivision_index, self.subdivisions_per_bar - 1))
                syllable_start_subdivisions.append(subdivision_index)
        
        # Remove duplicates and sort, as multiple syllables might land in the same subdivision
        # or if we want to preserve all landings, don't make it a set.
        # For now, let's keep all, as it represents a sequence of onsets.
        # If the tokenizer expects unique positions, this needs adjustment.
        # The current plan for tokenizer is a sequence of [SYLLABLE_STARTS_SUBDIV_X] tokens,
        # so duplicates are fine and represent multiple syllables hitting that subdivision.
        # Sorting might be good for canonical representation if order doesn't matter for the token list.
        # For now, keep chronological order.

        return {
            "bar_index": target_bar_index, 
            "line_index_in_bar": line_idx_in_bar, 
            "syllables": total_syllables_in_line,
            "start_offset_beats": round(start_offset_beats, 3),
            "duration_beats": round(duration_beats, 3),
            "syllable_start_subdivisions": syllable_start_subdivisions 
        }

    def extract_flow_for_song(self, 
                              alignment_data_path: str, 
                              song_beat_features: SongBeatFeatures 
                             ) -> Optional[FlowData]:
        word_alignments = parse_whisper_timestamped_json(alignment_data_path)
        if not word_alignments:
            print(f"FlowDataExtractor: Failed to get word alignments from {alignment_data_path}.")
            return None
        if not song_beat_features:
            print(f"FlowDataExtractor: Missing song_beat_features. Cannot process flow.")
            return None

        lines_of_words = self._segment_words_into_lines(word_alignments)
        if not lines_of_words:
            print("FlowDataExtractor: No lines segmented from word alignments.")
            return None

        all_flow_data: FlowData = []
        
        bar_absolute_start_times: Dict[int, float] = {}
        current_abs_time = 0.0
        for bar_feat in song_beat_features:
            bar_absolute_start_times[bar_feat["bar_index"]] = current_abs_time
            if bar_feat["bpm"] > 0:
                beat_dur = 60.0 / bar_feat["bpm"]
                bar_duration_sec = bar_feat["time_signature"][0] * beat_dur
                current_abs_time += bar_duration_sec
            else: 
                current_abs_time += 2.0

        bar_line_counters: Dict[int, int] = {bf["bar_index"]: 0 for bf in song_beat_features}

        for line_ws in lines_of_words:
            if not line_ws: continue
            line_actual_start_time = line_ws[0]["start_time"]
            
            assigned_bar_idx = -1
            min_positive_offset_to_bar_start = float('inf')
            closest_bar_for_pickup = -1
            
            for bf in song_beat_features:
                bar_idx = bf["bar_index"]
                bar_start_t = bar_absolute_start_times.get(bar_idx, -1)
                if bar_start_t < 0: continue 

                offset_to_bar = line_actual_start_time - bar_start_t
                
                if offset_to_bar >= 0: 
                    if offset_to_bar < min_positive_offset_to_bar_start:
                        min_positive_offset_to_bar_start = offset_to_bar
                        assigned_bar_idx = bar_idx
                elif closest_bar_for_pickup == -1 or abs(offset_to_bar) < abs(line_actual_start_time - bar_absolute_start_times.get(closest_bar_for_pickup, float('inf'))):
                     closest_bar_for_pickup = bar_idx

            if assigned_bar_idx == -1: 
                if closest_bar_for_pickup != -1: 
                    assigned_bar_idx = closest_bar_for_pickup
                elif song_beat_features : 
                    assigned_bar_idx = song_beat_features[0]["bar_index"]
                else: 
                    print(f"FlowDataExtractor: Cannot assign bar for line starting at {line_actual_start_time:.2f}s (no bar info). Skipping.")
                    continue
            
            bar_info_for_this_line = next((bf for bf in song_beat_features if bf["bar_index"] == assigned_bar_idx), None)
            if not bar_info_for_this_line:
                print(f"FlowDataExtractor: Could not retrieve bar_info for assigned_bar_idx {assigned_bar_idx}. Skipping.")
                continue
                
            line_idx_in_assigned_bar = bar_line_counters.get(assigned_bar_idx, 0)
            flow_datum = self._create_flow_datum_from_line(line_ws, bar_info_for_this_line, song_beat_features, line_idx_in_assigned_bar)
            
            if flow_datum:
                all_flow_data.append(flow_datum)
                bar_line_counters[assigned_bar_idx] = line_idx_in_assigned_bar + 1
        
        print(f"FlowDataExtractor: Extracted {len(all_flow_data)} flow segments (lines) from {alignment_data_path}.")
        return all_flow_data

if __name__ == '__main__':
    dummy_wt_content = {
        "text": "yo this is a test flow line another one right here",
        "segments": [
            {
                "words": [
                    {"word": "yo", "start_time": 0.5, "end_time": 0.9, "text": "yo"}, # Added text for WordTiming
                    {"word": "this", "start_time": 1.0, "end_time": 1.4, "text": "this"},
                    {"word": "is", "start_time": 1.5, "end_time": 1.8, "text": "is"}
                ]
            },
            {
                "words": [
                    {"word": "a", "start_time": 2.2, "end_time": 2.6, "text": "a"},
                    {"word": "test", "start_time": 2.7, "end_time": 3.3, "text": "test"}
                ]
            },
            {
                "words": [
                    {"word": "flow", "start_time": 4.1, "end_time": 4.8, "text": "flow"},
                    {"word": "line", "start_time": 5.0, "end_time": 5.5, "text": "line"},
                    {"word": "another", "start_time": 5.6, "end_time": 6.2, "text": "another"},
                    {"word": "one", "start_time": 6.3, "end_time": 6.6, "text": "one"},
                    {"word": "right", "start_time": 6.7, "end_time": 7.0, "text": "right"},
                    {"word": "here", "start_time": 7.1, "end_time": 7.5, "text": "here"}
                ]
            }
        ]
    }
    # Corrected dummy_wt_content to match LyricsData/WordTiming (text field)
    # and whisper-timestamped output format (word dicts directly in segment["words"])
    for seg in dummy_wt_content["segments"]:
        new_words = []
        for w_dict in seg["words"]:
             # Ensure the dummy data matches the structure parse_whisper_timestamped_json expects
            new_words.append({
                "text": w_dict["word"], # whisper-timestamped uses "text"
                "start": w_dict["start_time"],
                "end": w_dict["end_time"],
                "confidence": 0.9 # dummy confidence
            })
        seg["words"] = new_words


    dummy_json_path = "dummy_alignment_flow_test_v2.json"
    if not os.path.exists(dummy_json_path):
        with open(dummy_json_path, "w") as f:
            json.dump(dummy_wt_content, f, indent=2)
        print(f"Created {dummy_json_path} for testing.")

    dummy_sbf: SongBeatFeatures = [
        {"bar_index": 0, "bpm": 120.0, "time_signature": (4, 4), "kick_events": [0, 8], "snare_events": [4, 12], "hihat_events": list(range(0,16,2)), "bass_events": [0]}, # Bar duration = 2s
        {"bar_index": 1, "bpm": 120.0, "time_signature": (4, 4), "kick_events": [0, 8], "snare_events": [4, 12], "hihat_events": list(range(0,16,4)), "bass_events": [0,2,4,6]}, # Bar duration = 2s
        {"bar_index": 2, "bpm": 120.0, "time_signature": (4, 4), "kick_events": [0,6,10], "snare_events": [4], "hihat_events": [], "bass_events": [8]}, # Bar duration = 2s
        {"bar_index": 3, "bpm": 120.0, "time_signature": (4, 4), "kick_events": [], "snare_events": [], "hihat_events": [], "bass_events": []} # Bar duration = 2s
    ]

    # Test with subdivisions_per_bar = 16 (default)
    flow_extractor = FlowDataExtractor(subdivisions_per_bar=16)
    print(f"\nExtracting flow from: {dummy_json_path} with {flow_extractor.subdivisions_per_bar} subdivisions/bar")
    extracted_flow_data = flow_extractor.extract_flow_for_song(dummy_json_path, dummy_sbf)
    
    if extracted_flow_data:
        print("\nExtracted Flow Data (Test):")
        for i, fd in enumerate(extracted_flow_data): 
            print(f"  Segment {i}:")
            for key, val in fd.items():
                print(f"    {key}: {val}")
    else:
        print("No flow data extracted.")

    # Example: First line "yo this is" (0.5s to 1.8s) in bar 0 (starts 0s, duration 2s).
    # Word "yo" (0.5-0.9), 1 syllable. syl_time_in_bar = 0.5. norm = 0.5/2 = 0.25. subdiv_idx = 0.25*16 = 4.
    # Word "this" (1.0-1.4), 1 syllable. syl_time_in_bar = 1.0. norm = 1.0/2 = 0.5. subdiv_idx = 0.5*16 = 8.
    # Word "is" (1.5-1.8), 1 syllable. syl_time_in_bar = 1.5. norm = 1.5/2 = 0.75. subdiv_idx = 0.75*16 = 12.
    # Expected for first line: syllable_start_subdivisions: [4, 8, 12]