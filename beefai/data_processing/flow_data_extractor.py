import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from beefai.utils.data_types import FlowDatum, FlowData, SongBeatFeatures, LyricsData
from beefai.data_processing.text_processor import TextProcessor
import textgrid # Library for reading Praat TextGrid files (pip install praat-textgrids)
import os
import librosa # For loading acapella if needed for advanced pause detection (not primary here)

def parse_mfa_textgrid(textgrid_path: str) -> Optional[LyricsData]:
    """
    Parses a TextGrid file (standard output from Montreal Forced Aligner) to get word alignments.
    Assumes a tier named "words" contains word intervals with timestamps and text.
    Filters out silence or short pause markers.
    """
    if not os.path.exists(textgrid_path):
        print(f"FlowDataExtractor: TextGrid file not found: {textgrid_path}")
        return None
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
    except Exception as e:
        print(f"FlowDataExtractor: Error reading TextGrid file {textgrid_path}: {e}")
        return None

    word_alignments: LyricsData = []
    
    # MFA typically puts words in a tier named 'words'.
    # Some aligners might use the audio filename as the tier name.
    word_tier_name = None
    potential_tier_names = ["words", "word", os.path.splitext(os.path.basename(textgrid_path))[0]]
    
    for name_candidate in potential_tier_names:
        if name_candidate in tg.tierNameSet:
            tier = tg.getFirst(name_candidate)
            if isinstance(tier, textgrid.IntervalTier) and any(interval.mark for interval in tier): # Check if it has content
                word_tier_name = name_candidate
                break
    
    if not word_tier_name: # Fallback: try to find any interval tier with word-like content
        for tier in tg:
            if isinstance(tier, textgrid.IntervalTier):
                # Check if a few intervals look like words (not just 'sil' or 'sp')
                non_sil_intervals = [interval for interval in tier if interval.mark and interval.mark.lower() not in ["sil", "sp", "<s>", "</s>", "<eps>"]]
                if len(non_sil_intervals) > 0: # If at least one looks like a word
                    word_tier_name = tier.name
                    print(f"FlowDataExtractor: Using tier '{word_tier_name}' as it contains word-like intervals.")
                    break
    
    if not word_tier_name:
        print(f"FlowDataExtractor: Could not find a suitable 'words' tier in {textgrid_path}. Available tiers: {tg.tierNameSet}. Searched for {potential_tier_names}.")
        return None

    words_tier = tg.getFirst(word_tier_name)
    for interval in words_tier:
        # Filter out typical silence/pause markers from MFA/speech processing
        word_text = interval.mark.strip().lower()
        if word_text and word_text not in ["", "sp", "sil", "spn", "<eps>", "<s>", "</s>"]: # Common markers
            word_alignments.append({
                "word": word_text,
                "start_time": round(interval.minTime, 4), # Keep precision
                "end_time": round(interval.maxTime, 4)
            })
    
    if not word_alignments:
        print(f"FlowDataExtractor: No valid word alignments found in tier '{word_tier_name}' of {textgrid_path}.")
        return None
        
    return word_alignments


class FlowDataExtractor:
    def __init__(self, sample_rate_for_acapella: int = 44100): # Only if loading acapella audio directly
        self.sample_rate = sample_rate_for_acapella
        self.text_processor = TextProcessor() 

    def _segment_words_into_lines(self, 
                                  word_alignments: LyricsData,
                                  max_pause_sec_between_words_in_line: float = 0.35, # Pause to keep words in same line
                                  min_pause_sec_for_line_break: float = 0.5,       # Pause to definitely break line
                                  min_words_per_line: int = 2,
                                  max_words_per_line: int = 20 # Generous max
                                 ) -> List[List[Dict[str, Any]]]: # List of lines, where each line is list of word dicts
        """
        Segments a flat list of timed words into lines based on pauses and word counts.
        More sophisticated methods could use linguistic cues or bar boundaries.
        """
        if not word_alignments:
            return []

        lines_of_words: List[List[Dict[str, Any]]] = []
        current_line_words: List[Dict[str, Any]] = []
        
        for i, word_data in enumerate(word_alignments):
            if not current_line_words: # First word of a new line
                current_line_words.append(word_data)
            else:
                prev_word_end_time = current_line_words[-1]["end_time"]
                current_word_start_time = word_data["start_time"]
                pause_duration = current_word_start_time - prev_word_end_time

                # Conditions to break a line:
                # 1. Pause is too long.
                # 2. Current line exceeds max words.
                if pause_duration >= min_pause_sec_for_line_break or \
                   len(current_line_words) >= max_words_per_line:
                    if len(current_line_words) >= min_words_per_line:
                        lines_of_words.append(list(current_line_words))
                    current_line_words = [word_data] # Start new line with current word
                elif pause_duration < max_pause_sec_between_words_in_line:
                    current_line_words.append(word_data) # Add to current line
                else: # Pause is between the two thresholds - ambiguous, start new line
                    if len(current_line_words) >= min_words_per_line:
                        lines_of_words.append(list(current_line_words))
                    current_line_words = [word_data]
            
            # If it's the very last word in the entire alignment
            if i == len(word_alignments) - 1:
                if current_line_words and (lines_of_words and current_line_words != lines_of_words[-1]): # Add if not already added
                    if len(current_line_words) >= min_words_per_line or not lines_of_words : # Add if long enough or it's the only line
                        lines_of_words.append(list(current_line_words))
                    elif lines_of_words: # Append to previous line if too short and prev line exists
                        lines_of_words[-1].extend(current_line_words)


        # Final check for any remaining words (should be caught by last word logic)
        if current_line_words and (not lines_of_words or current_line_words != lines_of_words[-1]):
             if len(current_line_words) >= min_words_per_line :
                lines_of_words.append(list(current_line_words))
             elif lines_of_words: # Append to previous if short
                lines_of_words[-1].extend(current_line_words)

        return lines_of_words

    def _create_flow_datum_from_line(self, 
                                     line_words: List[Dict[str, Any]], 
                                     bar_info_for_line: BarBeatFeatures, # Bar this line is primarily in
                                     all_song_beat_features: SongBeatFeatures, # For calculating bar start times
                                     line_idx_in_bar: int # Relative to the assigned bar
                                     ) -> Optional[FlowDatum]:
        if not line_words: return None
        
        line_actual_start_time_sec = line_words[0]["start_time"]
        line_actual_end_time_sec = line_words[-1]["end_time"]
        
        syllables_count = sum(self.text_processor.count_syllables_in_word(wd["word"]) for wd in line_words)
        if syllables_count == 0: return None # Skip lines with no syllables

        # Determine the absolute start time of the bar this line belongs to
        current_bar_abs_start_time_sec = 0.0
        found_bar_start = False
        _bar_starts_map = {} # cache bar start times for performance

        if not _bar_starts_map: # Compute once if not already done
            temp_bar_time = 0.0
            for bf_idx, bf in enumerate(all_song_beat_features):
                _bar_starts_map[bf["bar_index"]] = temp_bar_time
                if bf["bpm"] > 0:
                    beat_dur = 60.0 / bf["bpm"]
                    bar_dur = bf["time_signature"][0] * beat_dur
                    temp_bar_time += bar_dur
                else: # Should not happen with valid beat features
                    print(f"Warning: Bar {bf['bar_index']} has BPM <= 0. Assuming 2s duration.")
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
        
        beat_duration_sec = 60.0 / bpm_of_bar

        # Calculate offset and duration in beats
        start_offset_beats = (line_actual_start_time_sec - current_bar_abs_start_time_sec) / beat_duration_sec
        duration_beats = (line_actual_end_time_sec - line_actual_start_time_sec) / beat_duration_sec
        
        # Clamp start_offset_beats: can be slightly negative (pickup) or extend beyond bar for syncopation
        # For simplicity now, let's clip to be within reasonable bounds relative to bar structure e.g. -1 to beats_per_bar+1
        beats_in_current_bar = bar_info_for_line["time_signature"][0]
        start_offset_beats = np.clip(start_offset_beats, -1.0, beats_in_current_bar + 1.0) 
        duration_beats = max(0.1, duration_beats) # Ensure minimum duration

        return {
            "bar_index": target_bar_index, 
            "line_index_in_bar": line_idx_in_bar, 
            "syllables": syllables_count,
            "start_offset_beats": round(start_offset_beats, 3),
            "duration_beats": round(duration_beats, 3)
            # Store actual times for reference/debugging if needed:
            # "actual_start_time_sec": line_actual_start_time_sec,
            # "actual_end_time_sec": line_actual_end_time_sec
        }

    def extract_flow_for_song(self, 
                              alignment_data_path: str, # Path to TextGrid file from MFA
                              song_beat_features: SongBeatFeatures # From BeatFeatureExtractor
                             ) -> Optional[FlowData]:
        """
        Extracts a sequence of FlowDatum for an entire song.
        """
        word_alignments = parse_mfa_textgrid(alignment_data_path)
        if not word_alignments:
            print(f"FlowDataExtractor: Failed to get word alignments from {alignment_data_path}.")
            return None
        if not song_beat_features:
            print(f"FlowDataExtractor: Missing song_beat_features. Cannot process flow.")
            return None

        # Segment all words from the acapella into lines first
        lines_of_words = self._segment_words_into_lines(word_alignments)
        if not lines_of_words:
            print("FlowDataExtractor: No lines segmented from word alignments.")
            return None

        all_flow_data: FlowData = []
        
        # Pre-calculate absolute start times for each bar for efficient lookup
        bar_absolute_start_times: Dict[int, float] = {}
        current_abs_time = 0.0
        for bar_feat in song_beat_features:
            bar_absolute_start_times[bar_feat["bar_index"]] = current_abs_time
            if bar_feat["bpm"] > 0:
                beat_dur = 60.0 / bar_feat["bpm"]
                bar_duration_sec = bar_feat["time_signature"][0] * beat_dur
                current_abs_time += bar_duration_sec
            else: # Should be handled by BFE, but as a fallback
                print(f"Warning: Bar {bar_feat['bar_index']} in beat features has invalid BPM. Assuming 2s duration.")
                current_abs_time += 2.0


        # Assign lines to bars and create FlowDatum
        # This logic needs to be robust: a line might span bars or be a pickup.
        # Heuristic: assign line to the bar in which it *starts*.
        bar_line_counters: Dict[int, int] = {bar_idx: 0 for bar_idx in range(len(song_beat_features))}

        for line_ws in lines_of_words:
            if not line_ws: continue
            line_actual_start_time = line_ws[0]["start_time"]
            
            # Find the bar this line most likely starts in or just before (pickup)
            assigned_bar_idx = -1
            min_positive_offset_to_bar_start = float('inf')
            closest_bar_for_pickup = -1
            
            for bf in song_beat_features:
                bar_idx = bf["bar_index"]
                bar_start_t = bar_absolute_start_times.get(bar_idx, -1)
                if bar_start_t < 0: continue # Should not happen

                offset_to_bar = line_actual_start_time - bar_start_t
                
                if offset_to_bar >= 0: # Line starts in or after this bar start
                    if offset_to_bar < min_positive_offset_to_bar_start:
                        min_positive_offset_to_bar_start = offset_to_bar
                        assigned_bar_idx = bar_idx
                elif closest_bar_for_pickup == -1 or abs(offset_to_bar) < abs(line_actual_start_time - bar_absolute_start_times.get(closest_bar_for_pickup, float('inf'))):
                    # If line starts before this bar, consider it a potential pickup for this bar
                     closest_bar_for_pickup = bar_idx


            if assigned_bar_idx == -1: # Line might be a pickup to the first bar or fall before any detected bar
                if closest_bar_for_pickup != -1: # Likely a pickup for 'closest_bar_for_pickup'
                    assigned_bar_idx = closest_bar_for_pickup
                elif song_beat_features : # Default to first bar if no other assignment
                    assigned_bar_idx = song_beat_features[0]["bar_index"]
                else: # No bars to assign to
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
    # To test, you would need:
    # 1. A dummy TextGrid file (e.g., "dummy_alignment.TextGrid")
    #    Content example for TextGrid (simplified for testing):
    dummy_tg_content = """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0 
xmax = 10
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "words" 
        xmin = 0 
        xmax = 10 
        intervals: size = 7
        intervals [1]:
            xmin = 0.5 
            xmax = 0.9 
            text = "yo" 
        intervals [2]:
            xmin = 1.0 
            xmax = 1.4
            text = "this" 
        intervals [3]:
            xmin = 1.5
            xmax = 1.8
            text = "is" 
        intervals [4]:
            xmin = 2.2  
            xmax = 2.6 
            text = "a" 
        intervals [5]:
            xmin = 2.7 
            xmax = 3.3
            text = "test"
        intervals [6]:
            xmin = 4.1
            xmax = 4.8
            text = "flow"
        intervals [7]:
            xmin = 5.0
            xmax = 5.5
            text = "line"
"""
    dummy_tg_path = "dummy_alignment_flow_test.TextGrid"
    if not os.path.exists(dummy_tg_path):
        with open(dummy_tg_path, "w") as f:
            f.write(dummy_tg_content)
        print(f"Created {dummy_tg_path} for testing.")

    # 2. Dummy SongBeatFeatures (output from BeatFeatureExtractor)
    # Bar 0: 0-2s (120BPM, 4/4), Bar 1: 2-4s (120BPM, 4/4), Bar 2: 4-6s (120BPM, 4/4)
    dummy_sbf: SongBeatFeatures = [
        {"bar_index": 0, "bpm": 120.0, "time_signature": (4, 4), "kick_events": [0, 8], "snare_events": [4, 12], "hihat_events": list(range(0,16,2)), "bass_events": [0]},
        {"bar_index": 1, "bpm": 120.0, "time_signature": (4, 4), "kick_events": [0, 8], "snare_events": [4, 12], "hihat_events": list(range(0,16,4)), "bass_events": [0,2,4,6]},
        {"bar_index": 2, "bpm": 120.0, "time_signature": (4, 4), "kick_events": [0,6,10], "snare_events": [4], "hihat_events": [], "bass_events": [8]}
    ]

    flow_extractor = FlowDataExtractor()
    print(f"\nExtracting flow from: {dummy_tg_path}")
    extracted_flow_data = flow_extractor.extract_flow_for_song(dummy_tg_path, dummy_sbf)
    
    if extracted_flow_data:
        print("\nExtracted Flow Data (Test):")
        for i, fd in enumerate(extracted_flow_data): 
            print(f"  Segment {i}: {fd}")
    else:
        print("No flow data extracted.")