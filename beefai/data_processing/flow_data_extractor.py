# beefai/data_processing/flow_data_extractor.py

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from beefai.utils.data_types import FlowDatum, FlowData, SongBeatFeatures, LyricsData, BarBeatFeatures, WordTiming, SyllableDetail
from beefai.data_processing.text_processor import TextProcessor
from beefai.flow_model.tokenizer import FlowTokenizer
import os
import json

DEBUG_FLOW_EXTRACTOR = False
LOG_PREFIX_FDE = "[FDE]"


def parse_whisper_timestamped_json(json_file_path: str) -> Optional[LyricsData]:
    if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: parse_whisper: Attempting to parse {json_file_path}", flush=True)
    if not os.path.exists(json_file_path):
        if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: parse_whisper: File not found: {json_file_path}", flush=True)
        return None
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: parse_whisper: Error reading JSON {json_file_path}: {e}", flush=True)
        return None

    word_alignments: LyricsData = []
    if "segments" not in data or not isinstance(data["segments"], list):
        if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: parse_whisper: JSON {json_file_path} lacks 'segments' list.", flush=True)
        return None

    total_words_in_file = 0
    parsed_words_count = 0
    for segment_idx, segment in enumerate(data["segments"]):
        if "words" not in segment or not isinstance(segment["words"], list):
            if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: parse_whisper: Segment {segment_idx} in {json_file_path} lacks 'words' list.", flush=True)
            continue 
        total_words_in_file += len(segment["words"])
        for word_idx, word_info in enumerate(segment["words"]):
            if not isinstance(word_info, dict) or not all(k in word_info for k in ["text", "start", "end"]):
                if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: parse_whisper: Segment {segment_idx}, Word {word_idx} in {json_file_path} is malformed: {word_info}", flush=True)
                continue
            word_text = str(word_info["text"]).strip().lower() 
            ignore_texts = ["", "[music]", "[silence]", "(music)", "(silence)", "[ موسیقی ]", "[ سکوت ]", "[موسيقى]", "[صمت]"]
            if word_text and word_text not in ignore_texts: 
                try:
                    start_time = float(word_info["start"])
                    end_time = float(word_info["end"])
                    if start_time > end_time + 1e-3: 
                         if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: parse_whisper: Segment {segment_idx}, Word {word_idx} has start_time > end_time: {word_info}", flush=True)
                         continue
                    end_time = max(start_time, end_time) 
                    word_alignments.append({
                        "word": word_text,
                        "start_time": round(start_time, 4),
                        "end_time": round(end_time, 4)
                    })
                    parsed_words_count +=1
                except (ValueError, TypeError) as e:
                    if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: parse_whisper: Segment {segment_idx}, Word {word_idx} has invalid time: {word_info}, error: {e}", flush=True)
                    continue
    
    if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: parse_whisper: From {json_file_path}, total words in JSON: {total_words_in_file}, successfully parsed and kept: {parsed_words_count}", flush=True)
    return word_alignments if word_alignments else None


class FlowDataExtractor:
    def __init__(self, 
                 tokenizer: FlowTokenizer,
                 sample_rate_for_acapella: int = 44100, 
                 subdivisions_per_bar: int = 16 
                ): 
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate_for_acapella 
        self.text_processor = TextProcessor() 
        self.subdivisions_per_bar = subdivisions_per_bar
        # if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG: FlowDataExtractor initialized. Subdivisions per bar: {self.subdivisions_per_bar}", flush=True)

    def _estimate_syllable_details_for_word(self, word_data: WordTiming) -> List[SyllableDetail]:
        """
        Estimates start times, end times, and stress for each syllable within a word.
        Returns: List of SyllableDetail dicts.
        """
        syllable_details_list: List[SyllableDetail] = []
        word_text = word_data["word"]
        word_start_time = word_data["start_time"]
        word_end_time = word_data["end_time"]
        
        syllables_with_stress_info = self.text_processor.get_syllables_with_stress(word_text)
        num_syllables = len(syllables_with_stress_info)

        if num_syllables == 0: return [] 

        word_duration = word_end_time - word_start_time
        if word_duration < 0: word_duration = 0 

        current_syllable_start_time = word_start_time
        min_syl_dur = 0.001 

        if word_duration == 0:
            for syl_text, stress_val in syllables_with_stress_info:
                syllable_details_list.append({
                    "syllable_text": syl_text, 
                    "start_time": word_start_time, 
                    "end_time": word_end_time,
                    "stress": stress_val
                })
            return syllable_details_list

        avg_syllable_duration = word_duration / num_syllables

        for i, (syl_text, stress_val) in enumerate(syllables_with_stress_info):
            syl_start = current_syllable_start_time
            
            # Calculate ideal end time for this syllable
            ideal_syl_end = current_syllable_start_time + avg_syllable_duration
            
            # For the last syllable, ensure it ends exactly at word_end_time
            if i == num_syllables - 1: 
                syl_end = word_end_time
            else:
                syl_end = ideal_syl_end
            
            # Ensure syllable duration is at least min_syl_dur
            syl_end = max(syl_start + min_syl_dur, syl_end)
            
            # Ensure syllable does not extend beyond word_end_time
            syl_end = min(syl_end, word_end_time)

            syllable_details_list.append({
                "syllable_text": syl_text, 
                "start_time": round(syl_start, 4), 
                "end_time": round(syl_end, 4),
                "stress": stress_val
            })
            current_syllable_start_time = syl_end
        return syllable_details_list

    def _segment_words_into_phrases(self, 
                                  word_alignments: LyricsData,
                                  max_pause_sec_between_words_in_phrase: float = 0.35, 
                                  min_pause_sec_for_phrase_break: float = 0.5,       
                                  min_words_per_phrase: int = 1, 
                                  max_words_per_phrase: int = 30 
                                 ) -> List[List[WordTiming]]: 
        if not word_alignments: return []
        phrases_of_words: List[List[WordTiming]] = []
        current_phrase_words: List[WordTiming] = []
        
        for i, word_data in enumerate(word_alignments):
            if not current_phrase_words: 
                current_phrase_words.append(word_data)
            else:
                prev_word_end_time = current_phrase_words[-1]["end_time"]
                current_word_start_time = word_data["start_time"]
                pause_duration = current_word_start_time - prev_word_end_time

                if pause_duration >= min_pause_sec_for_phrase_break or \
                   len(current_phrase_words) >= max_words_per_phrase:
                    if len(current_phrase_words) >= min_words_per_phrase:
                        phrases_of_words.append(list(current_phrase_words))
                    current_phrase_words = [word_data] 
                elif pause_duration < max_pause_sec_between_words_in_phrase: 
                    current_phrase_words.append(word_data) 
                else: 
                    if len(current_phrase_words) >= min_words_per_phrase:
                        phrases_of_words.append(list(current_phrase_words))
                    current_phrase_words = [word_data]
            
        if current_phrase_words:
            if len(current_phrase_words) >= min_words_per_phrase:
                phrases_of_words.append(list(current_phrase_words))
        
        if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG _segment_words_into_phrases: Segmented into {len(phrases_of_words)} phrases.", flush=True)
        return phrases_of_words

    def _create_flow_datum_for_bar_segment(self, 
                                     words_in_bar_segment: List[WordTiming], 
                                     target_bar_info: BarBeatFeatures, 
                                     target_bar_abs_start_time_sec: float, 
                                     line_idx_in_bar: int 
                                     ) -> Optional[FlowDatum]:
        if not words_in_bar_segment: return None
        
        all_syllables_details_for_line: List[SyllableDetail] = []
        for word_detail in words_in_bar_segment:
            all_syllables_details_for_line.extend(self._estimate_syllable_details_for_word(word_detail))

        if not all_syllables_details_for_line: return None
        total_syllables_in_segment = len(all_syllables_details_for_line)

        segment_actual_start_time_sec = all_syllables_details_for_line[0]["start_time"]
        segment_actual_end_time_sec = all_syllables_details_for_line[-1]["end_time"]
        
        target_bar_index = target_bar_info["bar_index"]
        bpm_of_bar = target_bar_info.get("bpm", 0.0)
        if bpm_of_bar <= 0: 
            if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG _create_flow_datum: Invalid BPM ({bpm_of_bar}) for bar {target_bar_index}. Cannot quantize syllable durations properly. Skipping datum.", flush=True)
            return None
        
        beats_in_target_bar = target_bar_info.get("time_signature", (4,4))[0]
        beat_duration_sec = 60.0 / bpm_of_bar
        
        bar_duration_sec = target_bar_info.get("bar_duration_sec", beats_in_target_bar * beat_duration_sec)
        if bar_duration_sec <= 1e-6: return None

        start_offset_beats = (segment_actual_start_time_sec - target_bar_abs_start_time_sec) / beat_duration_sec
        duration_beats = (segment_actual_end_time_sec - segment_actual_start_time_sec) / beat_duration_sec
        
        start_offset_beats = np.clip(start_offset_beats, -0.5 * beats_in_target_bar, 1.5 * beats_in_target_bar) 
        duration_beats = np.clip(duration_beats, 0.05, 2.0 * beats_in_target_bar) 

        syllable_start_subdivisions: List[int] = []
        syllable_durations_quantized_indices: List[int] = []
        syllable_stress_values: List[int] = []
        
        subdivision_duration_sec = bar_duration_sec / self.subdivisions_per_bar
        if subdivision_duration_sec < 1e-6 : subdivision_duration_sec = 1e-6 

        for syl_detail in all_syllables_details_for_line:
            syl_abs_start_time = syl_detail["start_time"]
            syl_abs_end_time = syl_detail["end_time"]
            syl_stress = syl_detail.get("stress", 0) 

            syl_time_relative_to_target_bar_start_sec = syl_abs_start_time - target_bar_abs_start_time_sec
            subdivision_index_float = syl_time_relative_to_target_bar_start_sec / subdivision_duration_sec
            subdivision_index = int(round(subdivision_index_float)) 
            subdivision_index = max(0, min(subdivision_index, self.subdivisions_per_bar - 1))
            syllable_start_subdivisions.append(subdivision_index)

            syl_duration_raw_sec = syl_abs_end_time - syl_abs_start_time
            quantized_dur_idx = self.tokenizer.quantize_syllable_duration_to_bin_index(
                duration_sec=syl_duration_raw_sec, 
                bpm=bpm_of_bar
            )
            syllable_durations_quantized_indices.append(quantized_dur_idx)
            syllable_stress_values.append(syl_stress)
        
        if DEBUG_FLOW_EXTRACTOR:
            line_text_for_debug = "".join(s['syllable_text'] for s in all_syllables_details_for_line[:10]) 
            print(f"{LOG_PREFIX_FDE} DEBUG _create_flow_datum: Bar {target_bar_index}, LineInBar {line_idx_in_bar}, SylText: '{line_text_for_debug}...' ({total_syllables_in_segment} syls)", flush=True)
            print(f"    QuantSylDurs_TokenizerBeats: {syllable_durations_quantized_indices}", flush=True)
        
        return {
            "bar_index": target_bar_index, 
            "line_index_in_bar": line_idx_in_bar, 
            "syllables": total_syllables_in_segment,
            "start_offset_beats": round(start_offset_beats, 3),
            "duration_beats": round(duration_beats, 3),
            "syllable_start_subdivisions": syllable_start_subdivisions,
            "syllable_durations_quantized": syllable_durations_quantized_indices,
            "syllable_stresses": syllable_stress_values
        }

    def extract_flow_for_song(self, 
                              alignment_data_path: str, 
                              song_beat_features: SongBeatFeatures 
                             ) -> Optional[FlowData]:
        song_basename = os.path.basename(alignment_data_path).replace('.json', '')
        print(f"{LOG_PREFIX_FDE} [{song_basename}] Starting flow extraction for: {alignment_data_path}", flush=True)
        
        if DEBUG_FLOW_EXTRACTOR: print(f"\n{LOG_PREFIX_FDE} DEBUG extract_flow_for_song: Processing {alignment_data_path}", flush=True)
        
        word_alignments = parse_whisper_timestamped_json(alignment_data_path)
        if not word_alignments: 
            print(f"{LOG_PREFIX_FDE} [{song_basename}] No word alignments parsed from {alignment_data_path}. Cannot extract flow.", flush=True)
            return None 
        if not song_beat_features: 
            print(f"{LOG_PREFIX_FDE} [{song_basename}] No song beat features provided. Cannot extract flow.", flush=True)
            return None
        print(f"{LOG_PREFIX_FDE} [{song_basename}] Parsed {len(word_alignments)} word alignments.", flush=True)

        phrases_of_words = self._segment_words_into_phrases(word_alignments)
        if not phrases_of_words: 
            print(f"{LOG_PREFIX_FDE} [{song_basename}] Could not segment words into phrases.", flush=True)
            return None
        print(f"{LOG_PREFIX_FDE} [{song_basename}] Segmented into {len(phrases_of_words)} phrases.", flush=True)

        all_flow_data: FlowData = []
        
        bar_absolute_start_times: Dict[int, float] = {}
        bar_durations_map: Dict[int, float] = {} 
        has_abs_bar_start_times_in_sbf = all(isinstance(bf,dict) and "bar_start_time_sec" in bf and "bar_duration_sec" in bf for bf in song_beat_features)

        if not has_abs_bar_start_times_in_sbf:
            if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG [{song_basename}] 'bar_start_time_sec' or 'bar_duration_sec' missing in SBF, re-calculating.", flush=True)
            current_calc_abs_time = 0.0
            for bar_feat_idx, bar_feat in enumerate(song_beat_features):
                if not isinstance(bar_feat, dict): continue
                bar_idx_key = bar_feat.get("bar_index", bar_feat_idx) 
                bar_absolute_start_times[bar_idx_key] = current_calc_abs_time
                bpm = bar_feat.get("bpm", 120.0) 
                time_sig_num = bar_feat.get("time_signature", (4,4))[0]
                bar_dur_calc = 0.0
                if bpm > 0 and time_sig_num > 0:
                    beat_dur = 60.0 / bpm
                    bar_dur_calc = time_sig_num * beat_dur
                else: 
                    bar_dur_calc = 2.0 
                bar_durations_map[bar_idx_key] = bar_dur_calc
                current_calc_abs_time += bar_dur_calc
        else:
            for bar_feat in song_beat_features:
                 if isinstance(bar_feat, dict):
                    bar_absolute_start_times[bar_feat["bar_index"]] = bar_feat["bar_start_time_sec"]
                    bar_durations_map[bar_feat["bar_index"]] = bar_feat["bar_duration_sec"]
        
        bar_line_counters: Dict[int, int] = {idx: 0 for idx in bar_absolute_start_times.keys()}

        for phrase_idx, phrase_words in enumerate(phrases_of_words):
            if not phrase_words: continue
            if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG [{song_basename}] Processing Phrase {phrase_idx+1}/{len(phrases_of_words)}, words: {' '.join(w['word'] for w in phrase_words[:4])}...", flush=True)

            current_word_idx_in_phrase = 0
            while current_word_idx_in_phrase < len(phrase_words):
                first_word_of_segment = phrase_words[current_word_idx_in_phrase]
                
                # Find the bar that the first word of the segment belongs to.
                # Prioritize bars where the word starts within or just before the bar (pickup).
                assigned_bar_idx_for_segment = -1
                min_offset_to_bar_start = float('inf')

                # First pass: Find the bar where the word starts within or slightly before (pickup)
                for bar_idx_cand, bar_start_abs_time_cand in bar_absolute_start_times.items():
                    bar_dur_cand = bar_durations_map.get(bar_idx_cand, 2.0)
                    # A word is considered "in" a bar if it starts within the bar, or up to half a bar before it (pickup)
                    if first_word_of_segment["start_time"] >= bar_start_abs_time_cand - (bar_dur_cand / 2.0):
                        # Among these candidates, pick the one whose start time is closest to the word's start time
                        offset = first_word_of_segment["start_time"] - bar_start_abs_time_cand
                        if offset < min_offset_to_bar_start:
                            min_offset_to_bar_start = offset
                            assigned_bar_idx_for_segment = bar_idx_cand
                
                # If no bar was assigned by the primary logic (e.g., word starts much earlier than first bar, or much later than last bar)
                if assigned_bar_idx_for_segment == -1:
                    if not bar_absolute_start_times:
                        if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG [{song_basename}] Phrase {phrase_idx+1} - No bars available. Skipping phrase.", flush=True)
                        break # No bars at all, cannot assign.
                    
                    # Fallback: Find the bar that is closest in absolute time, regardless of pickup rule.
                    closest_bar_idx = -1
                    min_abs_diff = float('inf')
                    for bar_idx_cand, bar_start_abs_time_cand in bar_absolute_start_times.items():
                        abs_diff = abs(first_word_of_segment["start_time"] - bar_start_abs_time_cand)
                        if abs_diff < min_abs_diff:
                            min_abs_diff = abs_diff
                            closest_bar_idx = bar_idx_cand
                    assigned_bar_idx_for_segment = closest_bar_idx
                    
                    if assigned_bar_idx_for_segment == -1: # Should not happen if bar_absolute_start_times is not empty
                        if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG [{song_basename}] Phrase {phrase_idx+1} - Fallback: Still could not assign to any bar. Skipping phrase.", flush=True)
                        break

                target_bar_info = next((bf for bf in song_beat_features if isinstance(bf,dict) and bf["bar_index"] == assigned_bar_idx_for_segment), None)
                target_bar_abs_start_time = bar_absolute_start_times.get(assigned_bar_idx_for_segment)
                target_bar_duration = bar_durations_map.get(assigned_bar_idx_for_segment)

                if not target_bar_info or target_bar_abs_start_time is None or target_bar_duration is None:
                    if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG [{song_basename}] Phrase {phrase_idx+1} - Missing full bar info for bar {assigned_bar_idx_for_segment}. Skipping.", flush=True)
                    current_word_idx_in_phrase +=1 
                    continue
                
                words_for_this_bar_flow_datum: List[WordTiming] = []
                temp_phrase_word_idx = current_word_idx_in_phrase
                effective_bar_end_time = target_bar_abs_start_time + target_bar_duration 
                
                while temp_phrase_word_idx < len(phrase_words):
                    word_in_phrase = phrase_words[temp_phrase_word_idx]
                    # Include the first word of the segment even if it slightly overshoots the bar end,
                    # or if it starts within the bar.
                    if word_in_phrase["start_time"] < effective_bar_end_time or (current_word_idx_in_phrase == temp_phrase_word_idx):
                        words_for_this_bar_flow_datum.append(word_in_phrase)
                        temp_phrase_word_idx += 1
                    else:
                        break 

                if words_for_this_bar_flow_datum:
                    line_idx_val = bar_line_counters.get(assigned_bar_idx_for_segment, 0)
                    flow_datum = self._create_flow_datum_for_bar_segment(
                        words_for_this_bar_flow_datum,
                        target_bar_info,
                        target_bar_abs_start_time,
                        line_idx_val
                    )
                    if flow_datum:
                        all_flow_data.append(flow_datum)
                        bar_line_counters[assigned_bar_idx_for_segment] = line_idx_val + 1
                    current_word_idx_in_phrase = temp_phrase_word_idx 
                else:
                    current_word_idx_in_phrase += 1
                    if DEBUG_FLOW_EXTRACTOR: print(f"{LOG_PREFIX_FDE} DEBUG [{song_basename}] Phrase {phrase_idx+1} - No words collected for bar {assigned_bar_idx_for_segment}, though first word was assigned. Advancing.", flush=True)
        
        print(f"{LOG_PREFIX_FDE} [{song_basename}] Finished flow extraction. Total FlowDatum entries created: {len(all_flow_data)}", flush=True)
        return all_flow_data if all_flow_data else None


if __name__ == '__main__':
    DEBUG_FLOW_EXTRACTOR = True 
    dummy_wt_content = {
        "text": "yo this is a test flow line another one right here then a much longer line that will definitely span multiple bars to test the new segmentation logic okay",
        "segments": [
            { "words": [ {"text": "yo", "start": 0.5, "end": 0.9}, {"text": "this", "start": 1.0, "end": 1.4}, {"text": "is", "start": 1.5, "end": 1.8} ] },
            { "words": [ {"text": "a", "start": 2.2, "end": 2.6}, {"text": "test", "start": 2.7, "end": 3.3} ] },
            { "words": [ {"text": "flow", "start": 4.1, "end": 4.8}, {"text": "line", "start": 5.0, "end": 5.5}, {"text": "another", "start": 5.6, "end": 6.2}, {"text": "one", "start": 6.3, "end": 6.6}, {"text": "right", "start": 6.7, "end": 7.0}, {"text": "here", "start": 7.1, "end": 7.5}, {"text": "then", "start": 8.1, "end": 8.5}, {"text": "a", "start": 8.6, "end": 8.8}, {"text": "much", "start": 8.9, "end": 9.3} ] }
        ]
    }
    dummy_json_path = "dummy_alignment_flow_test_stress_fde.json" 
    if not os.path.exists(dummy_json_path) or True: 
        with open(dummy_json_path, "w") as f: json.dump(dummy_wt_content, f, indent=2)
        print(f"Created/Overwrote {dummy_json_path} for testing.", flush=True)

    dummy_sbf: SongBeatFeatures = [
        {"bar_index": 0, "bpm": 120.0, "time_signature": (4, 4), "bar_start_time_sec": 0.0, "bar_duration_sec": 2.0, "kick_events": [], "snare_events": [], "hihat_events": [], "bass_events": []},
        {"bar_index": 1, "bpm": 120.0, "time_signature": (4, 4), "bar_start_time_sec": 2.0, "bar_duration_sec": 2.0, "kick_events": [], "snare_events": [], "hihat_events": [], "bass_events": []},
        {"bar_index": 2, "bpm": 120.0, "time_signature": (4, 4), "bar_start_time_sec": 4.0, "bar_duration_sec": 2.0, "kick_events": [], "snare_events": [], "hihat_events": [], "bass_events": []},
        {"bar_index": 3, "bpm": 120.0, "time_signature": (4, 4), "bar_start_time_sec": 6.0, "bar_duration_sec": 2.0, "kick_events": [], "snare_events": [], "hihat_events": [], "bass_events": []},
        {"bar_index": 4, "bpm": 120.0, "time_signature": (4, 4), "bar_start_time_sec": 8.0, "bar_duration_sec": 2.0, "kick_events": [], "snare_events": [], "hihat_events": [], "bass_events": []}
    ]

    flow_tokenizer = FlowTokenizer()
    flow_extractor = FlowDataExtractor(tokenizer=flow_tokenizer, subdivisions_per_bar=16)
    print(f"\n--- Running Test with FlowDataExtractor (with Syllable Stress) ---", flush=True)
    extracted_flow_data = flow_extractor.extract_flow_for_song(dummy_json_path, dummy_sbf)
    
    if extracted_flow_data:
        print("\n--- Extracted Flow Data (with Syllable Stress) ---", flush=True)
        for i, fd in enumerate(extracted_flow_data): 
            print(f"  FlowDatum {i+1}: Bar {fd['bar_index']}, LineInBar {fd['line_index_in_bar']}, Syls: {fd['syllables']}, "
                  f"Offset: {fd['start_offset_beats']:.2f}b, Dur: {fd['duration_beats']:.2f}b, "
                  f"Subdivs: {fd['syllable_start_subdivisions']}, QuantSylDurs: {fd['syllable_durations_quantized']}, "
                  f"Stresses: {fd['syllable_stresses']}", flush=True)
        
        print(f"\nTotal FlowDatum entries generated: {len(extracted_flow_data)}", flush=True)
        assert len(extracted_flow_data) >= 3 
        if extracted_flow_data:
            assert "syllable_stresses" in extracted_flow_data[0]
            assert isinstance(extracted_flow_data[0]["syllable_stresses"], list)
            if extracted_flow_data[0]["syllables"] > 0:
                 assert len(extracted_flow_data[0]["syllable_stresses"]) == len(extracted_flow_data[0]["syllable_start_subdivisions"])
                 assert isinstance(extracted_flow_data[0]["syllable_stresses"][0], int)
    else:
        print("No flow data extracted from test file with stress addition.", flush=True)

    if os.path.exists(dummy_json_path):
        os.remove(dummy_json_path)