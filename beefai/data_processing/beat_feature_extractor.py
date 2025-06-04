# beefai/data_processing/beat_feature_extractor.py
import librosa
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional, Any
import concurrent.futures
import time # Added for timing in __main__

# Adjust import path if beefai is not directly in PYTHONPATH
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from beefai.utils.data_types import BeatInfo, BarBeatFeatures, SongBeatFeatures

LOG_PREFIX_BFE = "[BFE]"

class BeatFeatureExtractor:
    def __init__(self, 
                 sample_rate: int = 44100, 
                 subdivisions_per_bar: int = 16,
                 max_workers_onset: int = 4,      # Max workers for parallel onset detection
                 max_workers_stem_load: int = 2): # Max workers for parallel stem loading
        self.sample_rate = sample_rate
        self.subdivisions_per_bar = subdivisions_per_bar
        self.max_workers_onset = max(1, max_workers_onset) 
        self.max_workers_stem_load = max(1, max_workers_stem_load)

    def _load_stem(self, stems_dir: str, stem_filename: str, 
                   offset_sec: Optional[float] = None, 
                   duration_sec: Optional[float] = None) -> Optional[np.ndarray]:
        """Loads a specific stem if it exists, with offset and duration."""
        stem_path = os.path.join(stems_dir, stem_filename)
        if os.path.exists(stem_path):
            try:
                y, sr = librosa.load(stem_path, sr=self.sample_rate, mono=True,
                                     offset=offset_sec if offset_sec is not None else 0.0,
                                     duration=duration_sec)
                return y
            except Exception as e:
                print(f"{LOG_PREFIX_BFE}   Error loading stem {stem_path}: {e}", flush=True)
        return None

    def _get_onsets_for_instrument(self, 
                                   y_instrument: Optional[np.ndarray], 
                                   y_full_mix_mono: np.ndarray, 
                                   instrument_name: str) -> np.ndarray:
        """Detects onsets for a given instrument, preferring its stem if available."""
        
        y_target = y_instrument if y_instrument is not None and y_instrument.size > 0 else y_full_mix_mono
        
        if y_target is None or y_target.size == 0: 
            return np.array([])

        # Define onset detection parameters for each instrument
        onset_params = {
            "kick":    {"pre_max": 20, "post_max": 20, "pre_avg": 100, "post_avg": 100, "delta": 0.1,  "wait": 1, "hop_length": 512},
            "snare":   {"pre_max": 15, "post_max": 15, "pre_avg": 80,  "post_avg": 80,  "delta": 0.08, "wait": 1, "hop_length": 512},
            "hihat":   {"pre_max": 5,  "post_max": 5,  "pre_avg": 20,  "post_avg": 20,  "delta": 0.03, "wait": 0, "hop_length": 512 // 2},
            "bass":    {"pre_max": 25, "post_max": 25, "pre_avg": 120, "post_avg": 120, "delta": 0.15, "wait": 1, "hop_length": 512},
        }
        
        params = onset_params.get(instrument_name)
        if params:
            onsets = librosa.onset.onset_detect(y=y_target, sr=self.sample_rate, units='time', 
                                                backtrack=True, **params)
        else:
            onsets = np.array([]) 

        return onsets if onsets is not None and onsets.size > 0 else np.array([])


    def _quantize_onsets_to_bar_subdivisions(self, 
                                             onset_times_sec: np.ndarray, 
                                             bar_start_time_sec: float, 
                                             bar_duration_sec: float) -> List[int]:
        """
        Quantizes onset times (in seconds) to subdivisions within a bar.
        Returns a list of 0-indexed subdivision indices where events occurred.
        """
        if bar_duration_sec <= 1e-6: return [] 
        subdivision_duration_sec = bar_duration_sec / self.subdivisions_per_bar
        quantized_indices = set()

        for onset_sec in onset_times_sec:
            time_within_bar_sec = onset_sec - bar_start_time_sec
            if -1e-6 <= time_within_bar_sec < bar_duration_sec + 1e-6 : # Loosen tolerance slightly
                subdivision_index_float = time_within_bar_sec / subdivision_duration_sec
                # Round carefully, ensuring it stays within [0, subdivisions_per_bar - 1]
                subdivision_index = int(round(subdivision_index_float))
                subdivision_index = max(0, min(subdivision_index, self.subdivisions_per_bar - 1))
                quantized_indices.add(subdivision_index)
        return sorted(list(quantized_indices))

    def extract_features_for_song(self, audio_path: str, 
                                  stems_input_dir: Optional[str] = None,
                                  audio_offset_sec: Optional[float] = None,
                                  audio_duration_sec: Optional[float] = None
                                 ) -> SongBeatFeatures:
        """
        Extracts beat features for an entire song (or a segment of it).
        Args:
            audio_path: Path to the full instrumental audio file.
            stems_input_dir: Path to a directory containing pre-separated stems.
            audio_offset_sec: Start loading audio from this time (in seconds).
            audio_duration_sec: Load only this duration of audio (in seconds).
        """
        song_basename = os.path.basename(audio_path)
        offset_str = f", offset={audio_offset_sec}s" if audio_offset_sec is not None else ""
        duration_str = f", duration={audio_duration_sec}s" if audio_duration_sec is not None else ""
        print(f"{LOG_PREFIX_BFE} [{song_basename}] Starting feature extraction for: {audio_path}{offset_str}{duration_str}", flush=True)

        try:
            y_full_mix, sr = librosa.load(audio_path, sr=self.sample_rate, 
                                          offset=audio_offset_sec if audio_offset_sec is not None else 0.0,
                                          duration=audio_duration_sec)
            if y_full_mix.size == 0:
                print(f"{LOG_PREFIX_BFE} [{song_basename}] Loaded audio is empty (possibly due to offset/duration). Cannot extract features.", flush=True)
                return []
            y_mono = librosa.to_mono(y_full_mix) if y_full_mix.ndim > 1 else y_full_mix
            print(f"{LOG_PREFIX_BFE} [{song_basename}] Audio loaded successfully.", flush=True)
        except Exception as e:
            print(f"{LOG_PREFIX_BFE} [{song_basename}] Error loading audio {audio_path}: {e}", flush=True)
            return []

        tempo_raw, beat_frames = librosa.beat.beat_track(y=y_mono, sr=self.sample_rate, units='frames', hop_length=512)
        
        tempo: float
        if isinstance(tempo_raw, np.ndarray):
            if tempo_raw.size > 0:
                tempo = float(tempo_raw[0]) 
            else: tempo = 120.0 
        elif isinstance(tempo_raw, (float, int)): tempo = float(tempo_raw)
        else: tempo = 120.0
        if tempo <= 0: tempo = 120.0

        # beat_times_sec are relative to the start of the loaded segment (y_mono)
        beat_times_sec = librosa.frames_to_time(beat_frames, sr=self.sample_rate, hop_length=512)
        print(f"{LOG_PREFIX_BFE} [{song_basename}] Effective Tempo: {tempo:.2f} BPM, Beats detected in segment: {len(beat_times_sec)}", flush=True)
        
        if beat_times_sec.size == 0:
            print(f"{LOG_PREFIX_BFE} [{song_basename}] Warning: No beats detected in segment. Cannot extract bar features.", flush=True)
            return []

        beats_per_bar = 4 
        # Downbeat indices are relative to the beats found *within the segment*
        downbeat_indices_in_segment = np.arange(0, len(beat_frames), beats_per_bar)
        # Ensure indices are valid for beat_times_sec derived from the segment
        valid_downbeat_indices = downbeat_indices_in_segment[downbeat_indices_in_segment < len(beat_times_sec)]
        downbeat_times_sec_in_segment = beat_times_sec[valid_downbeat_indices]


        if downbeat_times_sec_in_segment.size == 0: 
            if beat_times_sec.size > 0:
                downbeat_times_sec_in_segment = np.array([beat_times_sec[0]])
            else: # Should be caught by earlier check, but for safety
                print(f"{LOG_PREFIX_BFE} [{song_basename}] Critical Warning: No beats or downbeats in segment. Returning empty features.", flush=True)
                return []

        # Parallel Stem Loading
        y_drums_stem: Optional[np.ndarray] = None
        y_bass_stem: Optional[np.ndarray] = None
        if stems_input_dir and os.path.isdir(stems_input_dir):
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_stem_load) as executor:
                future_drums = executor.submit(self._load_stem, stems_input_dir, "drums.wav", audio_offset_sec, audio_duration_sec)
                future_bass = executor.submit(self._load_stem, stems_input_dir, "bass.wav", audio_offset_sec, audio_duration_sec)
                try:
                    y_drums_stem = future_drums.result()
                    y_bass_stem = future_bass.result()
                except Exception as e_stem_load:
                    print(f"{LOG_PREFIX_BFE} [{song_basename}] Error during parallel stem loading: {e_stem_load}", flush=True)

        # Parallel Onset Detection for the entire segment
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_onset) as executor:
            future_kick_onsets = executor.submit(self._get_onsets_for_instrument, y_drums_stem, y_mono, "kick")
            future_snare_onsets = executor.submit(self._get_onsets_for_instrument, y_drums_stem, y_mono, "snare")
            future_hihat_onsets = executor.submit(self._get_onsets_for_instrument, y_drums_stem, y_mono, "hihat")
            future_bass_onsets = executor.submit(self._get_onsets_for_instrument, y_bass_stem, y_mono, "bass")
            
            try:
                kick_onsets_sec = future_kick_onsets.result()
                snare_onsets_sec = future_snare_onsets.result()
                hihat_onsets_sec = future_hihat_onsets.result()
                bass_onsets_sec = future_bass_onsets.result()
            except Exception as e_onset:
                 print(f"{LOG_PREFIX_BFE} [{song_basename}] Error during parallel onset detection: {e_onset}", flush=True)
                 # Fallback to empty onsets if parallel execution fails, to prevent crash
                 kick_onsets_sec, snare_onsets_sec, hihat_onsets_sec, bass_onsets_sec = np.array([]), np.array([]), np.array([]), np.array([])

        song_features: SongBeatFeatures = []
        num_bars_in_segment = len(downbeat_times_sec_in_segment)

        for i in range(num_bars_in_segment):
            bar_start_time_in_segment = downbeat_times_sec_in_segment[i]
            if (i + 1) < num_bars_in_segment:
                bar_end_time_in_segment = downbeat_times_sec_in_segment[i+1]
            else: 
                # Estimate end of last bar in segment
                avg_beat_duration_segment = (beat_times_sec[-1] - beat_times_sec[0]) / (len(beat_times_sec) -1) if len(beat_times_sec) > 1 else (60.0/tempo if tempo > 0 else 0.5)
                bar_end_time_in_segment = bar_start_time_in_segment + beats_per_bar * avg_beat_duration_segment
                # Ensure it doesn't exceed the duration of y_mono (the loaded segment)
                segment_duration_samples = len(y_mono)
                segment_duration_seconds = segment_duration_samples / self.sample_rate
                bar_end_time_in_segment = min(bar_end_time_in_segment, segment_duration_seconds)


            bar_duration = bar_end_time_in_segment - bar_start_time_in_segment

            if bar_duration <= 1e-6: 
                continue
            
            # The bar_index should be continuous, even if processing a segment.
            # If audio_offset_sec is provided, we need to estimate what original bar index this corresponds to.
            # For now, let's keep bar_index 0-indexed for the segment. The calling function
            # (`visualize_flow_rhythm`) will handle the global bar context for generated flow.
            current_bar_index_for_output = i 

            bar_feat: BarBeatFeatures = {
                "bar_index": current_bar_index_for_output, # 0-indexed within the segment
                "bpm": float(tempo), 
                "time_signature": (beats_per_bar, 4),
                "kick_events": self._quantize_onsets_to_bar_subdivisions(kick_onsets_sec, bar_start_time_in_segment, bar_duration),
                "snare_events": self._quantize_onsets_to_bar_subdivisions(snare_onsets_sec, bar_start_time_in_segment, bar_duration),
                "hihat_events": self._quantize_onsets_to_bar_subdivisions(hihat_onsets_sec, bar_start_time_in_segment, bar_duration),
                "bass_events": self._quantize_onsets_to_bar_subdivisions(bass_onsets_sec, bar_start_time_in_segment, bar_duration),
                "bar_start_time_sec": round(bar_start_time_in_segment + (audio_offset_sec if audio_offset_sec else 0.0), 3), # Absolute time in original audio
                "bar_duration_sec": round(bar_duration, 3)
            }
            song_features.append(bar_feat)
        
        print(f"{LOG_PREFIX_BFE} [{song_basename}] Finished feature extraction for segment. Total bars with features: {len(song_features)}.", flush=True)
        return song_features


if __name__ == "__main__":
    # This __main__ block in BeatFeatureExtractor is for its own testing,
    # not directly called by the rhythm_visualizer.
    # It would need updates to test the offset/duration functionality if desired.
    import soundfile as sf_dummy
    
    dummy_data_dir = os.path.join("data", "temp_testing_bfe") 
    dummy_instrumentals_dir = os.path.join(dummy_data_dir, "instrumentals")
    os.makedirs(dummy_instrumentals_dir, exist_ok=True)
    dummy_sr = 44100
    dummy_full_duration = 60 # seconds for a longer dummy track
    
    # Create a longer dummy track to test offset/duration
    dummy_instrumental_audio_full = np.random.rand(dummy_sr * dummy_full_duration) * 0.2 - 0.1
    for i in range(int(dummy_full_duration * 2)): 
        pulse_time = i * 0.5
        if pulse_time + 0.1 < dummy_full_duration:
            start_sample = int(pulse_time * dummy_sr)
            end_sample = int((pulse_time + 0.05) * dummy_sr)
            if start_sample < len(dummy_instrumental_audio_full) and end_sample < len(dummy_instrumental_audio_full) and start_sample < end_sample:
                dummy_instrumental_audio_full[start_sample : end_sample] += 0.3 * np.random.rand(end_sample - start_sample)

    dummy_instrumental_path_full = os.path.join(dummy_instrumentals_dir, "dummy_song_bfe_full.wav")
    sf_dummy.write(dummy_instrumental_path_full, dummy_instrumental_audio_full, dummy_sr)
    print(f"Created dummy full instrumental: {dummy_instrumental_path_full}", flush=True)

    # Test with default workers
    extractor = BeatFeatureExtractor(sample_rate=dummy_sr) 
    # Or test with specific worker counts:
    # extractor = BeatFeatureExtractor(sample_rate=dummy_sr, max_workers_onset=2, max_workers_stem_load=1)
    
    print("\n--- Test Case 1: Full dummy instrumental ---", flush=True)
    start_time_full = time.time()
    features_full = extractor.extract_features_for_song(dummy_instrumental_path_full)
    end_time_full = time.time()
    print(f"Full extraction time: {end_time_full - start_time_full:.2f}s")
    if features_full: print(f"Extracted {len(features_full)} bars (Full). First bar BPM: {features_full[0]['bpm'] if features_full else 'N/A'}")

    print("\n--- Test Case 2: Segment of dummy instrumental (offset 10s, duration 20s) ---", flush=True)
    start_time_segment = time.time()
    features_segment = extractor.extract_features_for_song(dummy_instrumental_path_full, 
                                                           audio_offset_sec=10.0, 
                                                           audio_duration_sec=20.0)
    end_time_segment = time.time()
    print(f"Segment extraction time: {end_time_segment - start_time_segment:.2f}s")
    if features_segment: 
        print(f"Extracted {len(features_segment)} bars (Segment). First bar BPM: {features_segment[0]['bpm'] if features_segment else 'N/A'}")
        if features_segment:
            print(f"  First bar in segment starts at original audio time: {features_segment[0]['bar_start_time_sec']:.2f}s (should be >= 10.0s)")
            assert features_segment[0]['bar_start_time_sec'] >= 10.0 - 0.5 # Allow for some beat detection variance

    # Cleanup (optional)
    # import shutil
    # if os.path.exists(dummy_data_dir):
    #     shutil.rmtree(dummy_data_dir)