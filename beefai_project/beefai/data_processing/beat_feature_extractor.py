import librosa
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional, Any

# Adjust import path if beefai is not directly in PYTHONPATH
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from beefai.utils.data_types import BeatInfo, BarBeatFeatures, SongBeatFeatures

LOG_PREFIX_BFE = "[BFE]"

class BeatFeatureExtractor:
    def __init__(self, sample_rate: int = 44100, subdivisions_per_bar: int = 16):
        self.sample_rate = sample_rate
        self.subdivisions_per_bar = subdivisions_per_bar
        # This initial print is fine, worker-specific logs will follow.
        # print("BeatFeatureExtractor initialized. NOTE: Drum/Bass transcription is currently a STUB.")
        # print("It will simulate features using librosa.onset.onset_detect on relevant stems or full mix.")

    def _load_stem(self, stems_dir: str, stem_filename: str) -> Optional[np.ndarray]:
        """Loads a specific stem if it exists."""
        stem_path = os.path.join(stems_dir, stem_filename)
        if os.path.exists(stem_path):
            try:
                y, sr = librosa.load(stem_path, sr=self.sample_rate, mono=True)
                # print(f"{LOG_PREFIX_BFE}   Successfully loaded stem: {stem_filename} from {stems_dir}", flush=True)
                return y
            except Exception as e:
                print(f"{LOG_PREFIX_BFE}   Error loading stem {stem_path}: {e}", flush=True)
        # else:
            # print(f"{LOG_PREFIX_BFE}   Warning: Stem file {stem_path} not found.", flush=True)
        return None

    def _get_onsets_for_instrument(self, 
                                   y_instrument: Optional[np.ndarray], 
                                   y_full_mix_mono: np.ndarray, 
                                   instrument_name: str) -> np.ndarray:
        """Detects onsets for a given instrument, preferring its stem if available."""
        
        y_target = y_instrument if y_instrument is not None and y_instrument.size > 0 else y_full_mix_mono
        
        hop_length = 512 
        backtrack = True 
        
        if instrument_name == "kick":
            onsets = librosa.onset.onset_detect(y=y_target, sr=self.sample_rate, units='time', 
                                                hop_length=hop_length, backtrack=backtrack,
                                                pre_max=20, post_max=20, pre_avg=100, post_avg=100, 
                                                delta=0.1, wait=1)
        elif instrument_name == "snare":
            onsets = librosa.onset.onset_detect(y=y_target, sr=self.sample_rate, units='time',
                                                hop_length=hop_length, backtrack=backtrack,
                                                pre_max=15, post_max=15, pre_avg=80, post_avg=80,
                                                delta=0.08, wait=1)
        elif instrument_name == "hihat":
            onsets = librosa.onset.onset_detect(y=y_target, sr=self.sample_rate, units='time', 
                                                hop_length=hop_length//2, backtrack=backtrack, 
                                                pre_max=5, post_max=5, pre_avg=20, post_avg=20, 
                                                delta=0.03, wait=0) 
        elif instrument_name == "bass":
            onsets = librosa.onset.onset_detect(y=y_target, sr=self.sample_rate, units='time', 
                                                hop_length=hop_length, backtrack=backtrack,
                                                pre_max=25, post_max=25, pre_avg=120, post_avg=120,
                                                delta=0.15, wait=1)
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
            if -1e-6 <= time_within_bar_sec < bar_duration_sec + 1e-6 :
                subdivision_index = int(round(time_within_bar_sec / subdivision_duration_sec))
                subdivision_index = max(0, min(subdivision_index, self.subdivisions_per_bar - 1))
                quantized_indices.add(subdivision_index)
        return sorted(list(quantized_indices))

    def extract_features_for_song(self, audio_path: str, stems_input_dir: Optional[str] = None) -> SongBeatFeatures:
        """
        Extracts beat features for an entire song, bar by bar.
        Args:
            audio_path: Path to the full instrumental audio file.
            stems_input_dir: Path to a directory containing pre-separated stems (bass.wav, drums.wav).
        """
        song_basename = os.path.basename(audio_path)
        print(f"{LOG_PREFIX_BFE} [{song_basename}] Starting feature extraction for: {audio_path}", flush=True)

        try:
            y_full_mix, sr = librosa.load(audio_path, sr=self.sample_rate)
            y_mono = librosa.to_mono(y_full_mix) if y_full_mix.ndim > 1 else y_full_mix
            print(f"{LOG_PREFIX_BFE} [{song_basename}] Audio loaded successfully.", flush=True)
        except Exception as e:
            print(f"{LOG_PREFIX_BFE} [{song_basename}] Error loading audio {audio_path}: {e}", flush=True)
            return []

        # tempo_raw is what librosa.beat.beat_track returns. It might be a float or an array.
        tempo_raw, beat_frames = librosa.beat.beat_track(y=y_mono, sr=self.sample_rate, units='frames', hop_length=512)
        
        # Process tempo_raw to get a single float value for 'tempo'
        tempo: float
        if isinstance(tempo_raw, np.ndarray):
            if tempo_raw.size > 0:
                tempo = float(tempo_raw[0]) # Take the first estimated tempo if multiple
                print(f"{LOG_PREFIX_BFE} [{song_basename}] Multiple tempos detected by librosa, using first: {tempo:.2f} BPM.", flush=True)
            else:
                tempo = 120.0 # Default if empty array
                print(f"{LOG_PREFIX_BFE} [{song_basename}] Librosa returned empty array for tempo, defaulting to {tempo:.2f} BPM.", flush=True)
        elif isinstance(tempo_raw, (float, int)):
            tempo = float(tempo_raw)
        else: # Should not happen, but robust fallback
            tempo = 120.0
            print(f"{LOG_PREFIX_BFE} [{song_basename}] Unknown type for tempo from librosa ({type(tempo_raw)}), defaulting to {tempo:.2f} BPM.", flush=True)

        if tempo <= 0: # If tempo is 0 or negative (e.g. from a bad estimate or default)
            print(f"{LOG_PREFIX_BFE} [{song_basename}] Warning: Tempo estimated as {tempo:.2f} BPM, which is invalid. Using fallback 120 BPM.", flush=True)
            tempo = 120.0


        beat_times_sec = librosa.frames_to_time(beat_frames, sr=self.sample_rate, hop_length=512)
        # This print statement is now safe because 'tempo' is guaranteed to be a float.
        print(f"{LOG_PREFIX_BFE} [{song_basename}] Effective Tempo: {tempo:.2f} BPM, Beats detected: {len(beat_times_sec)}", flush=True)
        
        if beat_times_sec.size == 0:
            print(f"{LOG_PREFIX_BFE} [{song_basename}] Warning: No beats detected. Cannot extract bar features.", flush=True)
            return []

        beats_per_bar = 4 
        downbeat_indices = np.arange(0, len(beat_frames), beats_per_bar)
        downbeat_times_sec = beat_times_sec[downbeat_indices[downbeat_indices < len(beat_times_sec)]]

        if downbeat_times_sec.size == 0: 
            if beat_times_sec.size > 0:
                print(f"{LOG_PREFIX_BFE} [{song_basename}] Warning: No downbeats explicitly found, using first beat as start of first bar.", flush=True)
                downbeat_times_sec = np.array([beat_times_sec[0]])
            else:
                print(f"{LOG_PREFIX_BFE} [{song_basename}] Critical Warning: No beats or downbeats. Returning empty features.", flush=True)
                return []
        print(f"{LOG_PREFIX_BFE} [{song_basename}] Downbeats estimated: {len(downbeat_times_sec)}", flush=True)

        y_drums_stem: Optional[np.ndarray] = None
        y_bass_stem: Optional[np.ndarray] = None
        if stems_input_dir and os.path.isdir(stems_input_dir):
            print(f"{LOG_PREFIX_BFE} [{song_basename}] Attempting to load stems from: {stems_input_dir}", flush=True)
            y_drums_stem = self._load_stem(stems_input_dir, "drums.wav")
            y_bass_stem = self._load_stem(stems_input_dir, "bass.wav")
            if y_drums_stem is not None: print(f"{LOG_PREFIX_BFE} [{song_basename}] Drums stem loaded.", flush=True)
            else: print(f"{LOG_PREFIX_BFE} [{song_basename}] Drum stem not found/loaded from {stems_input_dir}. Drum onsets will use full mix.", flush=True)
            if y_bass_stem is not None: print(f"{LOG_PREFIX_BFE} [{song_basename}] Bass stem loaded.", flush=True)
            else: print(f"{LOG_PREFIX_BFE} [{song_basename}] Bass stem not found/loaded from {stems_input_dir}. Bass onsets will use full mix.", flush=True)
        else:
            print(f"{LOG_PREFIX_BFE} [{song_basename}] No valid stems_input_dir provided ({stems_input_dir}). Onset detection will use full mix.", flush=True)

        song_features: SongBeatFeatures = []
        num_bars = len(downbeat_times_sec)
        print(f"{LOG_PREFIX_BFE} [{song_basename}] Processing {num_bars} bars...", flush=True)

        for i in range(num_bars):
            bar_start_time = downbeat_times_sec[i]
            if (i + 1) < num_bars:
                bar_end_time = downbeat_times_sec[i+1]
            else: 
                beats_in_last_bar_mask = (beat_times_sec >= bar_start_time)
                if np.any(beats_in_last_bar_mask):
                    # last_beat_in_bar = beat_times_sec[beats_in_last_bar_mask][-1] # Not used
                    avg_beat_duration_song = (beat_times_sec[-1] - beat_times_sec[0]) / (len(beat_times_sec) -1) if len(beat_times_sec) > 1 else (60.0/tempo if tempo > 0 else 0.5)
                    bar_end_time = bar_start_time + beats_per_bar * avg_beat_duration_song
                else: 
                    bar_end_time = bar_start_time + (beats_per_bar * (60.0 / tempo) if tempo > 0 else 2.0) # Use the processed 'tempo'
            bar_duration = bar_end_time - bar_start_time

            if bar_duration <= 1e-6: 
                # print(f"{LOG_PREFIX_BFE} [{song_basename}] Skipping bar {i} due to zero or negative duration ({bar_duration:.3f}s).", flush=True)
                continue
            
            kick_onsets_sec = self._get_onsets_for_instrument(y_drums_stem, y_mono, "kick")
            snare_onsets_sec = self._get_onsets_for_instrument(y_drums_stem, y_mono, "snare")
            hihat_onsets_sec = self._get_onsets_for_instrument(y_drums_stem, y_mono, "hihat")
            bass_onsets_sec = self._get_onsets_for_instrument(y_bass_stem, y_mono, "bass")
            
            bar_feat: BarBeatFeatures = {
                "bar_index": i,
                "bpm": float(tempo), # Use the processed 'tempo'
                "time_signature": (beats_per_bar, 4),
                "kick_events": self._quantize_onsets_to_bar_subdivisions(kick_onsets_sec, bar_start_time, bar_duration),
                "snare_events": self._quantize_onsets_to_bar_subdivisions(snare_onsets_sec, bar_start_time, bar_duration),
                "hihat_events": self._quantize_onsets_to_bar_subdivisions(hihat_onsets_sec, bar_start_time, bar_duration),
                "bass_events": self._quantize_onsets_to_bar_subdivisions(bass_onsets_sec, bar_start_time, bar_duration),
                "bar_start_time_sec": round(bar_start_time, 3), 
                "bar_duration_sec": round(bar_duration, 3)
            }
            song_features.append(bar_feat)
            if (i + 1) % 10 == 0 or (i + 1) == num_bars: 
                 print(f"{LOG_PREFIX_BFE} [{song_basename}] Processed bar {i+1}/{num_bars}.", flush=True)
        
        print(f"{LOG_PREFIX_BFE} [{song_basename}] Finished feature extraction. Total bars with features: {len(song_features)}.", flush=True)
        return song_features


if __name__ == "__main__":
    import soundfile as sf 
    
    dummy_data_dir = os.path.join("data", "temp_testing_bfe") 
    dummy_instrumentals_dir = os.path.join(dummy_data_dir, "instrumentals")
    dummy_stems_base_dir = os.path.join(dummy_data_dir, "stems_separated_bfe")
    
    os.makedirs(dummy_instrumentals_dir, exist_ok=True)

    dummy_sr = 44100
    dummy_duration = 20 
    
    dummy_instrumental_audio = np.random.rand(dummy_sr * dummy_duration) * 0.2 - 0.1
    for i in range(int(dummy_duration * 2)): 
        pulse_time = i * 0.5
        if pulse_time + 0.1 < dummy_duration:
            start_sample = int(pulse_time * dummy_sr)
            end_sample = int((pulse_time + 0.05) * dummy_sr)
            if start_sample < len(dummy_instrumental_audio) and end_sample < len(dummy_instrumental_audio) and start_sample < end_sample:
                dummy_instrumental_audio[start_sample : end_sample] += 0.3 * np.random.rand(end_sample - start_sample)

    dummy_instrumental_path = os.path.join(dummy_instrumentals_dir, "dummy_song_bfe.wav")
    sf.write(dummy_instrumental_path, dummy_instrumental_audio, dummy_sr)
    print(f"Created dummy instrumental: {dummy_instrumental_path}", flush=True)

    demucs_model_name = "htdemucs_ft" 
    dummy_song_stems_dir = os.path.join(dummy_stems_base_dir, demucs_model_name, "dummy_song_bfe")
    os.makedirs(dummy_song_stems_dir, exist_ok=True)
    
    dummy_drums_stem = np.random.rand(dummy_sr * dummy_duration) * 0.15 - 0.075
    sf.write(os.path.join(dummy_song_stems_dir, "drums.wav"), dummy_drums_stem, dummy_sr)
    print(f"Created dummy drums stem: {os.path.join(dummy_song_stems_dir, 'drums.wav')}", flush=True)
    
    dummy_bass_stem = np.random.rand(dummy_sr * dummy_duration) * 0.1 - 0.05
    sf.write(os.path.join(dummy_song_stems_dir, "bass.wav"), dummy_bass_stem, dummy_sr)
    print(f"Created dummy bass stem: {os.path.join(dummy_song_stems_dir, 'bass.wav')}", flush=True)


    extractor = BeatFeatureExtractor(sample_rate=dummy_sr)
    
    print("\n--- Test Case 1: Using dummy instrumental, no specific stems_input_dir provided ---", flush=True)
    song_features_case1 = extractor.extract_features_for_song(dummy_instrumental_path, stems_input_dir=None)
    if song_features_case1:
        print(f"Extracted {len(song_features_case1)} bars of features (Case 1).", flush=True)
        if len(song_features_case1) > 0: print("First bar (Case 1):", song_features_case1[0], flush=True)
    else:
        print(f"Could not extract features (Case 1).", flush=True)

    print("\n--- Test Case 2: Using dummy instrumental AND providing the dummy_song_stems_dir ---", flush=True)
    song_features_case2 = extractor.extract_features_for_song(dummy_instrumental_path, stems_input_dir=dummy_song_stems_dir)
    if song_features_case2:
        print(f"Extracted {len(song_features_case2)} bars of features (Case 2).", flush=True)
        if len(song_features_case2) > 0: print("First bar (Case 2):", song_features_case2[0], flush=True)
    else:
        print(f"Could not extract features (Case 2).", flush=True)

    if song_features_case1 and song_features_case2 and len(song_features_case1) > 0 and len(song_features_case2) > 0:
        if song_features_case1[0]["kick_events"] != song_features_case2[0]["kick_events"]:
            print("\nNote: Kick events differ between Case 1 (full mix) and Case 2 (stems), as expected if stems provide clearer signal.", flush=True)
        else:
            print("\nNote: Kick events are the same. This might happen if stems are very similar to full mix for kicks or onset detection parameters are not sensitive enough.", flush=True)

    # import shutil
    # if os.path.exists(dummy_data_dir):
    #     shutil.rmtree(dummy_data_dir)
    #     print(f"\nCleaned up dummy data directory: {dummy_data_dir}", flush=True)