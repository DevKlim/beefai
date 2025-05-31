import librosa
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional, Any

# Adjust import path if beefai is not directly in PYTHONPATH
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from beefai.utils.data_types import BeatInfo, BarBeatFeatures, SongBeatFeatures

# Configuration for simulated features - these can be adjusted
SIM_KICK_INTERVAL_BEATS = [1.0, 0.5]  # e.g., on the beat, or on the half beat
SIM_SNARE_POS_BEATS = [1.0, 3.0]      # e.g., on beats 2 and 4 (0-indexed)
SIM_HIHAT_INTERVAL_BEATS = [0.25]     # e.g., 16th notes
SIM_BASS_INTERVAL_BEATS = [0.5, 1.0]  # e.g., 8th or quarter notes

class BeatFeatureExtractor:
    def __init__(self, sample_rate: int = 44100, subdivisions_per_bar: int = 16):
        self.sample_rate = sample_rate
        self.subdivisions_per_bar = subdivisions_per_bar
        # Placeholder for actual source separation and drum transcription models
        self.source_separator = None # e.g., Demucs model
        self.drum_transcriber = None # e.g., a trained drum MIDI transcription model
        print("BeatFeatureExtractor initialized. NOTE: Drum/Bass transcription is currently a STUB.")
        print("It will simulate features. For real features, integrate Demucs/Spleeter and a drum transcription model.")

    def _separate_sources(self, audio_path: str, output_dir: str) -> Dict[str, np.ndarray]:
        """
        Placeholder for separating audio into stems (drums, bass, vocals, other).
        In a real implementation, this would use Demucs, Spleeter, or similar.
        """
        print(f"STUB: Simulating source separation for {audio_path}. Looking for pre-existing stems in {output_dir} or returning empty if not found.")
        stems = {}
        expected_stems = {
            "drums": "drums.wav",
            "bass": "bass.wav"
            # "vocals": "vocals.wav", # Not strictly needed for beat features but good for completeness
            # "other": "other.wav"
        }
        if not os.path.isdir(output_dir):
            print(f"  Warning: Stems directory {output_dir} does not exist. Cannot load pre-separated stems.")
            return stems # Return empty dict

        for stem_name, stem_filename in expected_stems.items():
            stem_path = os.path.join(output_dir, stem_filename)
            if os.path.exists(stem_path):
                try:
                    y, sr = librosa.load(stem_path, sr=self.sample_rate, mono=True)
                    stems[stem_name] = y
                    print(f"  Loaded pre-existing stem: {stem_name} from {stem_path}")
                except Exception as e:
                    print(f"  Error loading stem {stem_path}: {e}")
            else:
                print(f"  Warning: Stem file {stem_path} not found.")
        return stems

    def _transcribe_drums(self, drum_track: np.ndarray, beat_times: np.ndarray) -> Dict[str, List[float]]:
        """
        Placeholder for transcribing drum events (kick, snare, hi-hat) from a drum track.
        Should return a dictionary with instrument names as keys and lists of onset times (sec) as values.
        """
        print("STUB: Simulating drum transcription. This would use a dedicated model.")
        # Simplified simulation: random onsets for demonstration
        # In a real scenario, use onset detection on the drum track, then classify onsets.
        kick_onsets = np.random.choice(beat_times, size=len(beat_times)//2, replace=False).tolist()
        snare_onsets = np.random.choice(beat_times, size=len(beat_times)//4, replace=False).tolist()
        # Filter snare onsets that are too close to kick onsets for this simple simulation
        snare_onsets = [s for s in snare_onsets if not any(abs(s - k) < 0.1 for k in kick_onsets)]
        
        hihat_onsets = []
        for i in range(len(beat_times) -1):
            hihat_onsets.extend(np.linspace(beat_times[i], beat_times[i+1], 4, endpoint=False))


        return {
            "kick": sorted(kick_onsets),
            "snare": sorted(snare_onsets),
            "hihat": sorted(list(set(hihat_onsets))) # set to remove duplicates from linspace edges
        }

    def _get_simulated_instrument_onsets(self, y_mono: np.ndarray, instrument_name: str, bar_beat_times: np.ndarray) -> np.ndarray:
        """ Simulates onsets for a given instrument type based on beat times. Returns a NumPy array. """
        onsets_sec_list = [] # Start with a list

        if instrument_name == "kick":
            # Simulate kicks on some strong beats
            detected_onsets = librosa.onset.onset_detect(y=y_mono, sr=self.sample_rate, units='time', backtrack=True, pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.1, wait=1)
            if detected_onsets.size > 0: onsets_sec_list = detected_onsets.tolist()
        elif instrument_name == "snare":
            # Simulate snares on typical backbeats
            detected_onsets = librosa.onset.onset_detect(y=y_mono, sr=self.sample_rate, units='time', backtrack=True, pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.15, wait=1)
            if detected_onsets.size > 0: onsets_sec_list = detected_onsets.tolist()
        elif instrument_name == "hihat":
            # Simulate hi-hats more frequently
            detected_onsets = librosa.onset.onset_detect(y=y_mono, sr=self.sample_rate, units='time', backtrack=True, hop_length=256, pre_max=10, post_max=10, pre_avg=50, post_avg=50, delta=0.05, wait=0)
            if detected_onsets.size > 0: onsets_sec_list = detected_onsets.tolist()
        elif instrument_name == "bass": # Added explicit handling for bass simulation
            detected_onsets = librosa.onset.onset_detect(y=y_mono, sr=self.sample_rate, units='time', backtrack=True, pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.2, wait=1) # Adjusted delta for bass
            if detected_onsets.size > 0: onsets_sec_list = detected_onsets.tolist()
        else: # Default case if instrument_name is not specifically handled
            onsets_sec_list = []

        # Convert to NumPy array before further processing
        onsets_sec_np = np.array(onsets_sec_list)

        if onsets_sec_np.size > 0:
            # Snap to nearest beat for simplicity in simulation
            snapped_onsets = []
            if bar_beat_times.size > 0: # Ensure bar_beat_times is not empty
                for onset_t in onsets_sec_np:
                    nearest_beat_idx = np.argmin(np.abs(bar_beat_times - onset_t))
                    snapped_onsets.append(bar_beat_times[nearest_beat_idx])
            return np.array(sorted(list(set(snapped_onsets)))) # Unique sorted onsets
        return np.array([])


    def _extract_bass_activity(self, bass_track: np.ndarray, beat_times: np.ndarray) -> List[float]:
        """
        Placeholder for extracting bass activity.
        Could involve onset detection, pitch analysis, etc.
        """
        print("STUB: Simulating bass activity detection.")
        if bass_track is None or len(bass_track) == 0:
            return []
        # Simplified: detect onsets in the bass track
        bass_onsets = librosa.onset.onset_detect(y=bass_track, sr=self.sample_rate, units='time', backtrack=True)
        return sorted(bass_onsets.tolist()) if bass_onsets.size > 0 else []


    def _quantize_onsets_to_bar_subdivisions(self, onset_times_sec: List[float], bar_start_time_sec: float, bar_duration_sec: float) -> List[int]:
        """
        Quantizes onset times (in seconds) to subdivisions within a bar.
        Returns a list of 0-indexed subdivision indices where events occurred.
        """
        if bar_duration_sec <= 0: return []
        subdivision_duration_sec = bar_duration_sec / self.subdivisions_per_bar
        quantized_indices = set()

        for onset_sec in onset_times_sec:
            time_within_bar_sec = onset_sec - bar_start_time_sec
            if 0 <= time_within_bar_sec < bar_duration_sec: # Ensure onset is within this bar
                # Add a small epsilon to handle floating point inaccuracies at the very end of a subdivision
                subdivision_index = int((time_within_bar_sec + 1e-6) / subdivision_duration_sec)
                # Clamp to max subdivision index
                subdivision_index = min(subdivision_index, self.subdivisions_per_bar - 1)
                quantized_indices.add(subdivision_index)
        return sorted(list(quantized_indices))

    def extract_features_for_song(self, audio_path: str, stems_input_dir: Optional[str] = None) -> SongBeatFeatures:
        """
        Extracts beat features for an entire song, bar by bar.
        This is the main method to be called externally.
        Args:
            audio_path: Path to the full instrumental audio file.
            stems_input_dir: Path to a directory containing pre-separated stems (bass.wav, drums.wav).
                             If None, source separation will be (simulatedly) attempted.
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            y_mono = librosa.to_mono(y) if y.ndim > 1 else y
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return []

        tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=self.sample_rate)
        beat_times_sec = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        
        if beat_times_sec.size == 0:
            print(f"Warning: No beats detected for {audio_path}. Cannot extract bar features.")
            return []

        # Estimate downbeats (start of bars) - assumes 4/4 for simplicity here
        # A more robust solution would use librosa.beat.plp or analyze meter
        beats_per_bar = 4 # Common default
        downbeat_frames = beat_frames[::beats_per_bar]
        downbeat_times_sec = librosa.frames_to_time(downbeat_frames, sr=self.sample_rate)
        
        if downbeat_times_sec.size == 0: # If no downbeats, use first beat as first downbeat
            print(f"Warning: No downbeats explicitly found for {audio_path}, using first beat as start of first bar.")
            downbeat_times_sec = np.array([beat_times_sec[0]])


        # --- Source Separation (Simulated or using pre-separated) ---
        separated_stems = {}
        if stems_input_dir and os.path.isdir(stems_input_dir):
            print(f"Attempting to load pre-separated stems from: {stems_input_dir}")
            separated_stems = self._separate_sources(audio_path, stems_input_dir) # This will load if files exist
        else:
            # This part is mostly a stub unless you implement actual separation
            # For simulation, it might try to create a cache dir based on audio_path
            # For now, if stems_input_dir is not provided or invalid, separated_stems will be empty.
            print(f"STUB: Simulating source separation (no valid stems_input_dir: {stems_input_dir}).")
            # If you had a default cache location for on-the-fly separation:
            # temp_stems_output_dir = os.path.join("data", "temp_demucs_separated", "htdemucs_ft", os.path.splitext(os.path.basename(audio_path))[0])
            # separated_stems = self._separate_sources(audio_path, temp_stems_output_dir)


        y_drums = separated_stems.get("drums")
        y_bass = separated_stems.get("bass")

        if y_drums is None: print(f"  Warning: Drum stem not available for {audio_path}. Drum features will be empty/simulated from full mix.")
        if y_bass is None: print(f"  Warning: Bass stem not available for {audio_path}. Bass features will be empty/simulated from full mix.")


        song_features: SongBeatFeatures = []

        for i in range(len(downbeat_times_sec)):
            bar_start_time = downbeat_times_sec[i]
            bar_end_time = downbeat_times_sec[i+1] if (i + 1) < len(downbeat_times_sec) else beat_times_sec[-1] + (beat_times_sec[-1] - beat_times_sec[-2] if len(beat_times_sec) > 1 else 2.0) # Estimate last bar end
            bar_duration = bar_end_time - bar_start_time

            if bar_duration <= 0: continue # Skip empty or invalid bars

            # Get beats relevant to the current bar for onset simulation context
            current_bar_beat_indices = np.where((beat_times_sec >= bar_start_time) & (beat_times_sec < bar_end_time))[0]
            current_bar_beat_times = beat_times_sec[current_bar_beat_indices]


            # --- Drum Transcription (Simulated) ---
            # Use drum stem if available, otherwise full mix for simulation
            kick_onsets_sec_np = self._get_simulated_instrument_onsets(y_drums if y_drums is not None else y_mono, "kick", current_bar_beat_times)
            snare_onsets_sec_np = self._get_simulated_instrument_onsets(y_drums if y_drums is not None else y_mono, "snare", current_bar_beat_times)
            hihat_onsets_sec_np = self._get_simulated_instrument_onsets(y_drums if y_drums is not None else y_mono, "hihat", current_bar_beat_times)
            
            # --- Bass Activity (Simulated) ---
            bass_onsets_sec_np = self._get_simulated_instrument_onsets(y_bass if y_bass is not None else y_mono, "bass", current_bar_beat_times)


            bar_feat: BarBeatFeatures = {
                "bar_index": i,
                "bpm": float(tempo), # Global tempo for the song
                "time_signature": (beats_per_bar, 4), # Assuming 4/4
                "kick_events": self._quantize_onsets_to_bar_subdivisions(kick_onsets_sec_np.tolist(), bar_start_time, bar_duration),
                "snare_events": self._quantize_onsets_to_bar_subdivisions(snare_onsets_sec_np.tolist(), bar_start_time, bar_duration),
                "hihat_events": self._quantize_onsets_to_bar_subdivisions(hihat_onsets_sec_np.tolist(), bar_start_time, bar_duration),
                "bass_events": self._quantize_onsets_to_bar_subdivisions(bass_onsets_sec_np.tolist(), bar_start_time, bar_duration),
                "bar_start_time_sec": bar_start_time, # For reference
                "bar_duration_sec": bar_duration   # For reference
            }
            song_features.append(bar_feat)
            
        return song_features


# Example usage:
if __name__ == "__main__":
    # This example assumes you have an audio file and optionally pre-separated stems.
    # Create a dummy audio file for testing if you don't have one.
    # import soundfile as sf
    # dummy_sr = 44100
    # dummy_duration = 20 # seconds
    # dummy_audio = np.random.rand(dummy_sr * dummy_duration) * 0.2 - 0.1
    # dummy_audio_path = "dummy_instrumental.wav"
    # sf.write(dummy_audio_path, dummy_audio, dummy_sr)
    
    # Create dummy stem files (replace with actual paths if you have them)
    # DUMMY_STEMS_DIR = "data/temp_demucs_separated/htdemucs_ft/dummy_instrumental" # Matches Demucs output structure
    # os.makedirs(DUMMY_STEMS_DIR, exist_ok=True)
    # sf.write(os.path.join(DUMMY_STEMS_DIR, "drums.wav"), dummy_audio[:dummy_sr*10]*0.5, dummy_sr) # Shorter dummy stem
    # sf.write(os.path.join(DUMMY_STEMS_DIR, "bass.wav"), dummy_audio[dummy_sr*5:dummy_sr*15]*0.4, dummy_sr)

    extractor = BeatFeatureExtractor()
    
    # --- Test Case 1: Audio file exists, but no specific stems_input_dir provided ---
    # This will rely purely on the _get_simulated_instrument_onsets using the full mix.
    print("\n--- Test Case 1: No specific stems_input_dir (simulated from full mix) ---")
    # Make sure this path is correct for your test file
    # test_audio_file = "data/instrumentals/Alright.mp3" # Example
    test_audio_file = "dummy_instrumental.wav" # Use the dummy if created
    if not os.path.exists(test_audio_file):
        print(f"Test audio file {test_audio_file} not found. Skipping Test Case 1.")
    else:
        song_features_case1 = extractor.extract_features_for_song(test_audio_file, stems_input_dir=None)
        if song_features_case1:
            print(f"Extracted {len(song_features_case1)} bars of features for {os.path.basename(test_audio_file)} (Case 1).")
            # print("First bar features (Case 1):", song_features_case1[0])
        else:
            print(f"Could not extract features for {os.path.basename(test_audio_file)} (Case 1).")

    # --- Test Case 2: Audio file exists, AND stems_input_dir is provided and valid ---
    # This will try to load drums.wav and bass.wav from DUMMY_STEMS_DIR.
    # If found, _get_simulated_instrument_onsets will use these stems.
    # If not found, it falls back to using the full mix (y_mono).
    print("\n--- Test Case 2: With specific stems_input_dir ---")
    # test_stems_dir = DUMMY_STEMS_DIR # Use the dummy stems dir
    # if not os.path.exists(test_audio_file) or not os.path.isdir(test_stems_dir):
    #     print(f"Test audio file {test_audio_file} or stems_dir {test_stems_dir} not found/valid. Skipping Test Case 2.")
    # else:
    #     song_features_case2 = extractor.extract_features_for_song(test_audio_file, stems_input_dir=test_stems_dir)
    #     if song_features_case2:
    #         print(f"Extracted {len(song_features_case2)} bars of features for {os.path.basename(test_audio_file)} (Case 2).")
    #         # print("First bar features (Case 2):", song_features_case2[0])
    #     else:
    #         print(f"Could not extract features for {os.path.basename(test_audio_file)} (Case 2).")

    # Cleanup dummy file if created
    # if os.path.exists(dummy_audio_path): os.remove(dummy_audio_path)
    # if os.path.exists(DUMMY_STEMS_DIR): import shutil; shutil.rmtree(DUMMY_STEMS_DIR, ignore_errors=True)