import librosa
import numpy as np
from typing import List, Tuple
from beefai.utils.data_types import BeatInfo, AudioData

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def load_audio(self, audio_path: str) -> AudioData:
        """
        Loads an audio file.
        Returns:
            Tuple[np.ndarray, int]: Waveform and sample rate.
        """
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return waveform, sr
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return np.array([]), self.sample_rate

    def get_beat_info(self, waveform: np.ndarray, sr: int) -> BeatInfo:
        """
        Extracts beat information from an audio waveform, including BPM, beat times, and downbeat times.
        """
        if waveform.size == 0:
            return {"bpm": 0.0, "beat_times": [], "downbeat_times": [], "estimated_bar_duration": 0.0, "beats_per_bar": 4}

        try:
            # Use a tighter setting for beat tracking if possible, good for clear instrumentals
            tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr, trim=False, tightness=100)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            # Filter out beat_times that are too close together (e.g. < 50ms), could be errors
            if len(beat_times) > 1:
                min_beat_diff = 0.050 # 50 ms
                valid_beat_times = [beat_times[0]]
                for i in range(1, len(beat_times)):
                    if (beat_times[i] - beat_times[i-1]) > min_beat_diff:
                        valid_beat_times.append(beat_times[i])
                beat_times = np.array(valid_beat_times)
        
        except Exception as e:
            print(f"Error in librosa.beat.beat_track: {e}. Returning empty beat info.")
            return {"bpm": 0.0, "beat_times": [], "downbeat_times": [], "estimated_bar_duration": 0.0, "beats_per_bar": 4}

        downbeat_times = []
        estimated_bar_duration = 0.0
        beats_per_bar = 4 # Common assumption for rap music

        if tempo > 0 and len(beat_times) > 0:
            avg_beat_duration = 60.0 / tempo
            estimated_bar_duration = beats_per_bar * avg_beat_duration
            
            # Heuristic for downbeats: assume the first beat is a downbeat,
            # then find subsequent beats that are closest to N*bar_duration later.
            # This is sensitive to the quality of beat_times and tempo.
            
            # A simpler heuristic, if beat_times are reliable:
            # Assume the first beat_time is a downbeat.
            # Then, every 'beats_per_bar'-th beat in the beat_times array is a downbeat.
            # This works well if the beat tracking is very accurate and tempo is stable.
            if len(beat_times) > 0:
                # Try to align first beat with a musically significant point
                # For simplicity, we still use the first detected beat as a potential start.
                # A more advanced approach might involve onset strength or spectral analysis.
                
                # Heuristic: first beat_time is a downbeat candidate
                downbeat_times.append(beat_times[0])
                
                # Attempt to find subsequent downbeats based on bar duration
                # This is challenging because beat_track doesn't guarantee alignment with bar structure.
                # We'll stick to the Nth beat method for now for simplicity.
                current_downbeat_candidate_idx = 0
                while True:
                    next_potential_downbeat_idx = current_downbeat_candidate_idx + beats_per_bar
                    if next_potential_downbeat_idx < len(beat_times):
                        # Check if this Nth beat is reasonably spaced
                        # (i.e., tempo hasn't drifted wildly making this Nth beat too early/late for a bar)
                        expected_next_downbeat_time = beat_times[current_downbeat_candidate_idx] + estimated_bar_duration
                        actual_next_downbeat_time = beat_times[next_potential_downbeat_idx]
                        
                        # Tolerance: e.g., within 25% of a beat duration from expected
                        tolerance = avg_beat_duration * 0.35 
                        if abs(actual_next_downbeat_time - expected_next_downbeat_time) < tolerance :
                            downbeat_times.append(actual_next_downbeat_time)
                            current_downbeat_candidate_idx = next_potential_downbeat_idx
                        else:
                            # Try to find the *closest* beat to expected_next_downbeat_time
                            # search window: from next_potential_downbeat_idx -1 to +1 if they exist
                            search_indices = [idx for idx in 
                                              [next_potential_downbeat_idx -1, next_potential_downbeat_idx, next_potential_downbeat_idx +1]
                                              if 0 <= idx < len(beat_times) and idx > current_downbeat_candidate_idx]
                            
                            best_match_idx = -1
                            min_diff = float('inf')
                            for s_idx in search_indices:
                                diff = abs(beat_times[s_idx] - expected_next_downbeat_time)
                                if diff < tolerance and diff < min_diff :
                                    min_diff = diff
                                    best_match_idx = s_idx
                            
                            if best_match_idx != -1:
                                downbeat_times.append(beat_times[best_match_idx])
                                current_downbeat_candidate_idx = best_match_idx
                            else:
                                # print(f"  Could not find a reliable next downbeat after {beat_times[current_downbeat_candidate_idx]:.2f}s. Expected around {expected_next_downbeat_time:.2f}s.")
                                break # Stop if structure seems lost
                    else:
                        break # No more full bars possible
                
                downbeat_times = sorted(list(set(downbeat_times))) # Ensure uniqueness and order

        return {
            "bpm": float(tempo) if tempo is not None else 0.0,
            "beat_times": beat_times.tolist() if beat_times is not None else [],
            "downbeat_times": downbeat_times, # Use the refined list
            "estimated_bar_duration": round(estimated_bar_duration, 3),
            "beats_per_bar": beats_per_bar
        }

    def get_spectrogram(self, waveform: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Computes the Mel spectrogram for a waveform.
        """
        if waveform.size == 0:
            return np.array([])
        try:
            mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            return log_mel_spectrogram
        except Exception as e:
            print(f"Error computing spectrogram: {e}")
            return np.array([])

# Example Usage (for testing this module)
if __name__ == "__main__":
    # Test with the dummy instrumental from main.py
    # This requires running this script from the project root or adjusting path
    import os
    # Try to find the beefai_default_instrumental.wav in ../output relative to current file
    # This assumes data_processing is a subdir of beefai, and output is also a subdir of beefai
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # up one level to 'beefai'
    
    # Path for dummy instrumental, assuming it's generated by main.py in 'beefai/output/'
    # This test is a bit fragile due to path assumptions.
    # For more robust testing, it might be better to call create_dummy_instrumental here
    # or have a fixed test asset.
    
    # Re-create a dummy instrumental for testing AudioProcessor specifically
    DUMMY_TEST_SR = 22050
    DUMMY_TEST_BPM = 100.0
    DUMMY_TEST_DURATION = 15 # seconds
    
    test_waveform = np.zeros(int(DUMMY_TEST_SR * DUMMY_TEST_DURATION))
    test_beat_interval_samples = int(DUMMY_TEST_SR / (DUMMY_TEST_BPM / 60.0))

    for i in range(0, len(test_waveform), test_beat_interval_samples):
        if i + 256 < len(test_waveform): 
            test_waveform[i : i + 128] = 0.7 
            if (i // test_beat_interval_samples) % 4 == 0 : 
                 test_waveform[i : i + 256] = 1.0 # Stronger downbeats

    processor = AudioProcessor(sample_rate=DUMMY_TEST_SR)
    
    print("Testing AudioProcessor with generated synthetic waveform (100 BPM)...")
    beat_info = processor.get_beat_info(test_waveform, DUMMY_TEST_SR)
    print("\nBeat Info (Synthetic 100BPM):")
    print(f"  BPM: {beat_info.get('bpm')}")
    print(f"  Beats per bar: {beat_info.get('beats_per_bar')}")
    print(f"  Number of detected beats: {len(beat_info.get('beat_times', []))}")
    print(f"  Number of detected downbeats: {len(beat_info.get('downbeat_times', []))}")
    print(f"  First few downbeat times: {beat_info.get('downbeat_times', [])[:8]}")
    print(f"  Estimated bar duration: {beat_info.get('estimated_bar_duration')}")

    # Verify downbeat spacing for the 100 BPM test (expected bar duration: 4 * 60/100 = 2.4s)
    if beat_info.get('downbeat_times') and len(beat_info['downbeat_times']) > 1:
        diffs = np.diff(beat_info['downbeat_times'])
        print(f"  Downbeat time differences: {np.round(diffs[:7], 3)}")
        expected_bar_dur = beat_info.get('estimated_bar_duration', 2.4)
        if not np.allclose(diffs, expected_bar_dur, atol=0.1):
            print(f"  WARNING: Downbeat spacing ({np.mean(diffs):.3f}s avg) is not consistently close to expected bar duration ({expected_bar_dur:.3f}s).")

    # It's better if main.py, when run, calls its own create_dummy_instrumental
    # and then uses its own AudioProcessor instance for the game.
    # This test here is just for the AudioProcessor module itself.