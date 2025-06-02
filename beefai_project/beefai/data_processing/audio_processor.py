import librosa
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import signal # For hann window if needed by other functions

from beefai.utils.data_types import AudioData, BeatInfo

class AudioProcessor:
    def __init__(self, default_sr: int = 44100): # Default SR for internal processing if not specified
        self.default_sr = default_sr
        print("AudioProcessor initialized.")

    def load_audio(self, file_path: str, target_sr: Optional[int] = None) -> AudioData:
        """
        Loads an audio file.

        Args:
            file_path: Path to the audio file.
            target_sr: Optional target sample rate to resample to. 
                       If None, loads at original sample rate.

        Returns:
            A tuple (waveform, sample_rate), where waveform is a NumPy array
            and sample_rate is the audio's sample rate (which will be target_sr if specified).
            Returns (empty array, 0) if loading fails.
        """
        try:
            # librosa.load can take 'sr=None' to load original, or 'sr=target_sr' to load and resample.
            # If target_sr is not provided, it will load at the original sample rate.
            # If target_sr is provided, it will resample to target_sr.
            waveform, sr = librosa.load(file_path, sr=target_sr, mono=True) # Ensure mono for consistency
            actual_sr = sr # librosa.load returns the sample rate of the loaded audio (which is target_sr if specified)
            # print(f"Audio loaded from {file_path}. Shape: {waveform.shape}, SR: {actual_sr}")
            return waveform, actual_sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return np.array([], dtype=np.float32), 0


    def get_beat_info(self, waveform: np.ndarray, sr: int) -> BeatInfo:
        """
        Extracts beat information (BPM, beat times, downbeat times) from an audio waveform.
        """
        if waveform.size == 0 or sr <= 0:
            print("Error: Empty waveform or invalid sample rate provided to get_beat_info.")
            return {
                "bpm": 0.0, "beat_times": [], "downbeat_times": [],
                "estimated_bar_duration": 0.0, "beats_per_bar": 0
            }

        try:
            # Estimate tempo and beat frames
            # tempo is a single float, beats are frame indices
            tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr, units='frames')
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            if not beat_times.any() or tempo <= 0:
                print("Warning: Beat tracking failed to find beats or tempo.")
                return {
                    "bpm": float(tempo) if tempo > 0 else 120.0, # fallback BPM
                    "beat_times": beat_times.tolist() if beat_times.any() else [],
                    "downbeat_times": [],
                    "estimated_bar_duration": 4 * (60.0 / (tempo if tempo > 0 else 120.0)), # fallback
                    "beats_per_bar": 4 # fallback
                }

            # Estimate downbeats (simplistic: assumes 4/4 time and tempo is fairly constant)
            # We'll assume beats_per_bar is 4 for this simple estimation.
            # A more robust system would detect time signature or use a more advanced downbeat tracker.
            beats_per_bar = 4 
            
            # Estimate bar duration based on tempo and assumed beats_per_bar
            estimated_bar_duration = beats_per_bar * (60.0 / tempo)

            # Simple downbeat estimation: pick every Nth beat_time as a downbeat
            # This is often not very accurate for complex music.
            downbeat_frames = beat_frames[::beats_per_bar] # Take every `beats_per_bar`-th beat frame
            downbeat_times = librosa.frames_to_time(downbeat_frames, sr=sr)
            
            # A basic check: if the first beat isn't close to the first downbeat, adjust.
            if len(downbeat_times) > 0 and len(beat_times) > 0 and not np.isclose(downbeat_times[0], beat_times[0], atol=0.1):
                # Try to align downbeats better if simple slicing is off, e.g., find beat closest to start
                # This is still heuristic. For real applications, a dedicated downbeat tracker or
                # manual annotation / more sophisticated beat analysis (e.g. madmom) is better.
                # For now, let's just use the sliced version.
                pass


            return {
                "bpm": float(tempo),
                "beat_times": beat_times.tolist(),
                "downbeat_times": downbeat_times.tolist(),
                "estimated_bar_duration": float(estimated_bar_duration),
                "beats_per_bar": beats_per_bar
            }
        except Exception as e:
            print(f"Error during beat analysis: {e}")
            # Return a default/empty BeatInfo structure on error
            return {
                "bpm": 0.0, "beat_times": [], "downbeat_times": [],
                "estimated_bar_duration": 0.0, "beats_per_bar": 0
            }

# Example usage (illustrative)
if __name__ == "__main__":
    processor = AudioProcessor()
    
    # Create a dummy audio file for testing
    dummy_audio_path = "temp_dummy_audio.wav"
    if not os.path.exists(dummy_audio_path):
        print(f"Creating dummy audio file: {dummy_audio_path}")
        try:
            import soundfile as sf
            sr_test = 44100
            duration_test = 5 # seconds
            t_test = np.linspace(0, duration_test, int(sr_test * duration_test), endpoint=False)
            # A simple tone that librosa can hopefully get a beat from
            y_test = 0.5 * np.sin(2 * np.pi * 220 * t_test) # A3 note
            # Add some rhythmic pulses (simulating beats at ~120 BPM)
            for i in range(int(duration_test * 2)): # 2 pulses per second = 120 BPM
                 pulse_time = i * 0.5
                 if pulse_time + 0.1 < duration_test:
                      y_test[int(pulse_time*sr_test) : int((pulse_time+0.05)*sr_test)] += 0.3 * np.random.rand(len(y_test[int(pulse_time*sr_test) : int((pulse_time+0.05)*sr_test)]))
            sf.write(dummy_audio_path, y_test, sr_test)
        except ImportError:
            print("Please install soundfile to run this example: pip install soundfile")
        except Exception as e_create:
            print(f"Error creating dummy audio: {e_create}")


    if os.path.exists(dummy_audio_path):
        print(f"\nTesting load_audio with original SR from {dummy_audio_path}:")
        waveform_orig, sr_orig = processor.load_audio(dummy_audio_path) # target_sr=None
        if waveform_orig.size > 0:
            print(f"  Loaded original. Waveform shape: {waveform_orig.shape}, SR: {sr_orig}")

            print(f"\nTesting beat info extraction (original SR):")
            beat_info_orig = processor.get_beat_info(waveform_orig, sr_orig)
            print(f"  BPM: {beat_info_orig['bpm']:.2f}")
            print(f"  Num Beats: {len(beat_info_orig['beat_times'])}")
            print(f"  Num Downbeats: {len(beat_info_orig['downbeat_times'])}")
            print(f"  First 5 beat times: {beat_info_orig['beat_times'][:5]}")
            print(f"  First 3 downbeat times: {beat_info_orig['downbeat_times'][:3]}")

        print(f"\nTesting load_audio with target_sr=22050 from {dummy_audio_path}:")
        waveform_resampled, sr_resampled = processor.load_audio(dummy_audio_path, target_sr=22050)
        if waveform_resampled.size > 0:
            print(f"  Loaded and resampled. Waveform shape: {waveform_resampled.shape}, SR: {sr_resampled}")
            
            # Test beat info on resampled audio too
            print(f"\nTesting beat info extraction (resampled SR={sr_resampled}):")
            beat_info_resampled = processor.get_beat_info(waveform_resampled, sr_resampled)
            print(f"  BPM: {beat_info_resampled['bpm']:.2f}")
            print(f"  Num Beats: {len(beat_info_resampled['beat_times'])}")

        # Clean up dummy file
        # os.remove(dummy_audio_path)
        # print(f"\nRemoved dummy audio file: {dummy_audio_path}")
    else:
        print(f"Dummy audio file {dummy_audio_path} not found or not created, skipping tests.")