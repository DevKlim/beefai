# beefai/data_processing/audio_processor.py
import librosa
import numpy as np
from typing import Optional, Tuple, List, Dict 
import soundfile as sf
import os
import sys

from beefai.utils.data_types import AudioData, BeatInfo


class AudioProcessor:
    def __init__(self, default_sample_rate: int = 44100):
        self.default_sample_rate = default_sample_rate
        # print("AudioProcessor initialized.")

    def load_audio(self, 
                   file_path: str, 
                   target_sr: Optional[int] = None, 
                   mono: bool = False, # Default is False here
                   offset_sec: Optional[float] = None,    
                   duration_sec: Optional[float] = None  
                  ) -> Optional[AudioData]:
        if not os.path.exists(file_path):
            print(f"[AudioProcessor] Error: File not found at {file_path}")
            return None
        
        effective_sr = target_sr if target_sr is not None else self.default_sample_rate
        
        try:
            waveform, sr = librosa.load(
                file_path, 
                sr=effective_sr, 
                mono=mono, # User can request mono loading here
                offset=offset_sec if offset_sec is not None else 0.0,
                duration=duration_sec
            )
            if waveform.size == 0: 
                print(f"[AudioProcessor] Warning: Loaded audio from {file_path} is empty (offset={offset_sec}, duration={duration_sec}).")
                return None 
            return waveform, sr
        except Exception as e:
            print(f"[AudioProcessor] Error loading audio file {file_path}: {e}")
            return None

    def save_audio(self, waveform: np.ndarray, sr: int, file_path: str):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            sf.write(file_path, waveform, sr)
        except Exception as e:
            print(f"[AudioProcessor] Error saving audio to {file_path}: {e}")

    def get_beat_info(self, waveform: np.ndarray, sr: int, beats_per_bar_hint: int = 4) -> BeatInfo:
        if waveform is None or waveform.size == 0:
            return {
                "bpm": 120.0, "beat_times": [], "downbeat_times": [],
                "estimated_bar_duration": 2.0, "beats_per_bar": 4
            }

        try:
            # --- FIX: Ensure waveform is mono before passing to beat_track ---
            if waveform.ndim > 1: # If stereo or more channels
                # print("[AudioProcessor] get_beat_info: Converting stereo waveform to mono for beat analysis.")
                y_mono = librosa.to_mono(waveform)
            else:
                y_mono = waveform
            # --- END FIX ---

            # Now use y_mono for beat tracking
            tempo_scalar, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr, hop_length=512, units='frames')
            
            bpm_val: float
            if isinstance(tempo_scalar, np.ndarray) and tempo_scalar.size > 0:
                bpm_val = float(tempo_scalar[0]) 
            elif isinstance(tempo_scalar, (float, int)):
                bpm_val = float(tempo_scalar)
            else: 
                bpm_val = 120.0 
            
            if bpm_val <= 0: bpm_val = 120.0 

            beat_times_sec = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

            downbeat_times_sec = []
            if len(beat_times_sec) >= beats_per_bar_hint:
                for i in range(0, len(beat_times_sec), beats_per_bar_hint):
                    downbeat_times_sec.append(beat_times_sec[i])

            estimated_bar_duration = 0.0
            if bpm_val > 0 and beats_per_bar_hint > 0:
                estimated_bar_duration = (60.0 / bpm_val) * beats_per_bar_hint
            elif len(downbeat_times_sec) > 1: 
                estimated_bar_duration = np.mean(np.diff(downbeat_times_sec))

            return {
                "bpm": round(bpm_val, 2),
                "beat_times": beat_times_sec.tolist() if isinstance(beat_times_sec, np.ndarray) else beat_times_sec,
                "downbeat_times": downbeat_times_sec,
                "estimated_bar_duration": round(estimated_bar_duration, 3) if estimated_bar_duration else 0.0,
                "beats_per_bar": beats_per_bar_hint
            }
        except Exception as e:
            print(f"[AudioProcessor] Error during beat analysis: {e}")
            return {
                "bpm": 120.0, "beat_times": [], "downbeat_times": [],
                "estimated_bar_duration": 2.0, "beats_per_bar": 4
            }


if __name__ == '__main__':
    processor = AudioProcessor()
    
    dummy_sr = 22050
    dummy_duration = 15 
    dummy_freq = 440
    t = np.linspace(0, dummy_duration, int(dummy_sr * dummy_duration), endpoint=False)
    
    # Create a stereo dummy audio
    dummy_audio_mono = 0.5 * np.sin(2 * np.pi * dummy_freq * t)
    dummy_audio_stereo = np.vstack((dummy_audio_mono * 0.8, dummy_audio_mono * 0.6)) # 2 channels
    
    dummy_dir = "temp_audio_processor_test"
    os.makedirs(dummy_dir, exist_ok=True)
    dummy_file_path_stereo = os.path.join(dummy_dir, "dummy_audio_stereo.wav")
    # Save as stereo (soundfile handles multi-channel if data is (samples, channels) or (channels, samples))
    # librosa load usually returns (channels, samples) if not mono, so let's save it like that
    processor.save_audio(dummy_audio_stereo.T, dummy_sr, dummy_file_path_stereo) # Transpose for (samples, channels)
    print(f"Saved dummy stereo audio to {dummy_file_path_stereo}")

    print("\n--- Test 1: Load stereo audio (as stereo) and get beat info ---")
    # Load explicitly as stereo (mono=False is default)
    loaded_stereo = processor.load_audio(dummy_file_path_stereo, target_sr=44100, mono=False) 
    if loaded_stereo:
        wf_stereo, sr_stereo = loaded_stereo
        print(f"Loaded stereo: Waveform shape {wf_stereo.shape}, Sample rate {sr_stereo}")
        assert wf_stereo.ndim == 2, "Loaded waveform should be stereo"
        beat_info_stereo = processor.get_beat_info(wf_stereo, sr_stereo) # This should now work
        print(f"Beat info (from stereo input, processed as mono): {beat_info_stereo}")
        assert beat_info_stereo["bpm"] > 0, "BPM should be positive"
    else:
        print("Failed to load stereo audio.")

    print("\n--- Test 2: Load stereo audio (requesting mono) and get beat info ---")
    loaded_as_mono = processor.load_audio(dummy_file_path_stereo, target_sr=44100, mono=True)
    if loaded_as_mono:
        wf_mono, sr_mono = loaded_as_mono
        print(f"Loaded as mono: Waveform shape {wf_mono.shape}, Sample rate {sr_mono}")
        assert wf_mono.ndim == 1, "Loaded waveform should be mono"
        beat_info_mono = processor.get_beat_info(wf_mono, sr_mono)
        print(f"Beat info (from mono input): {beat_info_mono}")
        assert beat_info_mono["bpm"] > 0, "BPM should be positive"

    else:
        print("Failed to load audio as mono.")


    import shutil
    if os.path.exists(dummy_dir):
        shutil.rmtree(dummy_dir)
        print(f"\nCleaned up {dummy_dir}")