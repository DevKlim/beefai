import librosa
import numpy as np
from typing import List, Tuple, Dict, Optional
from beefai.utils.data_types import BarBeatFeatures, SongBeatFeatures, AudioData
import os
import shutil # For copying files in simulation

# --- Source Separation Placeholder ---
def separate_sources(audio_path: str, output_dir_for_stems: str) -> Dict[str, str]:
    """
    **Placeholder for Source Separation (e.g., using Demucs).**
    
    A real implementation would:
    1. Ensure Demucs (or chosen tool) is installed.
    2. Call Demucs to process `audio_path` and save stems into `output_dir_for_stems`.
       Example Demucs CLI call (from Python using subprocess):
       `python -m demucs --two-stems=bass --two-stems=drums -n htdemucs_ft "{audio_path}" -o "{output_dir_for_stems}"`
       (Note: `htdemucs_ft` is a good general model. Adjust as needed.)
    3. Identify the paths to the 'bass' and 'drums' stems within Demucs' output structure
       (typically `output_dir_for_stems/htdemucs_ft/song_name/bass.wav` etc.).

    This placeholder simulates this by copying the original audio to stand in for stems
    if they don't already exist (e.g., from a previous real run).
    """
    song_base_name = os.path.splitext(os.path.basename(audio_path))[0]
    # Demucs typically creates a subdirectory with the model name, then song name
    # Simulate a common Demucs output structure for htdemucs_ft
    simulated_demucs_output_path = os.path.join(output_dir_for_stems, "htdemucs_ft", song_base_name)
    os.makedirs(simulated_demucs_output_path, exist_ok=True)

    stems_to_produce = {
        "bass": os.path.join(simulated_demucs_output_path, "bass.wav"),
        "drums": os.path.join(simulated_demucs_output_path, "drums.wav"),
        # "vocals": os.path.join(simulated_demucs_output_path, "vocals.wav"), # For MFA
        # "other": os.path.join(simulated_demucs_output_path, "other.wav"),
    }
    
    print(f"[BeatFeatureExtractor - SIMULATION] Source separation for: {audio_path}")
    print(f"[BeatFeatureExtractor - SIMULATION] Stems expected in: {simulated_demucs_output_path}")

    if not os.path.exists(audio_path):
        print(f"  [SIMULATION-ERROR] Original audio not found: {audio_path}. Cannot simulate stems.")
        return {key: "" for key in stems_to_produce} # Return empty paths

    # Check if actual stems exist (e.g., user ran Demucs separately)
    # If not, copy original audio as a placeholder for bass and drums.
    for stem_name, stem_path in stems_to_produce.items():
        if os.path.exists(stem_path):
            print(f"  [SIMULATION] Found existing stem: {stem_path} (perhaps from a real run).")
        else:
            if stem_name in ["bass", "drums"]: # Only simulate for bass and drums for BFE
                try:
                    shutil.copy(audio_path, stem_path)
                    print(f"  [SIMULATION] Copied original audio to '{stem_path}' as placeholder for '{stem_name}'.")
                except Exception as e:
                    print(f"  [SIMULATION-ERROR] Could not copy original audio to {stem_path}: {e}")
                    stems_to_produce[stem_name] = "" # Mark as unavailable
            else:
                print(f"  [SIMULATION] Placeholder for '{stem_name}' not created by this simulation (path: {stem_path}).")
                # stems_to_produce[stem_name] = "" # Or create empty files if needed by downstream

    return stems_to_produce

# --- Drum Transcription Placeholder ---
def transcribe_drum_stem(drum_stem_path: str, sr: int, bar_start_time: float, bar_duration: float) -> Dict[str, np.ndarray]:
    """
    **Placeholder for Transcribing a Drum Stem into Kick, Snare, Hi-Hat Onsets WITHIN A GIVEN BAR.**
    
    A real implementation would use advanced techniques:
    - Spectrogram analysis, NMF, or machine learning models (e.g., from Essentia, Madmom, custom CNNs)
      to identify and classify drum hits.
    
    This placeholder generates some plausible (but random or simplistic) onsets if a drum stem is found.
    It returns onset times *relative to the start of the audio file*, not bar-relative.
    Quantization happens later.
    """
    # print(f"[BeatFeatureExtractor - SIMULATION] Drum transcription for: {drum_stem_path} (bar: {bar_start_time:.2f}s, dur: {bar_duration:.2f}s)")
    
    if not os.path.exists(drum_stem_path):
        # print(f"  [SIMULATION] Drum stem not found: {drum_stem_path}. No drum events generated.")
        return {"kick": np.array([]), "snare": np.array([]), "hihat": np.array([])}

    try:
        y_drum_full, sr_drum = librosa.load(drum_stem_path, sr=sr, mono=True)
        if sr_drum != sr: y_drum_full = librosa.resample(y_drum_full, orig_sr=sr_drum, target_sr=sr)
    except Exception as e:
        print(f"  [SIMULATION-ERROR] Could not load drum stem {drum_stem_path}: {e}")
        return {"kick": np.array([]), "snare": np.array([]), "hihat": np.array([])}

    # Extract the segment of the drum stem corresponding to the current bar
    bar_start_sample = int(bar_start_time * sr)
    bar_end_sample = int((bar_start_time + bar_duration) * sr)
    y_drum_bar = y_drum_full[bar_start_sample : min(bar_end_sample, len(y_drum_full))]

    if len(y_drum_bar) < sr * 0.1: # If bar segment is too short
        # print("  [SIMULATION] Drum bar segment too short for onset detection.")
        return {"kick": np.array([]), "snare": np.array([]), "hihat": np.array([])}

    # Simulate some onsets within this bar segment
    # These are relative to the start of y_drum_bar (i.e., relative to bar_start_time)
    onsets_kick_bar = librosa.onset.onset_detect(y=y_drum_bar, sr=sr, units='time', hop_length=128, backtrack=False, delta=0.3, wait=5)
    onsets_snare_bar = librosa.onset.onset_detect(y=y_drum_bar, sr=sr, units='time', hop_length=128, backtrack=False, delta=0.35, wait=7)
    onsets_hihat_bar = librosa.onset.onset_detect(y=y_drum_bar, sr=sr, units='time', hop_length=64, backtrack=False, delta=0.2, wait=2)
    
    # Convert bar-relative onsets to absolute time (relative to start of audio_path)
    onsets_kick_abs = onsets_kick_bar + bar_start_time
    onsets_snare_abs = onsets_snare_bar + bar_start_time
    onsets_hihat_abs = onsets_hihat_bar + bar_start_time
    
    # Simulate distinct patterns (very crudely)
    # A more realistic simulation might place kicks on strong beats, snares on 2/4, etc.
    # For now, just take subsets of detected onsets.
    return {
        "kick": np.random.choice(onsets_kick_abs, size=min(len(onsets_kick_abs), np.random.randint(1,5)), replace=False) if len(onsets_kick_abs)>0 else np.array([]),
        "snare": np.random.choice(onsets_snare_abs, size=min(len(onsets_snare_abs), np.random.randint(0,3)), replace=False) if len(onsets_snare_abs)>0 else np.array([]),
        "hihat": np.random.choice(onsets_hihat_abs, size=min(len(onsets_hihat_abs), np.random.randint(0,9)), replace=False) if len(onsets_hihat_abs)>0 else np.array([]),
    }


class BeatFeatureExtractor:
    def __init__(self, sample_rate: int = 44100, subdivisions_per_bar: int = 16):
        self.sample_rate = sample_rate
        self.subdivisions_per_bar = subdivisions_per_bar

    def _quantize_onsets_to_subdivisions(self, onset_times: np.ndarray, bar_start_time: float, bar_duration: float) -> List[int]:
        if bar_duration <= 0.01: return [] # Avoid division by zero for very short/invalid bars
        active_subdivisions = set()
        for onset_time in onset_times:
            # Check if onset falls within the bar (relative to audio start)
            if bar_start_time <= onset_time < (bar_start_time + bar_duration):
                time_in_bar = onset_time - bar_start_time
                normalized_time = time_in_bar / bar_duration
                subdivision_index = int(normalized_time * self.subdivisions_per_bar)
                subdivision_index = max(0, min(subdivision_index, self.subdivisions_per_bar - 1))
                active_subdivisions.add(subdivision_index)
        return sorted(list(active_subdivisions))

    def _extract_features_for_one_song(self, 
                                      instrumental_audio_path: str, 
                                      bass_stem_path: Optional[str], 
                                      drums_stem_path: Optional[str]
                                      ) -> Optional[SongBeatFeatures]:
        try:
            y_instrumental, sr_orig = librosa.load(instrumental_audio_path, sr=None, mono=True)
            if sr_orig != self.sample_rate:
                y_instrumental = librosa.resample(y_instrumental, orig_sr=sr_orig, target_sr=self.sample_rate)
            
            y_bass = None
            if bass_stem_path and os.path.exists(bass_stem_path):
                y_bass, _ = librosa.load(bass_stem_path, sr=self.sample_rate, mono=True)
            else:
                print(f"  Warning: Bass stem not found or path not provided ({bass_stem_path}). Bass events will be empty.")
        except Exception as e:
            print(f"BeatFeatureExtractor: Error loading audio for feature extraction: {e}")
            return None

        # 1. Global Tempo and Beats/Downbeats from the full instrumental
        try:
            tempo_values, beat_frames = librosa.beat.beat_track(y=y_instrumental, sr=self.sample_rate, trim=False, hop_length=512)
            tempo = float(np.median(tempo_values)) if isinstance(tempo_values, np.ndarray) and tempo_values.size > 0 else float(tempo_values)
            beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate, hop_length=512)
            time_signature_tuple = (4, 4) # Assume 4/4, robust estimation is complex
            beats_per_bar_est = time_signature_tuple[0]

            if len(beat_times) > 0:
                downbeat_indices = np.arange(0, len(beat_times), beats_per_bar_est)
                downbeat_times = beat_times[downbeat_indices]
            else: # No beats detected, cannot proceed
                print(f"  No beats detected in {instrumental_audio_path}. Cannot extract beat features.")
                return None
        except Exception as e:
            print(f"BeatFeatureExtractor: Error in librosa beat tracking for {instrumental_audio_path}: {e}")
            return None
        
        if len(downbeat_times) == 0: # If still no downbeats (e.g. very short audio)
            if tempo > 0 and len(beat_times) > 0: # Try to synthesize if beats exist
                 bar_duration_est = beats_per_bar_est * (60.0 / tempo)
                 total_duration_audio = librosa.get_duration(y=y_instrumental, sr=self.sample_rate)
                 num_bars_synth = int(np.ceil(total_duration_audio / bar_duration_est))
                 if num_bars_synth > 0:
                    downbeat_times = np.array([beat_times[0] + i * bar_duration_est for i in range(num_bars_synth)])
                 else: # Not even enough for one synthetic bar
                    print(f"  Audio too short or no beats to synthesize downbeats for {instrumental_audio_path}.")
                    return None
            else:
                print(f"  No downbeats could be determined for {instrumental_audio_path}.")
                return None

        song_features: SongBeatFeatures = []
        num_bars = len(downbeat_times)

        for i in range(num_bars):
            bar_start_time = downbeat_times[i]
            bar_end_time = downbeat_times[i+1] if (i + 1) < num_bars else (bar_start_time + (beats_per_bar_est * (60.0/tempo)))
            bar_duration = bar_end_time - bar_start_time
            if bar_duration <= 0.01: continue

            # Drum events (simulated per bar)
            drum_events_this_bar = transcribe_drum_stem(drums_stem_path, self.sample_rate, bar_start_time, bar_duration) if drums_stem_path else {"kick":[], "snare":[], "hihat":[]}
            
            # Bass onsets (from full bass stem, then filtered per bar)
            bass_onsets_all = []
            if y_bass is not None:
                 bass_onsets_all = librosa.onset.onset_detect(y=y_bass, sr=self.sample_rate, units='time', hop_length=256, backtrack=False)
            
            bar_feature: BarBeatFeatures = {
                "bar_index": i, "bpm": tempo, "time_signature": time_signature_tuple,
                "kick_events": self._quantize_onsets_to_subdivisions(drum_events_this_bar.get("kick", np.array([])), bar_start_time, bar_duration),
                "snare_events": self._quantize_onsets_to_subdivisions(drum_events_this_bar.get("snare", np.array([])), bar_start_time, bar_duration),
                "hihat_events": self._quantize_onsets_to_subdivisions(drum_events_this_bar.get("hihat", np.array([])), bar_start_time, bar_duration),
                "bass_events": self._quantize_onsets_to_subdivisions(np.array(bass_onsets_all), bar_start_time, bar_duration)
            }
            song_features.append(bar_feature)
        
        return song_features

    def extract_song_beat_features(self, audio_path: str, stems_output_dir: Optional[str] = None) -> Optional[SongBeatFeatures]:
        if not os.path.exists(audio_path):
            print(f"BeatFeatureExtractor: Audio file not found at {audio_path}")
            return None

        if stems_output_dir is None:
            stems_output_dir = os.path.join(os.path.dirname(audio_path), f"{os.path.splitext(os.path.basename(audio_path))[0]}_stems_data")
        ensure_dir(stems_output_dir) # Ensure the base output dir for stems exists
        
        # Call source separation (placeholder). It will create subdirs based on song name.
        stem_paths = separate_sources(audio_path, stems_output_dir)

        bass_path = stem_paths.get("bass")
        drums_path = stem_paths.get("drums")

        # Check if simulation produced valid (even if copied) paths
        if not bass_path or not os.path.exists(bass_path):
             print(f"  Warning: Bass stem not found or not simulated properly from {audio_path} at expected path {bass_path}. Bass events will be empty.")
             bass_path = None # Set to None so _extract_features_for_one_song handles it
        if not drums_path or not os.path.exists(drums_path):
             print(f"  Warning: Drums stem not found or not simulated properly from {audio_path} at expected path {drums_path}. Drum events will be empty.")
             drums_path = None

        return self._extract_features_for_one_song(audio_path, bass_path, drums_path)

def ensure_dir(directory_path: str): # Helper, also in preprocess_dataset.py
    os.makedirs(directory_path, exist_ok=True)

if __name__ == '__main__':
    sr_test = 44100
    duration_test = 10 
    test_bfe_dummy_file = "dummy_bfe_standalone_test.wav"

    if not os.path.exists(test_bfe_dummy_file):
        print(f"Creating dummy audio for BFE standalone test: {test_bfe_dummy_file}")
        t_test = np.linspace(0, duration_test, int(sr_test * duration_test), endpoint=False)
        bpm_test = 120
        beat_interval_samples = int(sr_test / (bpm_test / 60.0))
        dummy_wav_test = np.zeros_like(t_test)
        for k_idx in range(0, len(dummy_wav_test), beat_interval_samples): # Kicks on beats
            if k_idx + 256 < len(dummy_wav_test): dummy_wav_test[k_idx : k_idx + 256] += 0.6 * np.sin(np.linspace(0, np.pi*4, 256))
        for s_idx in range(beat_interval_samples, len(dummy_wav_test), beat_interval_samples * 2): # Snares on 2 & 4
            if s_idx + 512 < len(dummy_wav_test): dummy_wav_test[s_idx : s_idx + 512] += 0.4 * (np.random.rand(512) - 0.5)
        
        import soundfile as sf
        sf.write(test_bfe_dummy_file, dummy_wav_test.astype(np.float32), sr_test)
        print(f"Created {test_bfe_dummy_file}")

    extractor = BeatFeatureExtractor(sample_rate=sr_test)
    # Output stems to a subdir of where this test script is run
    test_stems_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bfe_standalone_test_output")
    
    extracted_features = extractor.extract_song_beat_features(test_bfe_dummy_file, stems_output_dir=test_stems_dir)

    if extracted_features:
        print(f"\nSuccessfully extracted features for {len(extracted_features)} bars from {test_bfe_dummy_file}.")
        for bar_idx, bar_data in enumerate(extracted_features[:2]): # Print first 2 bars
            print(f"  Bar {bar_idx}: BPM={bar_data['bpm']:.1f}, TS={bar_data['time_signature']}")
            print(f"    Kicks: {bar_data['kick_events']}, Snares: {bar_data['snare_events']}, HiHats: {bar_data['hihat_events']}, Bass: {bar_data['bass_events']}")
    else:
        print(f"\nFailed to extract features from {test_bfe_dummy_file}.")