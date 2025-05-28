import numpy as np
import time
from typing import List, Tuple, Optional
from beefai.utils.data_types import FlowData, FlowDatum, AudioData
# For real TTS/SVS:
# from TTS.api import TTS # Example: Coqui TTS
# import soundfile as sf # Already used for saving

class RapSynthesizer:
    def __init__(self, model_name_or_path: Optional[str] = None, voice_sample_path: Optional[str] = None):
        """
        Placeholder for the Text-to-Speech / Singing Voice Synthesis model.
        """
        self.model_name = model_name_or_path
        self.voice_sample_path = voice_sample_path
        self.tts_model = None
        self.sample_rate = 22050 # Default sample rate

        # try:
        #     if self.model_name:
        #         self.tts_model = TTS(model_name=self.model_name, progress_bar=False, gpu=False)
        #     else:
        #         print("RapSynthesizer initialized with placeholder. Specify a TTS model for real synthesis.")
        #     if self.tts_model and hasattr(self.tts_model, 'synthesizer') and hasattr(self.tts_model.synthesizer, 'output_sample_rate'):
        #          self.sample_rate = self.tts_model.synthesizer.output_sample_rate
        # except Exception as e:
        #     print(f"Could not load TTS model {self.model_name}. Synthesizer will output silence or simple tones. Error: {e}")
        #     self.tts_model = None
        print(f"RapSynthesizer initialized. (TTS integration placeholder for: {self.model_name or 'default TTS'})")


    def synthesize_line(self, lyric_line: str, flow_datum: FlowDatum) -> AudioData:
        """
        Synthesizes a single line of rap lyrics based on its flow characteristics.
        This is a placeholder. A real SVS system would take detailed prosody.
        """
        duration_sec = flow_datum.get("duration_sec", 2.0)
        pitch_contour_id = flow_datum.get("pitch_contour_id", "mid")

        print(f"  Synthesizing line: \"{lyric_line}\" (Duration: {duration_sec:.2f}s, Pitch: {pitch_contour_id})")

        if self.tts_model:
            # # Real SVS/TTS logic with duration and pitch control (complex)
            # try:
            #     # This is highly simplified. True rhythmic/pitch control needs specific model support.
            #     # Some TTS models allow specifying phoneme durations, or overall speed.
            #     # For pitch, some allow SSML or similar for basic contours.
            #     wav = self.tts_model.tts(text=lyric_line, speaker_wav=self.voice_sample_path, language='en')
            #     waveform = np.array(wav, dtype=np.float32)
                
            #     # Crude duration adjustment: time-stretching (can sound bad) or truncate/pad
            #     # Proper way is to have the model generate to the target duration.
            #     target_samples = int(duration_sec * self.sample_rate)
            #     if len(waveform) > 0 and target_samples > 0:
            #         if len(waveform) != target_samples:
            #             # Using librosa.effects.time_stretch is one option, but it changes pitch by default if not careful
            #             # For placeholder, simple truncate/pad
            #             if len(waveform) > target_samples:
            #                 waveform = waveform[:target_samples]
            #             else:
            #                 waveform = np.pad(waveform, (0, target_samples - len(waveform)), 'constant')
            #     elif target_samples == 0:
            #         waveform = np.array([], dtype=np.float32)

            #     return waveform, self.sample_rate
            # except Exception as e:
            #     print(f"    TTS synthesis failed for line '{lyric_line}': {e}. Falling back to placeholder tone.")
            pass # Actual TTS/SVS call

        # Placeholder: Generate a simple tone sequence or modulated noise
        num_samples = int(duration_sec * self.sample_rate)
        if num_samples <= 0: return np.array([], dtype=np.float32), self.sample_rate

        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        waveform = np.zeros(num_samples, dtype=np.float32)
        
        # Base frequency
        base_freq = 150 # Male-ish pitch range
        if "low" in pitch_contour_id: base_freq = 100
        elif "high" in pitch_contour_id: base_freq = 200
        
        # Pitch contour
        if "rising" in pitch_contour_id:
            final_freq = base_freq * 1.5
            current_freq = np.linspace(base_freq, final_freq, num_samples)
        elif "falling" in pitch_contour_id:
            final_freq = base_freq * 0.75
            current_freq = np.linspace(base_freq, final_freq, num_samples)
        else: # mid
            current_freq = np.full_like(t, base_freq)

        # Basic sine wave with harmonics to sound less pure
        waveform = 0.2 * np.sin(2 * np.pi * current_freq * t)
        waveform += 0.1 * np.sin(2 * np.pi * current_freq * 2 * t) # Add 1st harmonic
        waveform += 0.05 * np.sin(2 * np.pi * current_freq * 3 * t) # Add 2nd harmonic
        
        # Simple amplitude envelope (e.g., attack-decay)
        attack_len = min(num_samples // 20, int(0.05 * self.sample_rate)) # 50ms or 5% attack
        decay_len = num_samples - attack_len 
        
        if attack_len > 0 :
            envelope = np.concatenate([
                np.linspace(0., 1., attack_len),
                np.linspace(1., 0.3, decay_len) if decay_len > 0 else np.array([1.0])
            ])
            if len(envelope) < num_samples: # Pad if decay_len was 0
                envelope = np.pad(envelope, (0, num_samples - len(envelope)), 'constant', constant_values=0.3)
            elif len(envelope) > num_samples: # Truncate if sum too long
                 envelope = envelope[:num_samples]
            waveform *= envelope


        # Add some filtered noise to simulate fricatives/breath
        noise = 0.02 * np.random.randn(num_samples)
        # Simple low-pass filter for noise (convolution with a small window)
        # For simplicity, we'll just add it directly with less amplitude.
        waveform += noise

        # Ensure overall amplitude is reasonable
        if np.max(np.abs(waveform)) > 0:
             waveform = waveform / np.max(np.abs(waveform)) * 0.5 # Normalize to 0.5 max amplitude

        # Fade in/out to avoid clicks at segment boundaries (if not handled by envelope)
        fade_samples = min(num_samples // 20, int(0.01 * self.sample_rate)) # 10ms fade
        if fade_samples > 1: # Ensure fade_samples is large enough for linspace
            fade_in_curve = np.linspace(0., 1., fade_samples)
            fade_out_curve = np.linspace(1., 0., fade_samples)
            waveform[:fade_samples] *= fade_in_curve
            waveform[-fade_samples:] *= fade_out_curve
        
        return waveform.astype(np.float32), self.sample_rate


    def synthesize_verse(self, verse_lyrics: List[str], flow_data: FlowData) -> Optional[AudioData]:
        """
        Synthesizes a full rap verse by placing synthesized lines onto a timeline.
        """
        print("RapSynthesizer: Synthesizing full verse...")
        if not verse_lyrics or not flow_data or len(verse_lyrics) != len(flow_data):
            print("Error: Mismatch between lyrics and flow data, or empty input.")
            print(f"  Lyrics count: {len(verse_lyrics)}, Flow data count: {len(flow_data)}")
            return None

        # Determine total duration for the full verse waveform
        total_duration_sec = 0
        if flow_data:
            # Find the time when the last segment ends
            max_end_time = 0
            for fd in flow_data:
                segment_end_time = fd.get("start_time_sec", 0) + fd.get("duration_sec", 0)
                if segment_end_time > max_end_time:
                    max_end_time = segment_end_time
            total_duration_sec = max_end_time
        
        if total_duration_sec <= 0:
            print("Warning: Cannot determine total duration for synthesis from flow_data. Returning empty audio.")
            return np.array([], dtype=np.float32), self.sample_rate

        total_samples = int(total_duration_sec * self.sample_rate) + int(0.1 * self.sample_rate) # Add small buffer
        full_verse_waveform = np.zeros(total_samples, dtype=np.float32)

        for i, (lyric_line, flow_datum) in enumerate(zip(verse_lyrics, flow_data)):
            line_audio_waveform, sr = self.synthesize_line(lyric_line, flow_datum)
            
            if sr != self.sample_rate:
                print(f"Warning: Sample rate mismatch for line {i+1}. Expected {self.sample_rate}, got {sr}. Resampling (not implemented) or ignoring.")
                # Ideally, resample here: line_audio_waveform = librosa.resample(line_audio_waveform, orig_sr=sr, target_sr=self.sample_rate)
                # For now, we'll assume sample rates match or ignore if they don't.
                continue 
            
            if line_audio_waveform.size == 0:
                print(f"    Skipping empty audio for line {i+1}: \"{lyric_line}\"")
                continue

            start_time_sec = flow_datum.get("start_time_sec", 0)
            start_sample_index = int(start_time_sec * self.sample_rate)
            end_sample_index = start_sample_index + len(line_audio_waveform)

            if start_sample_index < 0: # Should not happen with valid flow_data
                print(f"    Warning: Negative start_sample_index ({start_sample_index}) for line {i+1}. Clamping to 0.")
                start_sample_index = 0
                end_sample_index = len(line_audio_waveform)


            # Ensure the full_verse_waveform is long enough
            if end_sample_index > len(full_verse_waveform):
                print(f"    Info: Extending full_verse_waveform to accommodate line {i+1} (ends at {end_sample_index}, current length {len(full_verse_waveform)}).")
                padding = np.zeros(end_sample_index - len(full_verse_waveform) + int(0.1 * self.sample_rate), dtype=np.float32)
                full_verse_waveform = np.concatenate((full_verse_waveform, padding))
            
            # Add (mix) the synthesized line into the full verse waveform
            # Simple additive mixing. Could be more sophisticated (e.g., gain adjustment).
            try:
                full_verse_waveform[start_sample_index : end_sample_index] += line_audio_waveform
            except ValueError as e:
                 print(f"    Error during mixing line {i+1} into full waveform: {e}")
                 print(f"      start_sample_index: {start_sample_index}, end_sample_index: {end_sample_index}, line_audio_waveform length: {len(line_audio_waveform)}, full_verse_waveform length: {len(full_verse_waveform)}")
                 # Attempt to place what fits if shapes mismatch
                 len_to_mix = min(len(line_audio_waveform), len(full_verse_waveform) - start_sample_index)
                 if len_to_mix > 0:
                     full_verse_waveform[start_sample_index : start_sample_index + len_to_mix] += line_audio_waveform[:len_to_mix]


        # Normalize the final waveform to prevent clipping if it's not empty
        max_abs_val = np.max(np.abs(full_verse_waveform))
        if max_abs_val > 0: # Avoid division by zero if silent
            # Normalize to a target level e.g. -3dB (0.707) or -6dB (0.5) to leave headroom
            # If max_abs_val is already > 1.0 (clipping), definitely normalize.
            # If it's very quiet, boosting too much can amplify noise.
            # For now, simple normalization if it exceeds 1.0, or to a standard level.
            if max_abs_val > 1.0:
                full_verse_waveform /= max_abs_val
            # else: # Optionally boost if too quiet, but be careful with noise
            #    if max_abs_val < 0.1 and max_abs_val > 0:
            #        full_verse_waveform = full_verse_waveform / max_abs_val * 0.5
        
        # Trim leading/trailing silence if any (optional, can affect precise start time if not careful)
        # full_verse_waveform, _ = librosa.effects.trim(full_verse_waveform, top_db=40) # Adjust top_db as needed

        return full_verse_waveform, self.sample_rate

    def save_audio(self, audio_data: AudioData, output_path: str = "output_rap.wav"):
        """Saves the audio data to a file."""
        waveform, sr = audio_data
        if waveform.size == 0:
            print("No audio data to save.")
            return

        try:
            import soundfile as sf 
            sf.write(output_path, waveform, sr)
            print(f"Audio saved to {output_path}")
        except ImportError:
            print("Error: 'soundfile' library not found. Cannot save audio. Please install it: pip install soundfile")
        except Exception as e:
            print(f"Error saving audio: {e}. Make sure 'soundfile' and its dependencies (like 'libsndfile') are installed.")

# Example Usage
if __name__ == "__main__":
    synth = RapSynthesizer(model_name_or_path=None) 
    
    mock_lyrics = [
        "Yo this is a test run right now",
        "Flowing on the beat and I'm having some fun",
        "AI rap is clearly on the rise today",
        "Watch out world here comes a big surprise hey"
    ]
    # Ensure flow_data has start_time_sec and duration_sec for each segment
    mock_flow: FlowData = [
        {"duration_sec": 2.0, "syllables": 7, "pitch_contour_id": "mid-falling", "start_time_sec": 0.0},
        {"duration_sec": 2.5, "syllables": 10, "pitch_contour_id": "rising", "start_time_sec": 2.0},
        {"duration_sec": 2.0, "syllables": 8, "pitch_contour_id": "mid-high", "start_time_sec": 4.5},
        {"duration_sec": 2.8, "syllables": 11, "pitch_contour_id": "falling-mid", "start_time_sec": 6.5}
    ]
    
    generated_audio_data = synth.synthesize_verse(mock_lyrics, mock_flow)
    
    if generated_audio_data and generated_audio_data[0].size > 0:
        synth.save_audio(generated_audio_data, "placeholder_rap_verse_timed.wav")
    else:
        print("Synthesis failed or produced empty audio.")

    # Test with slightly overlapping flow data to check mixing
    mock_flow_overlap: FlowData = [
        {"duration_sec": 2.0, "syllables": 7, "pitch_contour_id": "mid", "start_time_sec": 0.0},
        {"duration_sec": 2.0, "syllables": 8, "pitch_contour_id": "high", "start_time_sec": 1.5}, # Overlaps with first
    ]
    mock_lyrics_overlap = ["First line here", "Second line overlaps"]
    generated_audio_overlap = synth.synthesize_verse(mock_lyrics_overlap, mock_flow_overlap)
    if generated_audio_overlap and generated_audio_overlap[0].size > 0:
        synth.save_audio(generated_audio_overlap, "placeholder_rap_verse_overlap.wav")