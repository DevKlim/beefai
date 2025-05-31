import numpy as np
from typing import List, Optional, Dict
from beefai.utils.data_types import FlowData, AudioData
import os
import soundfile as sf # For saving audio

class RapSynthesizer:
    def __init__(self, sample_rate: int = 22050): # Common sample rate for TTS
        self.sample_rate = sample_rate
        print("RapSynthesizer initialized. NOTE: This is a stub and requires TTS/SVS integration for audio synthesis.")

    def synthesize_line(self, lyric_line: str, flow_datum: FlowData) -> Optional[AudioData]:
        """
        Synthesizes a single line of rap. (Currently a STUB)
        """
        # This requires a proper Speech Synthesis (TTS) or Singing Voice Synthesis (SVS) engine.
        # For now, it returns silent audio of the expected duration.
        duration_beats = flow_datum.get("duration_beats", 2.0) # Default if missing
        bpm = flow_datum.get("bpm_of_bar", 120.0) # Need BPM to convert beats to seconds
        
        # This is problematic: flow_datum as defined in data_types.py for the TRANSFORMER TARGET
        # does not contain bpm_of_bar directly. It contains bar_index, which would need
        # to be cross-referenced with SongBeatFeatures to get the BPM for that bar.
        # For simplicity of this stub, we'll assume a fixed BPM or that the caller
        # enriches flow_datum.

        # A more robust stub would take BPM from BeatInfo or similar context.
        # For now, let's assume a default BPM if not in flow_datum.
        beat_duration_sec = 60.0 / bpm
        duration_sec = duration_beats * beat_duration_sec
        
        num_samples = int(duration_sec * self.sample_rate)
        silence = np.zeros(num_samples, dtype=np.float32)
        # print(f"  Synthesizer (STUB): Generating {duration_sec:.2f}s of silence for line: '{lyric_line}'")
        return (silence, self.sample_rate)


    def synthesize_verse(self, lyrics: List[str], flow_data: FlowData, beat_info: Optional[dict] = None) -> Optional[AudioData]:
        """
        Synthesizes a full rap verse from lines of lyrics and flow data.
        
        Args:
            lyrics: A list of lyric lines.
            flow_data: A list of FlowDatum dicts, corresponding to each lyric line.
                       Each FlowDatum should ideally have 'start_offset_beats', 'duration_beats', 
                       and 'bar_index' to map to the beat_info for timing.
            beat_info: BeatInfo dictionary containing 'bpm' and 'downbeat_times' or 'estimated_bar_duration'
                       to calculate absolute timing.

        Returns:
            A tuple (waveform, sample_rate) for the synthesized verse, or None.
        """
        print("RapSynthesizer.synthesize_verse() called.")
        error_message = (
            "Rap audio synthesis is not implemented. This component requires integration "
            "with a Text-to-Speech (TTS) or Singing Voice Synthesis (SVS) engine capable of "
            "rendering expressive rap vocals according to the provided flow (timing, rhythm)."
        )
        # print(f"ERROR: {error_message}")
        # To allow a demo pipeline to run, we'll generate silence structured by flow_data
        # raise NotImplementedError(error_message)
        
        if not lyrics or not flow_data or len(lyrics) != len(flow_data):
            print("  Synthesizer Error: Mismatch between lyrics and flow_data counts, or missing data.")
            return None
        if not beat_info or not beat_info.get("bpm"):
             print("  Synthesizer Warning: Missing BPM from beat_info. Using default 120 BPM for timing estimates.")
        
        bpm = beat_info.get("bpm", 120.0) if beat_info else 120.0
        beat_duration_sec = 60.0 / bpm
        
        # Calculate absolute start times for each bar
        bar_absolute_start_times_sec: Dict[int, float] = {}
        if beat_info and beat_info.get("downbeat_times"):
            for i, dt in enumerate(beat_info["downbeat_times"]):
                bar_absolute_start_times_sec[i] = dt # Assuming downbeats mark bar starts
        elif beat_info and beat_info.get("estimated_bar_duration", 0) > 0:
            bar_dur = beat_info["estimated_bar_duration"]
            num_bars_estimate = int( (flow_data[-1].get("bar_index",0)+1) ) # Estimate from max bar_index in flow
            for i in range(num_bars_estimate + 4): # Add some buffer bars
                bar_absolute_start_times_sec[i] = i * bar_dur
        else: # Fallback if no downbeats or bar duration
            print("  Synthesizer Warning: Cannot determine absolute bar start times. Lines will be concatenated with fixed pauses.")
            # Use a simpler concatenation approach.
            verse_waveform = np.array([], dtype=np.float32)
            for i, line_text in enumerate(lyrics):
                fd = flow_data[i]
                # enrich fd with bpm for the stub synthesize_line
                fd_enriched = {**fd, "bpm_of_bar": bpm}
                line_audio_data = self.synthesize_line(line_text, fd_enriched)
                if line_audio_data:
                    verse_waveform = np.concatenate((verse_waveform, line_audio_data[0]))
                    # Add a small fixed pause between lines if no other timing info
                    pause_samples = int(0.2 * self.sample_rate) 
                    verse_waveform = np.concatenate((verse_waveform, np.zeros(pause_samples, dtype=np.float32)))
            if len(verse_waveform) > 0:
                return (verse_waveform, self.sample_rate)
            return None


        # Determine total duration needed for the verse waveform
        max_verse_end_time_sec = 0
        for fd in flow_data:
            bar_idx = fd.get("bar_index")
            if bar_idx is None or bar_idx not in bar_absolute_start_times_sec:
                print(f"  Synthesizer Warning: Missing bar_index or bar start time for flow datum: {fd}. Cannot place line accurately.")
                continue # Skip this line if it can't be placed
            
            bar_start_sec = bar_absolute_start_times_sec[bar_idx]
            line_start_sec_abs = bar_start_sec + (fd.get("start_offset_beats", 0) * beat_duration_sec)
            line_duration_sec = fd.get("duration_beats", 0) * beat_duration_sec
            max_verse_end_time_sec = max(max_verse_end_time_sec, line_start_sec_abs + line_duration_sec)

        if max_verse_end_time_sec == 0:
            print("  Synthesizer Error: Could not determine verse duration.")
            return None

        # Create empty waveform for the whole verse
        total_samples_verse = int(max_verse_end_time_sec * self.sample_rate) + int(self.sample_rate * 0.5) # Add 0.5s buffer
        full_verse_waveform = np.zeros(total_samples_verse, dtype=np.float32)

        print(f"  Synthesizer (STUB): Generating structured silence for verse (approx {max_verse_end_time_sec:.2f}s).")
        for i, line_text in enumerate(lyrics):
            fd = flow_data[i]
            bar_idx = fd.get("bar_index")

            if bar_idx is None or bar_idx not in bar_absolute_start_times_sec:
                continue 

            bar_start_sec = bar_absolute_start_times_sec[bar_idx]
            line_start_offset_sec = fd.get("start_offset_beats", 0) * beat_duration_sec
            line_duration_sec = fd.get("duration_beats", 0) * beat_duration_sec

            line_abs_start_sample = int((bar_start_sec + line_start_offset_sec) * self.sample_rate)
            line_num_samples = int(line_duration_sec * self.sample_rate)
            
            # For the stub, we just mark its duration with a small pulse for audibility in the silent track
            # A real synthesizer would place the synthesized audio here.
            if line_abs_start_sample + line_num_samples <= total_samples_verse and line_num_samples > 0:
                # Create a short click or tone at the start of the line stub
                click_duration_samples = min(line_num_samples, int(0.01 * self.sample_rate)) # 10ms click
                if click_duration_samples > 0:
                     # Simple sine wave pulse instead of noise for the "mark"
                    t_click = np.linspace(0, click_duration_samples / self.sample_rate, click_duration_samples, endpoint=False)
                    click_signal = 0.3 * np.sin(2 * np.pi * 440 * t_click) # A4 note
                    full_verse_waveform[line_abs_start_sample : line_abs_start_sample + click_duration_samples] += click_signal

        return (full_verse_waveform, self.sample_rate)

    def save_audio(self, audio_data: AudioData, file_path: str):
        """Saves an audio waveform to a file."""
        waveform, sr = audio_data
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            sf.write(file_path, waveform, sr)
            print(f"Audio saved to {file_path}")
        except Exception as e:
            print(f"Error saving audio to {file_path}: {e}")


# Example Usage (illustrative)
if __name__ == "__main__":
    synth = RapSynthesizer()
    
    dummy_lyrics = [
        "Yo, check the mic, one two, this is a test.",
        "Gotta make sure the flow is put to the blessed."
    ]
    # FlowData needs bar_index, start_offset_beats, duration_beats
    # These timings are relative to their respective bars.
    dummy_flow: FlowData = [
        {"bar_index": 0, "syllables": 10, "start_offset_beats": 0.0, "duration_beats": 1.75},
        {"bar_index": 0, "syllables": 12, "start_offset_beats": 2.0, "duration_beats": 1.75},
    ]
    # BeatInfo provides the absolute timing context for bars
    dummy_beat_info = {
        "bpm": 90.0,
        "beat_times": [i * (60.0/90.0) for i in range(16)], # 4 bars worth of beats
        "downbeat_times": [i * 4 * (60.0/90.0) for i in range(4)], # Downbeats for 4 bars
        "beats_per_bar": 4,
        "estimated_bar_duration": 4 * (60.0/90.0)
    }

    print(f"\nAttempting to synthesize verse (stub implementation)...")
    synthesized_audio = synth.synthesize_verse(dummy_lyrics, dummy_flow, dummy_beat_info)
    
    if synthesized_audio and synthesized_audio[0].size > 0:
        print(f"Synthesized audio (stub) waveform length: {len(synthesized_audio[0])} samples, SR: {synthesized_audio[1]}")
        output_dir = "temp_synth_output"
        ensure_dir(output_dir)
        synth.save_audio(synthesized_audio, os.path.join(output_dir, "stub_verse.wav"))
        print(f"Stub verse saved to '{output_dir}/stub_verse.wav'. It will be mostly silent with clicks.")
    else:
        print("Synthesis (stub) failed or produced empty audio.")