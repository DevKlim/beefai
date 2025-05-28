import os
import time
import numpy as np # For dummy audio creation
from typing import List, Optional # For type hinting

from beefai.data_processing.audio_processor import AudioProcessor
from beefai.data_processing.text_processor import TextProcessor # For syllable counting if needed by LLM or eval
from beefai.flow_model.model import FlowModel # Using the original placeholder FlowModel
from beefai.lyric_generation.agent import LyricAgent
from beefai.synthesis.synthesizer import RapSynthesizer
from beefai.utils.data_types import BeatInfo, FlowData, AudioData

# Configuration
OUTPUT_DIR = "output"
DEFAULT_INSTRUMENTAL_PATH = "sample_instrumental.wav" # Replace with your instrumental
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_dummy_instrumental(path: str, duration_sec: int = 24, sr: int = 22050, bpm: float = 90.0):
    """Creates a more distinct dummy WAV file with a clear 4/4 beat for analysis."""
    if not os.path.exists(path) or os.path.getsize(path) < 1000: # Recreate if missing or too small
        print(f"Creating dummy instrumental at {path} (BPM: {bpm}, Duration: {duration_sec}s)")
        try:
            import soundfile as sf
        except ImportError:
            print("Error: soundfile library is required to create dummy instrumental. pip install soundfile")
            return

        t_linspace = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        waveform = np.zeros_like(t_linspace, dtype=np.float32)

        beats_per_second = bpm / 60.0
        beat_interval_samples = int(sr / beats_per_second)
        
        # --- Define drum sounds ---
        kick_freq = 60
        kick_duration_samples = int(sr * 0.1) 
        t_kick = np.linspace(0, kick_duration_samples/sr, kick_duration_samples, endpoint=False)
        kick_envelope = np.exp(-t_kick * 30) 
        kick_burst = 0.8 * np.sin(2 * np.pi * kick_freq * t_kick) * kick_envelope
        
        snare_duration_samples = int(sr * 0.15)
        t_snare = np.linspace(0, snare_duration_samples/sr, snare_duration_samples, endpoint=False)
        snare_envelope = np.exp(-t_snare * 25)
        snare_burst = 0.6 * (np.random.rand(snare_duration_samples) * 2 - 1) * snare_envelope
        
        hat_duration_samples = int(sr * 0.05)
        t_hat = np.linspace(0, hat_duration_samples/sr, hat_duration_samples, endpoint=False)
        hat_envelope = np.exp(-t_hat * 80) 
        hat_burst = 0.4 * (np.random.rand(hat_duration_samples) * 2 -1) * hat_envelope

        for beat_idx in range(int(duration_sec * beats_per_second)):
            current_sample_pos = beat_idx * beat_interval_samples
            
            if current_sample_pos + kick_duration_samples < len(waveform):
                kick_amp = 1.0 if beat_idx % 4 == 0 else 0.8 
                waveform[current_sample_pos : current_sample_pos + kick_duration_samples] += kick_burst * kick_amp
            
            if beat_idx % 4 == 1 or beat_idx % 4 == 3:
                if current_sample_pos + snare_duration_samples < len(waveform):
                    waveform[current_sample_pos : current_sample_pos + snare_duration_samples] += snare_burst
            
            if current_sample_pos + hat_duration_samples < len(waveform):
                 waveform[current_sample_pos : current_sample_pos + hat_duration_samples] += hat_burst * 0.7
            off_beat_pos = current_sample_pos + beat_interval_samples // 2
            if off_beat_pos + hat_duration_samples < len(waveform):
                 waveform[off_beat_pos : off_beat_pos + hat_duration_samples] += hat_burst * 0.5
        
        pad_notes_freq = [55, 55 * (2**(3/12)), 55 * (2**(7/12))] 
        pad_waveform = np.zeros_like(t_linspace, dtype=np.float32)
        for i, freq in enumerate(pad_notes_freq):
            pad_waveform += (0.05 / (i+1)) * np.sin(2 * np.pi * freq * t_linspace * (1 + 0.001 * i * np.sin(2*np.pi*0.1*t_linspace)))
        
        total_samples = len(t_linspace)
        pad_fade_samples = sr * 2 
        if total_samples > 2 * pad_fade_samples :
            pad_envelope_full = np.concatenate([
                np.linspace(0,1,pad_fade_samples),
                np.ones(total_samples - 2*pad_fade_samples),
                np.linspace(1,0,pad_fade_samples)
            ])
            pad_waveform *= pad_envelope_full
        waveform += pad_waveform

        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.7 
        
        sf.write(path, waveform.astype(np.float32), sr)
        print(f"Dummy instrumental saved to {path}")
    else:
        print(f"Instrumental file {path} already exists and is suitable. Skipping creation.")


class RapBattleGame:
    def __init__(self, instrumental_path: str = DEFAULT_INSTRUMENTAL_PATH):
        print("Initializing beefai: AI Rap Battle Game...")
        self.audio_processor = AudioProcessor()
        self.text_processor = TextProcessor() 
        self.flow_model = FlowModel() 
        self.lyric_agent = LyricAgent() 
        self.synthesizer = RapSynthesizer()

        self.instrumental_path = instrumental_path
        self.beat_info: BeatInfo = {}
        self._prepare_instrumental()

    def _prepare_instrumental(self):
        print(f"\nProcessing instrumental: {self.instrumental_path}")
        if not os.path.exists(self.instrumental_path):
            print(f"Instrumental file not found: {self.instrumental_path}. Attempting to create dummy.")
        # Use the more detailed dummy instrumental creation
        create_dummy_instrumental(self.instrumental_path, duration_sec=32, bpm=90)
        
        if not os.path.exists(self.instrumental_path) or os.path.getsize(self.instrumental_path) < 1000:
            print("Critical: Failed to find or create a valid instrumental. AI responses will be impaired.")
            bpm = 90.0; beat_duration = 60.0 / bpm
            self.beat_info = {"bpm": bpm, "beat_times": [i*beat_duration for i in range(128)], 
                              "downbeat_times": [i*4*beat_duration for i in range(32)], "beats_per_bar":4,
                              "estimated_bar_duration": 4 * beat_duration}
            print("Using hardcoded fallback beat info.")
            return

        waveform, sr = self.audio_processor.load_audio(self.instrumental_path)
        if waveform.size > 0:
            self.beat_info = self.audio_processor.get_beat_info(waveform, sr)
            print(f"Instrumental Beat Info: BPM={self.beat_info.get('bpm')}, "
                  f"Found {len(self.beat_info.get('beat_times',[]))} beats, "
                  f"{len(self.beat_info.get('downbeat_times',[]))} downbeats. "
                  f"Est. Bar Duration: {self.beat_info.get('estimated_bar_duration')}")
            if not self.beat_info.get('downbeat_times') and self.beat_info.get('bpm', 0) > 0:
                print("Warning: No downbeats detected, but BPM is present. AI timing might be less precise.")
        else:
            print("Could not load instrumental audio.")
            bpm = 90.0; beat_duration = 60.0 / bpm # Fallback
            self.beat_info = {"bpm": bpm, "beat_times": [i*beat_duration for i in range(128)], 
                              "downbeat_times": [i*4*beat_duration for i in range(32)], "beats_per_bar":4,
                              "estimated_bar_duration": 4 * beat_duration}
            print("Using fallback beat info.")


    def ai_responds(self, user_rap_text: Optional[str] = None, num_bars: int = 2, lines_per_bar: int = 2) -> Optional[AudioData]:
        """
        AI generates and performs a rap verse.
        """
        if not self.beat_info or (not self.beat_info.get("beat_times") and not self.beat_info.get("bpm", 0) > 0) :
            print("Cannot generate AI response: Beat information not available or insufficient.")
            return None

        print("\n--- AI's Turn ---")
        if user_rap_text:
            print(f"AI responding to: \"{user_rap_text[:100]}...\"")
        else:
            print("AI starting the rap...")

        # 1. Generate Flow Pattern
        print("1. Generating flow pattern...")
        start_time = time.time()
        # Ensure beat_info is passed to flow_model
        flow_data: FlowData = self.flow_model.generate_flow(self.beat_info, num_bars=num_bars, lines_per_bar=lines_per_bar)
        print(f"   Flow generation took {time.time() - start_time:.2f}s. Generated {len(flow_data)} flow segments.")
        if not flow_data:
            print("   Flow model did not produce any flow data. AI cannot respond.")
            return None

        # 2. Generate Lyrics
        print("\n2. Generating lyrics...")
        start_time = time.time()
        ai_lyrics: List[str] = self.lyric_agent.generate_verse(user_rap_text, flow_data)
        print(f"   Lyric generation took {time.time() - start_time:.2f}s.")
        if not ai_lyrics or not any(ai_lyrics) or len(ai_lyrics) != len(flow_data):
            print("   Lyric agent did not produce valid lyrics for the flow. AI cannot respond.")
            if flow_data and (not ai_lyrics or len(ai_lyrics) != len(flow_data)): # Provide dummy if flow exists
                print("   Lyric agent failed, providing placeholder lyrics for flow.")
                ai_lyrics = [f"Placeholder line {i+1} ({fd.get('syllables', 'N')} syl)" for i, fd in enumerate(flow_data)]
            else:
                return None
        
        print("\n   AI Generated Lyrics:")
        for i, line in enumerate(ai_lyrics):
             fd = flow_data[i]
             print(f"   L{i+1} (Bar {fd.get('bar_index', 'N')}.{fd.get('line_index_in_bar','A')}): {line} "
                  f"(Flow: {fd.get('syllables')} syl, {fd.get('duration_sec','?'):.2f}s @ {fd.get('start_time_sec','?'):.2f}s)")


        # 3. Synthesize Rap Audio
        print("\n3. Synthesizing rap audio...")
        start_time = time.time()
        ai_rap_audio: Optional[AudioData] = self.synthesizer.synthesize_verse(ai_lyrics, flow_data)
        print(f"   Synthesis took {time.time() - start_time:.2f}s.")

        if ai_rap_audio and ai_rap_audio[0].size > 0:
            output_filename = os.path.join(OUTPUT_DIR, f"ai_rap_response_{time.strftime('%Y%m%d_%H%M%S')}.wav")
            self.synthesizer.save_audio(ai_rap_audio, output_filename)
            print(f"   AI rap audio saved to: {output_filename}")
            return ai_rap_audio
        else:
            print("   Synthesis failed or produced empty audio.")
            return None

    def start_battle_simulation(self, rounds: int = 1): # Renamed from start_battle for clarity
        print("\nWelcome to the AI Rap Battle Simulation!")
        print("The AI will generate responses based on the instrumental and (simulated) user input.")

        current_opponent_text = None
        for i in range(rounds):
            print(f"\n--- ROUND {i+1} ---")
            ai_audio_data = None 
            if i == 0:
                print("AI drops the opening verse...")
                ai_audio_data = self.ai_responds(user_rap_text=None, num_bars=2, lines_per_bar=2) 
                if ai_audio_data:
                    print("   AI's opening verse generated (check .wav file in 'output/' folder).")
                # Simulate a generic follow-up from user to set context for next AI round
                current_opponent_text = "That was a decent start, AI, but can you handle the heat?" 
            else:
                # user_diss = f"Yo AI, round {i+1}, your rhymes are still a bit buggy, my flow is snuggy!" # Example user input
                print(f"User (simulated) raps: \"{current_opponent_text}\"") # Use the text from previous turn
                
                print("AI is preparing a comeback...")
                ai_audio_data = self.ai_responds(user_rap_text=current_opponent_text, num_bars=2, lines_per_bar=2)
                if ai_audio_data:
                    print(f"   AI's response for round {i+1} generated (check .wav file).")
                # Update opponent text for the *next* simulated user turn, if any
                current_opponent_text = f"AI's response in round {i+1} was noted, but I'm still the G.O.A.T.!"

            if not ai_audio_data:
                print("AI failed to generate a response. Battle might end here.")
                break
            
            time.sleep(0.2) # Shorter pause for quicker simulation

        print("\nBattle Simulation Finished.")
        print(f"Check the '{OUTPUT_DIR}' directory for generated audio files.")


if __name__ == "__main__":
    game = RapBattleGame() 
    game.start_battle_simulation(rounds=2) # Simulate a 2-round exchange