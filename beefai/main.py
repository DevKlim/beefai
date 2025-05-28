import os
import time
import numpy as np
from typing import List, Optional

from beefai.data_processing.audio_processor import AudioProcessor
from beefai.data_processing.text_processor import TextProcessor
from beefai.flow_model.model import FlowModel
from beefai.lyric_generation.agent import LyricAgent
from beefai.synthesis.synthesizer import RapSynthesizer
from beefai.utils.data_types import BeatInfo, FlowData, AudioData

# Configuration
OUTPUT_DIR = "output"
DEFAULT_INSTRUMENTAL_PATH = os.path.join(OUTPUT_DIR, "beefai_default_instrumental.wav")
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

        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        waveform = np.zeros_like(t, dtype=np.float32)

        beats_per_second = bpm / 60.0
        beat_interval_samples = int(sr / beats_per_second)
        
        # --- Define drum sounds ---
        # Kick drum (low frequency burst)
        kick_freq = 60
        kick_duration_samples = int(sr * 0.1) 
        t_kick = np.linspace(0, kick_duration_samples/sr, kick_duration_samples, endpoint=False)
        kick_envelope = np.exp(-t_kick * 30) 
        kick_burst = 0.8 * np.sin(2 * np.pi * kick_freq * t_kick) * kick_envelope
        
        # Snare drum (noise burst)
        snare_duration_samples = int(sr * 0.15)
        t_snare = np.linspace(0, snare_duration_samples/sr, snare_duration_samples, endpoint=False)
        snare_envelope = np.exp(-t_snare * 25)
        # Band-pass filtered noise for snare (approximate)
        noise = np.random.randn(snare_duration_samples)
        # Simple butterworth filter (requires scipy, so using simpler for now)
        # b, a = signal.butter(4, [200/(0.5*sr), 2000/(0.5*sr)], btype='band')
        # filtered_noise = signal.lfilter(b,a, noise)
        # snare_burst = 0.6 * filtered_noise * snare_envelope
        # Simplified snare: just noise with envelope
        snare_burst = 0.6 * (np.random.rand(snare_duration_samples) * 2 - 1) * snare_envelope
        
        # Hi-hat (short, high-frequency burst)
        hat_duration_samples = int(sr * 0.05)
        t_hat = np.linspace(0, hat_duration_samples/sr, hat_duration_samples, endpoint=False)
        hat_envelope = np.exp(-t_hat * 80) 
        # High-passed noise (approximate) - or just short noise burst
        hat_burst = 0.4 * (np.random.rand(hat_duration_samples) * 2 -1) * hat_envelope


        # --- Place drums in a 4/4 pattern ---
        for beat_idx in range(int(duration_sec * beats_per_second)):
            current_sample_pos = beat_idx * beat_interval_samples
            
            # Kick on every beat (stronger on 1)
            if current_sample_pos + kick_duration_samples < len(waveform):
                kick_amp = 1.0 if beat_idx % 4 == 0 else 0.8 # Downbeat kick stronger
                waveform[current_sample_pos : current_sample_pos + kick_duration_samples] += kick_burst * kick_amp
            
            # Snare on beat 2 and 4 (0-indexed beats: 1 and 3)
            if beat_idx % 4 == 1 or beat_idx % 4 == 3:
                if current_sample_pos + snare_duration_samples < len(waveform):
                    waveform[current_sample_pos : current_sample_pos + snare_duration_samples] += snare_burst
            
            # Hi-hat on every 8th note (on-beat and off-beat)
            # Main beat hi-hat
            if current_sample_pos + hat_duration_samples < len(waveform):
                 waveform[current_sample_pos : current_sample_pos + hat_duration_samples] += hat_burst * 0.7
            # Off-beat hi-hat (8th note)
            off_beat_pos = current_sample_pos + beat_interval_samples // 2
            if off_beat_pos + hat_duration_samples < len(waveform):
                 waveform[off_beat_pos : off_beat_pos + hat_duration_samples] += hat_burst * 0.5
        
        # Add a simple pad synth for harmonic context
        pad_notes_freq = [55, 55 * (2**(3/12)), 55 * (2**(7/12))] # A minor chord (A, C, E)
        pad_waveform = np.zeros_like(t, dtype=np.float32)
        for i, freq in enumerate(pad_notes_freq):
            pad_waveform += (0.05 / (i+1)) * np.sin(2 * np.pi * freq * t * (1 + 0.001 * i * np.sin(2*np.pi*0.1*t))) # Slight chorus
        
        # Fade pad in and out over the whole duration
        total_samples = len(t)
        pad_fade_samples = sr * 2 # 2 second fade
        if total_samples > 2 * pad_fade_samples :
            pad_envelope_full = np.concatenate([
                np.linspace(0,1,pad_fade_samples),
                np.ones(total_samples - 2*pad_fade_samples),
                np.linspace(1,0,pad_fade_samples)
            ])
            pad_waveform *= pad_envelope_full
        waveform += pad_waveform

        # Normalize
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.7 # Normalize to 0.7 to avoid clipping & leave headroom
        
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
        create_dummy_instrumental(self.instrumental_path, duration_sec=32, bpm=90) # Longer dummy for more content
        
        if not os.path.exists(self.instrumental_path) or os.path.getsize(self.instrumental_path) < 1000:
            print("Critical: Failed to find or create a valid instrumental. AI responses will be impaired.")
            # Fallback beat info if everything fails
            bpm = 90.0
            beat_duration = 60.0 / bpm
            self.beat_info = {
                "bpm": bpm, 
                "beat_times": [i * beat_duration for i in range(int(32 * (bpm/60.0)))], 
                "downbeat_times": [i * 4 * beat_duration for i in range(int(32 * (bpm/60.0) / 4))], 
                "beats_per_bar": 4,
                "estimated_bar_duration": 4 * beat_duration
            }
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
            print("Could not load instrumental audio. Using fallback beat info (as above).")
            bpm = 90.0; beat_duration = 60.0 / bpm
            self.beat_info = {"bpm": bpm, "beat_times": [i*beat_duration for i in range(128)], 
                              "downbeat_times": [i*4*beat_duration for i in range(32)], "beats_per_bar":4,
                              "estimated_bar_duration": 4 * beat_duration}


    def ai_responds(self, user_rap_text: Optional[str] = None, num_bars: int = 4, lines_per_bar: int = 2) -> Optional[AudioData]:
        """
        AI generates and performs a rap verse. num_bars is typically 2, 4, or 8 for a short response.
        """
        if not self.beat_info or (not self.beat_info.get("beat_times") and not self.beat_info.get("bpm", 0) > 0):
            print("Cannot generate AI response: Beat information not available or insufficient.")
            return None

        print("\n--- AI's Turn ---")
        if user_rap_text:
            print(f"AI responding to: \"{user_rap_text[:80]}...\"")
        else:
            print("AI starting the rap...")

        # 1. Generate Flow Pattern
        print("1. Generating flow pattern...")
        start_time = time.time()
        flow_data: FlowData = self.flow_model.generate_flow(self.beat_info, num_bars=num_bars, lines_per_bar=lines_per_bar)
        print(f"   Flow generation took {time.time() - start_time:.2f}s. Generated {len(flow_data)} flow segments for {num_bars} bars.")
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
            # Provide dummy valid lyrics if generation fails but flow exists
            if flow_data and (not ai_lyrics or len(ai_lyrics) != len(flow_data)):
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
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_filename = os.path.join(OUTPUT_DIR, f"ai_rap_response_{timestamp}.wav")
            self.synthesizer.save_audio(ai_rap_audio, output_filename)
            # For JS, it needs the path or a way to access this. For now, Python saves it.
            # In a web app, this would be served or path returned in API response.
            return ai_rap_audio 
        else:
            print("   Synthesis failed or produced empty audio.")
            return None

    def start_battle_simulation(self, rounds: int = 1):
        print("\nWelcome to the AI Rap Battle Simulation!")
        print("The AI will generate responses based on the instrumental and (simulated) user input.")

        current_opponent_text = None
        for i in range(rounds):
            print(f"\n--- ROUND {i+1} ---")
            ai_audio = None # Initialize ai_audio for the round
            if i == 0:
                print("AI drops the opening verse...")
                ai_audio = self.ai_responds(user_rap_text=None, num_bars=2, lines_per_bar=2) # Shorter for faster demo
                if ai_audio:
                    print("   AI's opening verse generated (check .wav file in 'output/' folder).")
                current_opponent_text = "AI started, that was okay, but my turn will make your circuits fray!" 
            else:
                user_diss = f"Yo AI, round {i+1}, your rhymes are getting old, my story's yet untold!"
                print(f"User (simulated) raps: \"{user_diss}\"")
                current_opponent_text = user_diss
                
                print("AI is preparing a comeback...")
                ai_audio = self.ai_responds(user_rap_text=current_opponent_text, num_bars=2, lines_per_bar=2)
                if ai_audio:
                    print(f"   AI's response for round {i+1} generated (check .wav file).")
            
            if not ai_audio:
                print("AI failed to generate a response. Battle might end here.")
                break
            
            time.sleep(0.5) # Shorter pause for quicker simulation

        print("\nBattle Simulation Finished.")
        print(f"Check the '{OUTPUT_DIR}' directory for generated audio files.")


if __name__ == "__main__":
    game = RapBattleGame() 
    
    # Test the beat info extraction with the dummy instrumental
    if game.beat_info:
        print("\n--- Initial Beat Info Check ---")
        print(f"  BPM: {game.beat_info.get('bpm')}")
        print(f"  Beats per bar: {game.beat_info.get('beats_per_bar')}")
        print(f"  Number of detected beats: {len(game.beat_info.get('beat_times', []))}")
        print(f"  Number of detected downbeats: {len(game.beat_info.get('downbeat_times', []))}")
        print(f"  First few downbeat times: {game.beat_info.get('downbeat_times', [])[:5]}")
        print(f"  Estimated bar duration: {game.beat_info.get('estimated_bar_duration')}")
        if not game.beat_info.get('downbeat_times'):
             print("  WARNING: Downbeats were not robustly detected. AI timing might be off.")
    else:
        print("\n--- Beat Info Check Failed ---")
        print("  No beat_info available for the game instance.")

    game.start_battle_simulation(rounds=2)