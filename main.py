import os
import time
import numpy as np # For dummy audio creation

from beefai.data_processing.audio_processor import AudioProcessor
from beefai.data_processing.text_processor import TextProcessor # For syllable counting if needed by LLM or eval
from beefai.flow_model.model import FlowModel
from beefai.lyric_generation.agent import LyricAgent
from beefai.synthesis.synthesizer import RapSynthesizer
from beefai.utils.data_types import BeatInfo, FlowData, AudioData

# Configuration
OUTPUT_DIR = "output"
DEFAULT_INSTRUMENTAL_PATH = "sample_instrumental.wav" # Replace with your instrumental
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_dummy_instrumental(path: str, duration_sec: int = 10, sr: int = 22050, bpm: float = 120.0):
    """Creates a simple dummy WAV file with a beat track if one doesn't exist."""
    if not os.path.exists(path):
        print(f"Creating dummy instrumental at {path} (BPM: {bpm})")
        import soundfile as sf
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        # Background noise/pad
        # waveform = 0.1 * np.random.randn(len(t)).astype(np.float32)
        pad_freq1 = 55
        pad_freq2 = 82.5
        waveform = 0.05 * (np.sin(2 * np.pi * pad_freq1 * t) + np.sin(2 * np.pi * pad_freq2 * t))


        # Add a simple beat
        beats_per_second = bpm / 60.0
        beat_interval_samples = int(sr / beats_per_second)
        
        # Kick drum (low frequency burst)
        for i in range(0, len(waveform), beat_interval_samples):
            if i + 512 < len(waveform): # Ensure within bounds for a short burst
                kick_burst = 0.6 * np.sin(2 * np.pi * 60 * np.linspace(0, 0.1, 512)) * np.exp(-np.linspace(0,5,512))
                waveform[i : i + 512] += kick_burst
        
        # Snare drum (noise burst on 2nd and 4th beat of a 4-beat bar)
        # Assuming 4/4 time, snare on beats 2 and 4
        bar_duration_samples = 4 * beat_interval_samples
        for bar_start in range(0, len(waveform), bar_duration_samples):
            snare_beat_2_sample = bar_start + beat_interval_samples
            snare_beat_4_sample = bar_start + 3 * beat_interval_samples
            
            for snare_sample_start in [snare_beat_2_sample, snare_beat_4_sample]:
                 if snare_sample_start + 1024 < len(waveform):
                    snare_burst = 0.3 * (np.random.rand(1024) - 0.5) * np.exp(-np.linspace(0,5,1024))
                    waveform[snare_sample_start : snare_sample_start + 1024] += snare_burst
        
        waveform = np.clip(waveform, -1.0, 1.0) # Clip to avoid distortion
        sf.write(path, waveform.astype(np.float32), sr)
        print(f"Dummy instrumental saved to {path}")


class RapBattleGame:
    def __init__(self, instrumental_path: str = DEFAULT_INSTRUMENTAL_PATH):
        print("Initializing beefai: AI Rap Battle Game...")
        self.audio_processor = AudioProcessor()
        self.text_processor = TextProcessor() # For potential use, e.g. user input analysis
        self.flow_model = FlowModel() # Provide model_path if you have a trained flow model
        self.lyric_agent = LyricAgent() # Provide model_name_or_path for a real LLM
        self.synthesizer = RapSynthesizer() # Provide model_name for real TTS/SVS

        self.instrumental_path = instrumental_path
        self.beat_info: BeatInfo = {}
        self._prepare_instrumental()

    def _prepare_instrumental(self):
        print(f"\nProcessing instrumental: {self.instrumental_path}")
        if not os.path.exists(self.instrumental_path):
            print(f"Instrumental file not found: {self.instrumental_path}")
            create_dummy_instrumental(self.instrumental_path, duration_sec=16, bpm=90) # Create a fallback
            if not os.path.exists(self.instrumental_path):
                print("Failed to create or find instrumental. Exiting.")
                return

        waveform, sr = self.audio_processor.load_audio(self.instrumental_path)
        if waveform.size > 0:
            self.beat_info = self.audio_processor.get_beat_info(waveform, sr)
            print(f"Instrumental Beat Info: BPM={self.beat_info.get('bpm')}, Found {len(self.beat_info.get('beat_times',[]))} beats.")
        else:
            print("Could not load instrumental audio.")
            # Create fallback beat_info
            self.beat_info = {"bpm": 120.0, "beat_times": [i*0.5 for i in range(32)], "downbeat_times": [i*2.0 for i in range(8)]}
            print("Using fallback beat info.")


    def ai_responds(self, user_rap_text: Optional[str] = None, num_bars: int = 2, lines_per_bar: int = 2) -> Optional[AudioData]:
        """
        AI generates and performs a rap verse.
        """
        if not self.beat_info or not self.beat_info.get("beat_times"):
            print("Cannot generate AI response: Beat information not available.")
            return None

        print("\n--- AI's Turn ---")
        if user_rap_text:
            print(f"AI responding to: \"{user_rap_text[:100]}...\"")
        else:
            print("AI starting the rap...")

        # 1. Generate Flow Pattern
        print("1. Generating flow pattern...")
        start_time = time.time()
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
        if not ai_lyrics or not any(ai_lyrics):
            print("   Lyric agent did not produce any lyrics. AI cannot respond.")
            return None
        
        print("\n   AI Generated Lyrics:")
        for i, line in enumerate(ai_lyrics):
            print(f"   L{i+1}: {line} (Flow: {flow_data[i].get('syllables')} syllables, {flow_data[i].get('duration_sec')}s)")

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

    def start_battle(self):
        print("\nWelcome to the AI Rap Battle!")
        print("The AI will generate a response based on the instrumental.")
        print("For this demo, we'll have the AI generate a couple of verses.")

        # AI's opening verse
        ai_opening_audio = self.ai_responds(user_rap_text=None, num_bars=2, lines_per_bar=2)
        if ai_opening_audio:
            print("   Playing AI's opening verse (placeholder - check .wav file)")
            # In a real game, you'd play this audio mixed with the instrumental.
            # from IPython.display import Audio, display # For Jupyter
            # display(Audio(ai_opening_audio[0], rate=ai_opening_audio[1]))

        # Simulate user turn (in a real game, this would be user input)
        user_input_example = "Yo, your rhymes are weak, I'm the lyrical peak!"
        print(f"\n--- User's Turn (Example) ---")
        print(f"User raps: \"{user_input_example}\"")
        
        # AI's response to user
        ai_response_audio = self.ai_responds(user_rap_text=user_input_example, num_bars=2, lines_per_bar=2)
        if ai_response_audio:
            print("   Playing AI's response (placeholder - check .wav file)")

        print("\nBattle Demo Finished.")
        print(f"Check the '{OUTPUT_DIR}' directory for generated audio files.")


if __name__ == "__main__":
    # You can specify a path to your own instrumental here
    # Ensure it's a WAV or MP3 file that librosa can read.
    # custom_instrumental = "path/to/your/beat.wav"
    # game = RapBattleGame(instrumental_path=custom_instrumental)
    
    game = RapBattleGame() # Uses default or creates dummy instrumental
    game.start_battle()