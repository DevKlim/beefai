import random
import numpy as np
from typing import List, Optional
from beefai.utils.data_types import BeatInfo, FlowData, FlowDatum

class FlowModel:
    def __init__(self, model_path: Optional[str] = None):
        """
        Placeholder for the flow generation model.
        """
        self.model_path = model_path
        # self.model = self._load_model(model_path)
        print(f"FlowModel initialized. (Mode: {'Pretrained' if model_path else 'Placeholder'})")

    def _load_model(self, path: Optional[str]):
        if path:
            print(f"Loading flow model from {path}...")
            return "dummy_trained_model"
        return "placeholder_model"

    def generate_flow(self, beat_info: BeatInfo, num_bars: int = 4, lines_per_bar: int = 2) -> FlowData:
        """
        Generates a sequence of flow data (rhythmic structure, syllable counts).
        This placeholder tries to align flow segments with downbeats and bars.
        """
        print("FlowModel: Generating placeholder flow...")
        flow_data: FlowData = []

        bpm = beat_info.get("bpm", 120.0)
        beat_times = beat_info.get("beat_times", [])
        downbeat_times = beat_info.get("downbeat_times", [])
        beats_per_bar = beat_info.get("beats_per_bar", 4)
        
        if not beat_times and bpm > 0: # Create synthetic beat_times if missing but BPM is present
            print("Warning: No beat_times in beat_info. Generating synthetic beat_times for flow.")
            total_duration_estimate = num_bars * beats_per_bar * (60.0 / bpm)
            beat_times = np.arange(0, total_duration_estimate, 60.0 / bpm).tolist()

        if not downbeat_times and beat_times: # If no downbeats, infer them
            print("Warning: No downbeat_times in beat_info. Inferring downbeats for flow.")
            downbeat_times = [beat_times[i] for i in range(0, len(beat_times), beats_per_bar) if i < len(beat_times)]
        
        if not downbeat_times and not beat_times: # No timing info at all
            print("Error: Insufficient beat_info (no beat_times or downbeat_times). Cannot generate meaningful flow.")
            # Fallback: generate very basic structure not tied to audio
            avg_beat_duration = 60.0 / bpm
            for bar_idx in range(num_bars):
                bar_start_time = bar_idx * beats_per_bar * avg_beat_duration
                for line_idx in range(lines_per_bar):
                    line_duration_beats = beats_per_bar / lines_per_bar
                    flow_datum: FlowDatum = {
                        "bar_index": bar_idx + 1,
                        "line_index_in_bar": line_idx + 1,
                        "start_time_sec": round(bar_start_time + line_idx * line_duration_beats * avg_beat_duration, 3),
                        "duration_sec": round(line_duration_beats * avg_beat_duration, 3),
                        "syllables": random.randint(int(line_duration_beats * 1.5), int(line_duration_beats * 3)),
                        "pitch_contour_id": random.choice(["mid", "mid-low", "mid-high"]),
                        "start_beat_global": round((bar_start_time + line_idx * line_duration_beats * avg_beat_duration) / avg_beat_duration, 2),
                        "duration_beats": round(line_duration_beats, 2)
                    }
                    flow_data.append(flow_datum)
            return flow_data


        avg_beat_duration = 60.0 / bpm if bpm > 0 else 0.5 # Fallback if BPM is 0

        # Iterate through bars, using downbeat_times as bar start markers
        # If not enough downbeats for num_bars, extrapolate based on avg_beat_duration.
        
        current_bar_start_time = 0.0
        if downbeat_times:
            current_bar_start_time = downbeat_times[0]
        elif beat_times:
            current_bar_start_time = beat_times[0]


        for bar_idx in range(num_bars):
            if bar_idx < len(downbeat_times):
                bar_start_time = downbeat_times[bar_idx]
                if bar_idx + 1 < len(downbeat_times):
                    bar_end_time = downbeat_times[bar_idx + 1]
                else: # Last provided downbeat, or only one downbeat
                    bar_end_time = bar_start_time + beats_per_bar * avg_beat_duration
            else: # Extrapolate bar start times if we ran out of detected downbeats
                # Start from the last known downbeat or the beginning
                last_known_downbeat = downbeat_times[-1] if downbeat_times else (beat_times[0] if beat_times else 0)
                num_bars_since_last_downbeat = bar_idx - (len(downbeat_times) -1 if downbeat_times else 0)
                bar_start_time = last_known_downbeat + num_bars_since_last_downbeat * beats_per_bar * avg_beat_duration
                bar_end_time = bar_start_time + beats_per_bar * avg_beat_duration
            
            actual_bar_duration = bar_end_time - bar_start_time
            if actual_bar_duration <= 0 : # Safety for weird beat info
                actual_bar_duration = beats_per_bar * avg_beat_duration


            for line_idx in range(lines_per_bar):
                # Distribute lines within the bar duration
                line_start_offset_ratio = line_idx / lines_per_bar
                line_duration_ratio = 1.0 / lines_per_bar

                line_start_time_sec = bar_start_time + line_start_offset_ratio * actual_bar_duration
                line_duration_sec = line_duration_ratio * actual_bar_duration
                
                # Syllables: random, e.g., 1.5 to 3 syllables per beat equivalent in the line
                # A line spans `actual_bar_duration * line_duration_ratio` seconds.
                # This is `(actual_bar_duration * line_duration_ratio) / avg_beat_duration` beats.
                line_duration_beats_val = (actual_bar_duration * line_duration_ratio) / avg_beat_duration if avg_beat_duration > 0 else beats_per_bar / lines_per_bar
                
                min_syllables = max(1, int(line_duration_beats_val * 1.5)) # e.g., 1.5 syllables per beat
                max_syllables = int(line_duration_beats_val * 3.0)   # e.g., 3 syllables per beat
                num_syllables = random.randint(min_syllables, max_syllables) if max_syllables > min_syllables else min_syllables


                flow_datum: FlowDatum = {
                    "bar_index": bar_idx + 1,
                    "line_index_in_bar": line_idx + 1,
                    "start_time_sec": round(line_start_time_sec, 3),
                    "duration_sec": round(line_duration_sec, 3),
                    "syllables": num_syllables,
                    "pitch_contour_id": random.choice(["mid", "mid-low", "mid-high", "rising", "falling"]),
                    "start_beat_global": round(line_start_time_sec / avg_beat_duration if avg_beat_duration > 0 else 0, 2),
                    "duration_beats": round(line_duration_beats_val, 2)
                }
                flow_data.append(flow_datum)
        
        return flow_data

# Example Usage
if __name__ == "__main__":
    mock_beat_info_detailed: BeatInfo = {
        "bpm": 90.0,
        "beat_times": [0.0, 0.667, 1.333, 2.0, 2.667, 3.333, 4.0, 4.667, # Bar 1 & 2
                       5.333, 6.0, 6.667, 7.333, 8.0, 8.667, 9.333, 10.0], # Bar 3 & 4
        "downbeat_times": [0.0, 2.667, 5.333, 8.0], # Downbeats at start of each bar (approx)
        "beats_per_bar": 4,
        "estimated_bar_duration": 2.667
    }
    
    flow_gen = FlowModel()
    generated_flow = flow_gen.generate_flow(mock_beat_info_detailed, num_bars=2, lines_per_bar=2)
    
    print("\nGenerated Flow Data (Detailed Beat Info):")
    if not generated_flow: print("  No flow generated.")
    for i, datum in enumerate(generated_flow):
        print(f"  Segment {i+1}:")
        for key, value in datum.items():
            print(f"    {key}: {value}")

    print("\nGenerating flow with minimal beat_info (only BPM):")
    minimal_beat_info: BeatInfo = {"bpm": 100.0, "beat_times": [], "downbeat_times": [], "beats_per_bar": 4}
    generated_flow_minimal = flow_gen.generate_flow(minimal_beat_info, num_bars=1, lines_per_bar=1)
    print("\nGenerated Flow Data (Minimal Info):")
    if not generated_flow_minimal: print("  No flow generated.")
    for i, datum in enumerate(generated_flow_minimal):
        print(f"  Segment {i+1}:")
        for key, value in datum.items():
            print(f"    {key}: {value}")

    print("\nGenerating flow with no beat_info at all (should fail gracefully or use defaults):")
    no_beat_info: BeatInfo = {"bpm": 0.0, "beat_times": [], "downbeat_times": []} # No valid info
    generated_flow_none = flow_gen.generate_flow(no_beat_info, num_bars=1, lines_per_bar=2)
    print("\nGenerated Flow Data (No Info):")
    if not generated_flow_none: print("  No flow generated.")
    for i, datum in enumerate(generated_flow_none):
        print(f"  Segment {i+1}:")
        for key, value in datum.items():
            print(f"    {key}: {value}")