from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Represents the rhythmic and structural information for a segment of rap
# This data guides the lyric generation and synthesis.
# Specifically for the Transformer-based Flow Model, the target is:
# (bar_index, line_index_in_bar, syllables, start_offset_beats, duration_beats)
# Pitch contours are ignored for this specific model as per README.
FlowDatum = Dict[str, Any]
# Example (structured, before tokenization):
# {
#     "bar_index": int,                 # 0-based index of the bar this line primarily belongs to
#     "line_index_in_bar": int,         # 0-based index of the line within its bar segment for flow
#     "syllables": int,                 # Target number of syllables for this flow segment
#     "start_offset_beats": float,      # Start time of the segment in beats, relative to the start of its bar
#     "duration_beats": float           # Duration of the segment in beats
# }

FlowData = List[FlowDatum] # A sequence of flow datums, typically representing a verse or section


# Represents beat information extracted from an audio track by AudioProcessor (legacy/general)
BeatInfo = Dict[str, Any]
# Example (basic, from AudioProcessor):
# {
#     "bpm": float,
#     "beat_times": List[float],       # List of timestamps for each detected beat
#     "downbeat_times": List[float],   # List of timestamps for each detected downbeat
#     "beats_per_bar": int,            # Estimated beats per bar (e.g., 4)
#     "estimated_bar_duration": float  # Estimated duration of a bar in seconds
# }

# Beat Features for Transformer Input (per bar)
# This will be the output of `beat_feature_extractor.py`
BarBeatFeatures = Dict[str, Any]
# Example:
# {
#   "bar_index": int, # 0-based index of the bar in the song
#   "bpm": float, 
#   "time_signature": Tuple[int, int], # e.g., (4, 4)
#   "kick_events": List[int],  # List of active 16th note subdivision indices (0-15) within this bar
#   "snare_events": List[int],
#   "hihat_events": List[int],
#   "bass_events": List[int]
# }
SongBeatFeatures = List[BarBeatFeatures] # Represents all bars in a song


# Represents lyrical content with word-level timing, typically from forced alignment
LyricsData = List[Dict[str, Any]] 
# Example: 
# [
#   {"word": "hello", "start_time": 0.5, "end_time": 0.9},
#   {"word": "world", "start_time": 1.0, "end_time": 1.5}
# ]

# Represents raw audio data
AudioData = Tuple[np.ndarray, int] # (waveform_array: np.ndarray, sample_rate: int)


# Data for training the flow model:
# A pair of (input_beat_features_for_song, target_flow_data_for_song)
# where target_flow_data_for_song is a list of FlowDatum items for that song, associated with the bars in SongBeatFeatures
TrainingInstance = Dict[str, Any]
# Example:
# {
#   "song_id": "some_unique_id",
#   "beat_features": SongBeatFeatures, # List of BarBeatFeatures
#   "flow_targets": FlowData          # List of FlowDatum for the entire song
# }
TrainingDataFileContent = List[TrainingInstance]

# Tokenized sequence for the transformer (conceptual, actual format may vary)
TokenSequence = List[int]