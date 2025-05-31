from typing import TypedDict, List, Tuple, Union, Any, Optional
import numpy as np

# Audio Processing Related
AudioData = Tuple[np.ndarray, int] # Waveform, Sample Rate

class BeatInfo(TypedDict):
    bpm: float
    beat_times: List[float]
    downbeat_times: List[float]
    estimated_bar_duration: float
    beats_per_bar: int

# Beat Feature Extractor Related
class BarBeatFeatures(TypedDict):
    bar_index: int
    bpm: float
    time_signature: Tuple[int, int] # e.g., (4, 4)
    kick_events: List[int]    # List of active subdivision indices (0-15 for 16ths)
    snare_events: List[int]
    hihat_events: List[int]
    bass_events: List[int]    # Onsets of bass notes, quantized

SongBeatFeatures = List[BarBeatFeatures]

# Text/Lyrics Related
class WordTiming(TypedDict):
    word: str
    start_time: float
    end_time: float

LyricsData = List[WordTiming] # A list of timed words

# Flow Data Extractor Related
class FlowDatum(TypedDict):
    bar_index: int             # Index of the bar this flow line belongs to
    line_index_in_bar: int     # Index of this line within its assigned bar (0, 1, 2...)
    syllables: int             # Total syllables in the line
    start_offset_beats: float  # Start of the line relative to bar start, in beats
    duration_beats: float      # Duration of the line, in beats
    syllable_start_subdivisions: List[int] # List of 0-indexed subdivisions (e.g., 0-15)
                                          # within the bar where syllables of this line start.

FlowData = List[FlowDatum] # A sequence of flow data for a song or section

# For combining features and targets for training
class TrainingInstance(TypedDict):
    song_id: str
    beat_features: SongBeatFeatures
    flow_targets: FlowData

# Placeholder for more complex phoneme info if needed later
class PhonemeInfo(TypedDict):
    phoneme: str
    duration: Optional[float]