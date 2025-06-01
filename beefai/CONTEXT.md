# BeefAI Project: File Context for LLM (Master Control Program Document)

This document provides a comprehensive summary of files and directories within the BeefAI project. Its purpose is to serve as a persistent knowledge base for an LLM, outlining each component's purpose, interactions, data flow, and role in the overall AI Rap Battle pipeline, particularly focusing on data processing and model training for flow generation.

## Project Root (`./`)
*   **`.gitignore`**: Standard.
*   **`README.md`**: High-level project documentation, pipeline concept, Flow Model architecture. Essential for overall understanding.
*   **`CONTEXT.md`**: This file.
*   **`print_struc.py`**: Developer utility.
*   **`requirements.txt`**: Python dependencies. Critical for setup.
*   **`setup.bat` / `setup.sh`**: Environment setup scripts.
*   **`test.ipynb`**: Jupyter Notebook for experiments.
*   **`153 _ 253 2025 Assignment 2.pdf`**: External reference/assignment.
*   **`my_songs.txt`, `rap_data.txt`, `rap_urls.txt`**: Input data files.

---

## `beefai/` (Main Python Package)

### `beefai/data_processing/`
*   **`audio_processor.py` (`AudioProcessor`)**: Basic audio loading, resampling, `get_beat_info()` (BPM, beat/downbeat times).
*   **`beat_feature_extractor.py` (`BeatFeatureExtractor`)**: Extracts detailed bar-level beat features for model conditioning. Outputs `SongBeatFeatures`.
*   **`flow_data_extractor.py` (`FlowDataExtractor`)**: Extracts `FlowData` from acapellas and alignments. Populates `syllable_start_subdivisions`, `syllable_durations_quantized` (bin indices based on its internal `SYLLABLE_DURATION_BINS_SEC_FDE`), and `syllable_stresses` in `FlowDatum`.
*   **`text_processor.py` (`TextProcessor`)**: Handles syllable counting/segmentation, phoneme generation, and stress extraction (using `phonemizer` with fallback). `get_syllables_with_stress()` is key for `FlowDataExtractor`.

### `beefai/evaluation/`
*   **`rhythm_visualizer.py`**:
    *   **Purpose:** Generates an audio "click track" to visualize model-predicted syllable rhythms, overlaying these on an instrumental.
    *   **Key Functionality Changes (Step 2A):**
        *   `generate_syllable_sound_event()`: New helper to create varied sounds for syllables based on duration and stress.
        *   `generate_syllable_click_track()`: Modified to use `generate_syllable_sound_event()`. It now represents syllable durations (by the length of the generated sound) and stress (by varying pitch/amplitude of the sound). Uses `tokenizer.dequantize_syllable_duration_bin()` to get approximate beat durations for syllables.
        *   Can now attempt to save in MP3 format (with WAV fallback if LAME is missing).
    *   **LLM Context:** Enhanced qualitative assessment tool. Provides richer auditory feedback on generated flow.

### `beefai/flow_model/`
*   **`dataset.py` (`FlowDataset`)**: PyTorch Dataset. Token sequences per line are longer due to duration and stress tokens.
*   **`model.py` (`FlowModel`)**: Legacy/placeholder.
*   **`tokenizer.py` (`FlowTokenizer`)**: Converts features/flow to/from token sequences.
    *   **Key Functionality Changes (Step 2/2A):**
        *   Vocabulary includes tokens for syllable duration bins and stress levels.
        *   `encode_flow_datum()`: Encodes a triplet (start_subdiv, duration_bin, stress_level) for each syllable.
        *   `decode_flow_tokens_to_datum()`: Parses this triplet.
        *   `dequantize_syllable_duration_bin()`: New helper method to convert a syllable duration bin index back to an approximate duration in beats (used by `rhythm_visualizer.py`).
*   **`transformer_model.py` (`FlowTransformerDecoder`, `FlowGPTConfig`)**: Main Transformer model. `max_intra_line_positions` in config needs to accommodate (start + duration + stress) tokens per syllable.
*   **`flow_tokenizer_config_v2.json`**: Stores tokenizer vocabulary/config. Updated by `FlowTokenizer` to include stress tokens and duration bin definitions.

### `beefai/lyric_generation/`
*   **`agent.py` (`LyricAgent`)**: Stub for LLM-based lyric generation.
*   **`prompt_formatter.py` (`LLMPromptFormatter`) (NEW in Step 2B):**
    *   **Purpose:** Formats `FlowData` and `BeatInfo` into a textual prompt suitable for an LLM to generate lyrics.
    *   **Key Functionality:** `format_flow_for_llm()` takes flow data, beat info, and optional context, then constructs a detailed textual prompt outlining the rhythmic structure (syllable counts, onsets, de-quantized durations, stress) line by line.
    *   **LLM Context:** Bridges the gap between the structured output of the flow model and the input requirements of a lyric-generating LLM.

### `beefai/synthesis/`
*   No changes in Step 2.

### `beefai/utils/`
*   **`utils/data_types.py`**: Defines common `TypedDict`s.
    *   `FlowDatum`: Includes `syllable_durations_quantized: List[int]` and `syllable_stresses: List[int]`.
    *   `SyllableDetail`: New TypedDict for internal processing.

### `beefai/webapp/`
*   No changes in Step 2.

---
(Rest of CONTEXT.md sections like `data/`, `lite_model_training/`, `scripts/` remain structurally similar, but the data they handle or produce now contains the richer `FlowDatum`.)

---
This `CONTEXT.md` updated for Step 2A & 2B changes (enhanced visualizer, LLM prompt formatter, and underlying data/tokenizer changes for stress and duration handling).