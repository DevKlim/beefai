# BeefAI Project: File Context for LLM

This document provides a summary of files and directories within the BeefAI project, outlining their purpose, interactions, and role in the overall pipeline. This is intended to help an LLM understand the codebase for tasks like data processing and model training.

## Root Directory (`./`)

### `.gitignore`
*   **Purpose:** Specifies intentionally untracked files for Git (e.g., virtual environments, downloaded data, temporary files).
*   **LLM Context Notes:** Defines what's considered source code vs. generated/transient data.

### `README.md` (Main Project README)
*   **Purpose:** Primary documentation. Outlines project goals, AI rap battle pipeline, data processing phases, model architecture (Decoder-only Transformer for flow), and development plans.
*   **Key Content:** Overall project vision, detailed data engineering pipeline (Phase 0-7).
*   **LLM Context Notes:** **Essential reading for understanding the entire project's architecture, data flow, and the intended functionality of various components.** Provides the "big picture."

### `setup.sh` / `setup.bat`
*   **Purpose:** Scripts for setting up the Python virtual environment and installing dependencies from `requirements.txt`.
*   **Role in Pipeline:** Phase 0 (Setup). Foundational for running any Python scripts.
*   **LLM Context Notes:** User needs to run this first. Dependencies include `torch`, `librosa`, `pyphen`, `whisper-timestamped`.

## `beefai/` (Main Python Package)

This directory contains the core application logic.

### `beefai/main.py`
*   **Purpose:** Main entry point for simulating the AI Rap Battle Game. Orchestrates various components for a demo.
*   **Key Functionality:** Initializes processors, models (often placeholders initially), generates dummy instrumentals, simulates battle rounds.
*   **Interactions:** Uses `AudioProcessor`, `TextProcessor`, `FlowModel` (placeholder), `LyricAgent`, `RapSynthesizer`.
*   **LLM Context Notes:** Useful for understanding end-to-end interaction, but not directly part of the dataset preparation pipeline for the Transformer.

### `beefai/data_processing/`
Modules for preparing audio and text data for model training.

#### `audio_processor.py`
*   **Purpose:** Basic audio loading and analysis (e.g., BPM, beat times using `librosa`).
*   **Key Functionality:** `load_audio`, `get_beat_info` (simpler beat analysis than `BeatFeatureExtractor`).
*   **Outputs:** `AudioData`, `BeatInfo` data types.
*   **LLM Context Notes:** Used by `main.py` for game simulation. Less critical for the detailed Transformer dataset features than `BeatFeatureExtractor`. Depends on `librosa` and `scipy`. Correct versions are important (re: `scipy.signal.hann` error, ensure `librosa` uses `scipy.signal.get_window` correctly).

#### `beat_feature_extractor.py`
*   **Purpose:** Extracts detailed, bar-level beat features from instrumentals, crucial for conditioning the flow Transformer.
*   **Key Functionality:**
    *   Placeholder for source separation (Demucs) to get bass/drum stems.
    *   Placeholder for drum transcription (kick, snare, hi-hat onsets).
    *   Global tempo, time signature, and downbeat detection.
    *   Quantizes percussive onsets to a 16-subdivision grid per bar.
*   **Inputs:** Instrumental audio file path.
*   **Outputs:** `SongBeatFeatures` (list of `BarBeatFeatures` dicts).
*   **Dependencies/Interactions:** `librosa`, `numpy`. Interacts with `scripts/preprocess_dataset.py`.
*   **Role in Pipeline:** Phase 2 (Beat Feature Engineering). Critical for Transformer input.
*   **LLM Context Notes:** The quality of features heavily depends on the (currently placeholder) source separation and drum transcription. Assumes 4/4 time signature for downbeat calculation if not robustly detected. `scipy.signal.hann` error was due to `librosa` potentially not finding this in `scipy` correctly; ensuring compatible `librosa` and `scipy` versions is vital.

#### `flow_data_extractor.py`
*   **Purpose:** Extracts `FlowData` (syllable counts, timings per line) from acapellas using word alignment data. This generates target sequences for the flow model.
*   **Key Functionality:**
    *   Parses whisper-timestamped JSON output for word timings (`parse_whisper_timestamped_json`).
    *   Segments words into lines based on pauses.
    *   Calculates syllable counts (via `TextProcessor`).
    *   Normalizes line start/duration to beats relative to bar features.
*   **Inputs:** Alignment JSON file path, `SongBeatFeatures` (for bar context).
*   **Outputs:** `FlowData` (list of `FlowDatum` dicts).
*   **Dependencies/Interactions:** `TextProcessor`, `beefai.utils.data_types`. Called by `scripts/preprocess_dataset.py`.
*   **Role in Pipeline:** Phase 3 (Flow Data Engineering). Critical for Transformer targets.
*   **LLM Context Notes:** Accuracy depends heavily on the quality of input alignments.

#### `text_processor.py`
*   **Purpose:** Handles text-specific tasks like syllable counting and (placeholder) phoneme generation.
*   **Key Functionality:** `count_syllables_in_word` (uses `pyphen`), `count_syllables_in_line`, placeholder `get_phonemes`.
*   **Dependencies/Interactions:** `pyphen`. Used by `FlowDataExtractor`.
*   **LLM Context Notes:** Syllable counting is important for `FlowDatum`. Phoneme generation is a placeholder.

### `beefai/flow_model/`
Components for the flow generation Transformer model.

#### `tokenizer.py` (`FlowTokenizer`)
*   **Purpose:** Converts raw beat features and flow data into token sequences for the Transformer, and vice-versa. Manages the vocabulary.
*   **Key Functionality:**
    *   Defines vocabulary for special tokens, BPM bins, time signatures, percussive events (e.g., `[KICK_AT_X]`), syllables, offset bins, duration bins.
    *   `encode_bar_features()`: Converts `BarBeatFeatures` to tokens.
    *   `encode_flow_datum()`: Converts `FlowDatum` to tokens.
    *   `encode_song_instance()`: Combines beat and flow tokens for a full song, generating `token_ids`, `segment_ids`, `intra_line_pos_ids`.
    *   Saves/loads vocabulary and configuration (`flow_tokenizer_config_v2.json`).
*   **Inputs:** `BarBeatFeatures`, `FlowDatum`.
*   **Outputs:** Lists of token IDs, segment IDs, intra-line position IDs.
*   **Dependencies/Interactions:** Used by `scripts/05a_tokenize_data_lite.py` and `FlowDataset`. Relies on `beefai.utils.data_types`.
*   **Role in Pipeline:** Phase 4 (Tokenization).
*   **LLM Context Notes:** The vocabulary definition and quantization schemes (for BPM, offset, duration) are critical. `segment_ids` and `intra_line_pos_ids` are key for providing context to the Transformer beyond just token embeddings.

#### `transformer_model.py` (`FlowTransformerDecoder`, `FlowGPTConfig`)
*   **Purpose:** Implements the Decoder-only Transformer architecture for flow generation.
*   **Key Functionality:**
    *   `FlowGPTConfig`: Configuration class for model hyperparameters.
    *   `CausalSelfAttention`, `MLP`, `Block`: Standard Transformer components.
    *   `FlowTransformerDecoder`: The main model class. Includes embeddings for tokens, absolute positions, segment types, and intra-line positions.
    *   `forward()` method for training (calculates loss).
    *   `generate()` method for inference (autoregressive token generation).
*   **Inputs (during training):** Token IDs, segment IDs, intra-line position IDs, target IDs.
*   **Outputs (during inference):** Sequence of generated token IDs.
*   **Dependencies/Interactions:** `torch`. Used by `train_lite_flow_model.py`.
*   **LLM Context Notes:** Key architectural choices: decoder-only, multiple embedding types for rich context. `pad_token_id` in config is used for ignoring padding in loss. `get_next_context_ids_for_token` is a helper for generation to determine context for new tokens.

#### `dataset.py` (`FlowDataset`)
*   **Purpose:** PyTorch `Dataset` class for loading tokenized data and preparing batches for training.
*   **Key Functionality:**
    *   Loads pre-tokenized song data (output of `05a_tokenize_data_lite.py`).
    *   Slices long song sequences into fixed-size blocks (`block_size`) for training.
    *   Prepares input IDs, target IDs, segment IDs, and intra-line position IDs for each training example.
*   **Inputs:** Path to a `.pt` file (e.g., `train_lite.pt`) containing tokenized songs, tokenizer pad ID, block size.
*   **Outputs:** Dictionaries of tensors for training batches.
*   **Dependencies/Interactions:** `torch`. Used by `train_lite_flow_model.py`.
*   **LLM Context Notes:** Handles the conversion of full tokenized songs into manageable chunks for the Transformer.

#### `model.py` (`FlowModel`)
*   **Purpose:** Older, simpler, or placeholder model for flow generation. Not the main Transformer.
*   **LLM Context Notes:** Primarily a placeholder; the focus for training is `transformer_model.py`.

### `beefai/lyric_generation/agent.py` (`LyricAgent`)
*   **Purpose:** Placeholder for lyric generation. Intended for integration with an LLM.
*   **LLM Context Notes:** Currently returns dummy/placeholder lyrics.

### `beefai/synthesis/synthesizer.py` (`RapSynthesizer`)
*   **Purpose:** Placeholder for audio synthesis (TTS/SVS).
*   **LLM Context Notes:** Currently returns dummy/placeholder audio.

### `beefai/utils/data_types.py`
*   **Purpose:** Defines common Python type hints and data structures used throughout the project.
*   **Key Content:** `FlowDatum`, `FlowData`, `BeatInfo`, `BarBeatFeatures`, `SongBeatFeatures`, `TrainingInstance`.
*   **LLM Context Notes:** Essential for understanding the expected structure of data passed between modules.

## `data/` Directory

Holds all datasets, intermediate processing outputs, and final model inputs.

*   `acapellas/`: Stores acapella audio files (output of source separation).
*   `instrumentals/`: Stores instrumental audio files (output of source separation or raw inputs).
*   `lyrics/`: Raw lyric text files, matched by filename to songs.
*   `raw_songs_full/`: Initial complete song audio files (input for source separation).
*   `alignments_json/`: Word-level alignment JSON files (output of `whisper-timestamped`).
    *   `whisper_temp_output/`: Temporary directory for whisper-timestamped.
*   `temp_demucs_separated/`: Temporary storage for Demucs outputs if used by `scripts/preprocess_dataset.py` or `run_full_data_pipeline.py`. (BeatFeatureExtractor's simulation might create subdirs here like `stems_cache`).
*   **`processed_for_transformer/`**:
    *   Intended output directory for `scripts/preprocess_dataset.py`.
    *   Contains `processed_training_data.pt`: A list of `TrainingInstance` dicts (untokenized `SongBeatFeatures` and `FlowData` per song).
    *   May also contain caches like `beat_features_cache/` and `flow_data_cache/` generated by `preprocess_dataset.py`.
*   **`tokenized_lite/`**:
    *   Output directory for `scripts/05a_tokenize_data_lite.py`.
    *   Contains `train_lite.pt` and `val_lite.pt`: Tokenized song sequences ready for `FlowDataset` for the "lite" model.

## `lite_model_training/`

Files specific to training the "lite" version of the flow model.

### `data_config_lite.yaml`
*   **Purpose:** Configuration for data paths and parameters related to the "lite" dataset.
*   **Key Content:** Paths for tokenizer config, source of processed data (from `preprocess_dataset.py`), output paths for tokenized "lite" data (`train_lite.pt`, `val_lite.pt`), max songs for lite dataset, validation split ratio.
*   **Interactions:** Read by `scripts/05a_tokenize_data_lite.py` and `lite_model_training/train_lite_flow_model.py`.
*   **LLM Context Notes:** Centralizes paths for lite data generation and training.

### `model_config_lite.yaml`
*   **Purpose:** Configuration for the "lite" Transformer model architecture.
*   **Key Content:** `block_size`, `n_layer`, `n_head`, `n_embd`, `max_segment_types`, `max_intra_line_positions`, `dropout`, `bias`.
*   **Interactions:** Read by `lite_model_training/train_lite_flow_model.py` to instantiate `FlowGPTConfig`.
*   **LLM Context Notes:** Defines a smaller version of the Transformer model.

### `train_lite_flow_model.py`
*   **Purpose:** Script to train the "lite" flow generation Transformer model.
*   **Key Functionality:**
    *   Loads model and data configurations from YAML files.
    *   Initializes `FlowTokenizer`, `FlowGPTConfig`, `FlowTransformerDecoder`, `FlowDataset`.
    *   Implements the training loop (optimizer, scheduler, loss calculation, evaluation).
    *   Handles model checkpointing.
*   **Inputs:** `train_lite.pt`, `val_lite.pt`, tokenizer config, model config.
*   **Outputs:** Trained model checkpoints in `data/checkpoints/flow_model_lite/`.
*   **LLM Context Notes:** The primary script for training the smaller, example model.

## `scripts/` Directory

Utility and pipeline scripts.

### `run_full_data_pipeline.py`
*   **Purpose:** Master script to orchestrate the entire data preparation pipeline from raw songs to tokenized data ready for model training.
*   **Key Functionality:** Sequentially calls other scripts:
    1.  (Optional) `organize_downloaded_songs.py`
    2.  (Optional) `remove_first_line_from_lyrics.py`
    3.  Source separation (Demucs or Audio-Separator via command line).
    4.  Forced alignment (`whisper-timestamped` via command line).
    5.  `preprocess_dataset.py` (for feature extraction and creating `processed_training_data.pt`).
    6.  `05a_tokenize_data_lite.py` (for creating `train_lite.pt`, `val_lite.pt`).
*   **LLM Context Notes:** Intended as the main user-facing script for dataset preparation. Handles conditional execution of steps. The error logs indicate this script (or its sub-processes) is where issues were encountered.

### `preprocess_dataset.py`
*   **Purpose:** Core script for Phase 2 & 3 of data prep. Extracts beat features and flow data for each song and saves them as `TrainingInstance` objects.
*   **Key Functionality:**
    *   Iterates through instrumentals and corresponding alignment files.
    *   Uses `BeatFeatureExtractor` to get `SongBeatFeatures`.
    *   Uses `FlowDataExtractor` to get `FlowData`.
    *   Saves a list of `TrainingInstance` dicts to `data/processed_for_transformer/processed_training_data.pt`.
    *   Manages caching of extracted features to avoid re-computation.
*   **Inputs:** Instrumentals dir, alignments JSON dir.
*   **Outputs:** `processed_training_data.pt` and feature caches.
*   **LLM Context Notes:** A critical script. Failures here (like the `scipy.signal.hann` error) block the entire pipeline. Produces intermediate data that `05a_tokenize_data_lite.py` consumes.

### `05a_tokenize_data_lite.py`
*   **Purpose:** Tokenizes the output of `preprocess_dataset.py` for the "lite" model. Creates `train_lite.pt` and `val_lite.pt`.
*   **Key Functionality:**
    *   Loads `processed_training_data.pt`.
    *   Loads/initializes `FlowTokenizer` based on `data_config_lite.yaml`.
    *   Selects a subset of songs for the "lite" dataset (configurable).
    *   For each selected song, uses `tokenizer.encode_song_instance()` to get tokenized sequences.
    *   Splits tokenized songs into training and validation sets.
    *   Saves `train_lite.pt` and `val_lite.pt` in `data/tokenized_lite/`.
*   **Inputs:** `processed_training_data.pt`, `data_config_lite.yaml`.
*   **Outputs:** `train_lite.pt`, `val_lite.pt`.
*   **LLM Context Notes:** The final data preparation step before training the lite model. Depends on `FlowTokenizer` and the output of `preprocess_dataset.py`.

### `download_youtube_lyrics.py`
*   **Purpose:** Downloads audio from YouTube URLs and attempts to fetch corresponding lyrics using `lyricsgenius`.
*   **LLM Context Notes:** An optional data acquisition script. Requires `GENIUS_ACCESS_TOKEN`. Outputs to `downloaded_songs/`.

### `organize_downloaded_songs.py`
*   **Purpose:** Moves files from `downloaded_songs/` into the structured `data/` subdirectories (`raw_songs_full/`, `lyrics/`).
*   **LLM Context Notes:** Utility script for managing downloaded data.

### `remove_first_line_from_lyrics.py`
*   **Purpose:** Cleans common metadata header lines from `.txt` lyric files (often artifacts from `lyricsgenius`).
*   **LLM Context Notes:** A data cleaning utility.

---

This summary should give an LLM a good overview of the project's structure, the role of each key file, and how they interact, particularly concerning the data pipeline for training the flow generation model.