# BeefAI Project: File Context for LLM (Master Control Program Document)

This document provides a comprehensive summary of files and directories within the BeefAI project. Its purpose is to serve as a persistent knowledge base for an LLM, outlining each component's purpose, interactions, data flow, and role in the overall AI Rap Battle pipeline, particularly focusing on data processing and model training for flow generation.

## Project Root (`./`)

*   **`.gitignore`**:
    *   **Purpose:** Specifies intentionally untracked files for Git (e.g., `__pycache__/`, `*.pyc`, `.venv/`, `downloaded_songs/`, `data/temp_*`, `data/checkpoints/flow_model_*/runs/`, `output/`).
    *   **LLM Context:** Defines the boundary between source code/configuration and generated/transient data or user-specific files.

*   **`README.md` (Main Project README)**:
    *   **Purpose:** High-level project documentation. Outlines project goals, the AI rap battle pipeline concept, data processing phases, the core Flow Generation Model architecture (Decoder-only Transformer), and development plans.
    *   **Key Content:** Overall project vision, detailed data engineering pipeline phases (0-7).
    *   **LLM Context:** **Essential for understanding the entire project's architecture, data flow, and the intended functionality of various components.** Provides the "big picture" and strategic direction.

*   **`CONTEXT.md` (This File)**:
    *   **Purpose:** This document itself. Provides detailed context on each file and directory for LLM understanding, acting as a knowledge base to reduce redundant code transmission.
    *   **LLM Context:** The primary reference for file-specific roles and interactions within the project.

*   **`print_struc.py`**:
    *   **Purpose:** A Python utility script likely used by developers to print the directory structure of the project, useful for documentation or understanding the layout.
    *   **LLM Context:** Developer utility; not part of the core data or model pipeline.

*   **`requirements.txt`**:
    *   **Purpose:** Lists all Python package dependencies required to run the project (e.g., `torch`, `librosa`, `pyphen`, `phonemizer`, `whisper-timestamped`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `python-dotenv`, `yt-dlp`, `lyricsgenius`, `soundfile`).
    *   **Role in Pipeline:** Phase 0 (Setup). Critical for environment reproducibility.
    *   **LLM Context:** Defines the software stack. Version compatibility between these libraries (especially `torch`, `librosa`, `scipy`) is crucial.

*   **`setup.bat` / `setup.sh`**:
    *   **Purpose:** Shell scripts for automating the setup of the Python virtual environment (`.venv/`) and installing dependencies from `requirements.txt` on Windows and Linux/macOS respectively.
    *   **Role in Pipeline:** Phase 0 (Setup). Foundational for running any Python scripts.
    *   **LLM Context:** User's first step to prepare their environment.

*   **`test.ipynb`**:
    *   **Purpose:** A Jupyter Notebook for interactive testing, experimentation, or debugging of various project components.
    *   **LLM Context:** Developer tool, content can vary. Not part of the automated pipeline unless explicitly run.

*   **`153 _ 253 2025 Assignment 2.pdf`**:
    *   **Purpose:** A document, likely related to an academic assignment or project specification that might have inspired or guided parts of BeefAI.
    *   **LLM Context:** External reference, potentially contains requirements or design ideas.

*   **`my_songs.txt`, `rap_data.txt`, `rap_urls.txt`**:
    *   **Purpose:** Text files likely used for input data. `my_songs.txt` and `rap_urls.txt` probably list song titles or URLs for `scripts/download_youtube_lyrics.py`. `rap_data.txt` might be a collection of rap lyrics or related textual data.
    *   **LLM Context:** Source data files for populating the `data/` directory or for other analyses.

---

## `beefai/` (Main Python Package)

This directory contains the core application logic, organized into sub-packages.

*   **`beefai/__init__.py`**:
    *   **Purpose:** Makes the `beefai` directory a Python package, allowing modules within it to be imported using `beefai.` prefix.
    *   **LLM Context:** Standard Python packaging file.

### `beefai/data_processing/`

Modules responsible for preparing audio and text data for the flow generation model.

*   **`data_processing/__init__.py`**:
    *   **Purpose:** Makes `data_processing` a sub-package.

*   **`data_processing/audio_processor.py` (`AudioProcessor`)**:
    *   **Purpose:** Handles basic audio loading, resampling, and extraction of high-level beat information like BPM and beat/downbeat times.
    *   **Key Functionality:**
        *   `load_audio()`: Loads audio files using `librosa`, supports resampling and mono conversion.
        *   `get_beat_info()`: Estimates tempo and beat times from a waveform using `librosa.beat.beat_track`. Provides a simpler beat analysis compared to `BeatFeatureExtractor`.
    *   **Outputs:** `AudioData` (waveform, sr), `BeatInfo` (TypedDict with bpm, beat_times, etc.).
    *   **Dependencies:** `librosa`, `numpy`, `scipy` (indirectly via librosa), `beefai.utils.data_types`.
    *   **LLM Context:** Used for general audio tasks and by components like `rhythm_visualizer.py` for timing click tracks. It's less detailed than `BeatFeatureExtractor` for model conditioning but useful for initial analysis or simpler applications.

*   **`data_processing/beat_feature_extractor.py` (`BeatFeatureExtractor`)**:
    *   **Purpose:** Extracts detailed, bar-level beat features from instrumental audio. These features are **critical inputs for conditioning the flow Transformer model.**
    *   **Key Functionality:**
        *   Loads instrumental audio and, if available, separated stems (bass, drums) from a specified directory structure (e.g., `data/temp_demucs_separated/htdemucs_ft/<song_id>/`).
        *   Detects global tempo and beat times using `librosa`.
        *   Estimates downbeats (bar starts), typically assuming 4/4 time.
        *   Performs onset detection for different instruments (kick, snare, hi-hat, bass) using `librosa.onset.onset_detect`. Uses instrument-specific stems if available, otherwise falls back to the full mix (less accurate).
        *   Quantizes detected onsets to a fixed number of subdivisions per bar (e.g., 16 subdivisions).
    *   **Inputs:** Path to instrumental audio, optionally a path to a directory containing its stems.
    *   **Outputs:** `SongBeatFeatures` (a list of `BarBeatFeatures` TypedDicts, one per bar). Each `BarBeatFeatures` contains `bar_index`, `bpm`, `time_signature`, and lists of active subdivision indices for `kick_events`, `snare_events`, `hihat_events`, `bass_events`.
    *   **Dependencies:** `librosa`, `numpy`, `beefai.utils.data_types`.
    *   **Role in Pipeline:** Phase 2 (Beat Feature Engineering). Consumed by `scripts/preprocess_dataset.py`.
    *   **LLM Context:** The quality of these features heavily impacts the flow model's ability to generate rhythmically coherent outputs. The current implementation is a STUB for actual drum transcription from stems, relying on `librosa.onset.onset_detect` which is a general onset detector, not a specific drum transcriber. Accuracy for distinct drum elements depends on stem quality and onset detection parameters.

*   **`data_processing/flow_data_extractor.py` (`FlowDataExtractor`)**:
    *   **Purpose:** Extracts `FlowData` (syllable counts, timings per line, and syllable start positions) from acapellas, using word-level alignment data. This data forms the **target sequences for training the flow Transformer model.**
    *   **Key Functionality:**
        *   `parse_whisper_timestamped_json()`: Parses JSON output from `whisper-timestamped` (or similar aligners) to get `LyricsData` (list of `WordTiming` dicts).
        *   `_segment_words_into_phrases()`: Groups timed words into phrases/lines based on pause durations.
        *   `_estimate_syllable_timings_for_word()`: Uses `TextProcessor` to get syllables for a word and distributes the word's duration among them to estimate per-syllable start/end times.
        *   `_create_flow_datum_for_bar_segment()`: For a given phrase/line segment and its target bar:
            *   Calculates total syllables.
            *   Calculates `start_offset_beats` (relative to bar start) and `duration_beats`.
            *   Quantizes the start time of each estimated syllable to a bar subdivision index (0-15).
    *   **Inputs:** Path to an alignment JSON file, `SongBeatFeatures` (from `BeatFeatureExtractor` for bar timing context), configuration for subdivisions per bar.
    *   **Outputs:** `FlowData` (a list of `FlowDatum` TypedDicts). Each `FlowDatum` contains `bar_index`, `line_index_in_bar`, `syllables`, `start_offset_beats`, `duration_beats`, and `syllable_start_subdivisions`.
    *   **Dependencies:** `beefai.data_processing.text_processor.TextProcessor`, `beefai.utils.data_types`.
    *   **Role in Pipeline:** Phase 3 (Flow Data Engineering). Consumed by `scripts/preprocess_dataset.py`.
    *   **LLM Context:** The accuracy of generated `FlowData` is highly dependent on the quality of the input word alignments and the syllable estimation logic. `DEBUG_FLOW_EXTRACTOR` flag enables verbose logging.

*   **`data_processing/text_processor.py` (`TextProcessor`)**:
    *   **Purpose:** Handles text-specific tasks, primarily syllable counting and phoneme generation.
    *   **Key Functionality:**
        *   `count_syllables_in_word()`: Uses `pyphen` for dictionary-based hyphenation to count syllables. Includes a naive fallback if `pyphen` fails.
        *   `get_syllables_from_word()`: Splits a word into syllable strings using `pyphen`.
        *   `get_phonemes()`: Converts text to phonemes using the `phonemizer` library with `espeak-ng` as the backend. Includes robust error handling and setup for `espeak-ng`, especially on Windows (checking `ESPEAK_NG_PATH`, common installation locations, and attempting `EspeakWrapper.set_library()`).
    *   **Dependencies:** `pyphen`, `phonemizer` (optional, with `espeak-ng` as a system dependency for phonemizer).
    *   **LLM Context:** Syllable counting is essential for `FlowDataExtractor`. Phoneme generation is available but might not be directly used by the current flow model; it's more relevant for advanced TTS/SVS. `PHONEMIZER_BACKEND_INITIALIZED` flag tracks if `espeak-ng` was successfully interfaced.

### `beefai/evaluation/`

Modules for evaluating model outputs or visualizing data.

*   **`evaluation/__init__.py`**:
    *   **Purpose:** Makes `evaluation` a sub-package.

*   **`evaluation/rhythm_visualizer.py`**:
    *   **Purpose:** Generates an audio "click track" to visualize the rhythmic placement of syllables predicted by the flow generation model, overlaying these clicks on the original instrumental.
    *   **Key Functionality:**
        *   Loads a trained `FlowTransformerDecoder` model checkpoint and its configuration.
        *   Loads `FlowTokenizer`.
        *   Loads an instrumental audio file and extracts its `BeatInfo` (using `AudioProcessor`) and `BarBeatFeatures` (using `BeatFeatureExtractor` as a prompt for the model).
        *   Uses the model's `generate()` method to produce a sequence of flow tokens.
        *   Decodes these tokens back into `FlowData` using `FlowTokenizer.decode_flow_tokens_to_datum()`.
        *   `generate_syllable_click_track()`: Creates an audio waveform with clicks at the start time of each syllable specified in the decoded `FlowData`, aligned with the instrumental's timing.
    *   **Inputs:** Model checkpoint path, model config path, tokenizer config path, instrumental audio path, generation parameters.
    *   **Outputs:** A WAV file saved to `output/flow_visualizations/` containing the instrumental mixed with syllable clicks.
    *   **Dependencies:** `torch`, `soundfile`, `beefai.flow_model.tokenizer.FlowTokenizer`, `beefai.flow_model.transformer_model.FlowTransformerDecoder`, `beefai.data_processing.audio_processor.AudioProcessor`, `beefai.data_processing.beat_feature_extractor.BeatFeatureExtractor`, `beefai.utils.data_types`.
    *   **LLM Context:** A key tool for qualitative assessment of the flow model's rhythmic output. It tests the end-to-end generation and decoding process.

### `beefai/flow_model/`

Core components of the flow generation Transformer model.

*   **`flow_model/__init__.py`**:
    *   **Purpose:** Makes `flow_model` a sub-package.

*   **`flow_model/dataset.py` (`FlowDataset`)**:
    *   **Purpose:** A PyTorch `Dataset` class for loading and serving pre-tokenized song data for training the `FlowTransformerDecoder`.
    *   **Key Functionality:**
        *   Loads data from `.pt` files (e.g., `train_lite.pt`, `train_full.pt`) which contain lists of dictionaries, each dict holding `token_ids`, `segment_ids`, and `intra_line_pos_ids` for a song.
        *   Chunks these long song sequences into smaller, fixed-size blocks (`block_size + 1`) suitable for Transformer training.
        *   For each chunk, it creates `input_ids` (chunk[:-1]) and `target_ids` (chunk[1:]), along with corresponding segment and intra-line position IDs.
    *   **Inputs:** Path to a tokenized data file, tokenizer's pad ID, model's `block_size`. Can also take `direct_data` (list of dicts) instead of a file path.
    *   **Outputs:** Dictionaries of tensors (`input_ids`, `target_ids`, `segment_ids`, `intra_line_pos_ids`) for the DataLoader.
    *   **Dependencies:** `torch`.
    *   **Role in Pipeline:** Consumes output from `scripts/05a_tokenize_data_lite.py` or `scripts/05b_tokenize_data_full.py`. Used by `lite_model_training/train_lite_flow_model.py` and `scripts/train_flow_model.py`.
    *   **LLM Context:** Bridges pre-tokenized data files and the model training loop by preparing input/target pairs in the correct format and length.

*   **`flow_model/model.py` (`FlowModel`)**:
    *   **Purpose:** Represents an older, simpler, or placeholder model for flow generation. **This is NOT the main Transformer model used for training.**
    *   **Key Functionality:** Contains a stub `generate_flow()` method that produces dummy `FlowData`. Its `train()` method raises `NotImplementedError`.
    *   **LLM Context:** Largely a legacy or illustrative component. The primary focus for flow generation is `transformer_model.py`.

*   **`flow_model/tokenizer.py` (`FlowTokenizer`)**:
    *   **Purpose:** Converts raw `BarBeatFeatures` (from instrumentals) and `FlowData` (from acapellas) into token sequences suitable for the Transformer model, and decodes token sequences back into structured data. Manages the specialized vocabulary.
    *   **Key Functionality:**
        *   Defines and manages a vocabulary including special tokens (`[PAD]`, `[BOS]`, `[EOS]`, `[BAR_START]`, `[LINE_START]`, `[SEP_INPUT_FLOW]`, `[END_SYLLABLE_SUBDIVISIONS]`), tokens for BPM bins, time signatures, percussive events (e.g., `[KICK_AT_0]`, `[NO_KICK_EVENTS]`, `[END_KICK_EVENTS]`), flow attributes (e.g., `[SYLLABLES_5]`, `[OFFSET_BIN_2]`, `[DURATION_BIN_8]`), and syllable start subdivisions (e.g., `[SYLLABLE_STARTS_SUBDIV_3]`).
        *   `encode_bar_features()`: Converts a `BarBeatFeatures` dict into a list of token IDs.
        *   `encode_flow_datum()`: Converts a `FlowDatum` dict into a list of token IDs.
        *   `encode_song_instance()`: Orchestrates the tokenization of a full song, combining beat features and flow data for multiple bars. Generates three lists: `token_ids`, `segment_ids` (to distinguish beat feature blocks from flow blocks, and bar-to-bar progression), and `intra_line_pos_ids` (to mark positions within a conceptual "line" of tokens, like within a bar's features or a flow line's attributes).
        *   `decode_flow_tokens_to_datum()`: Converts a list of flow-related token IDs back into a structured `FlowDatum` dictionary. Used during inference/evaluation.
        *   Saves and loads its vocabulary and configuration parameters (like bin sizes, max values) from/to a JSON file (e.g., `flow_tokenizer_config_v2.json`).
    *   **Inputs (for encoding):** `BarBeatFeatures`, `FlowDatum`.
    *   **Outputs (for encoding):** Lists of integer token IDs, segment IDs, and intra-line position IDs.
    *   **Role in Pipeline:** Phase 4 (Tokenization). Used by `scripts/05a_tokenize_data_lite.py`, `scripts/05b_tokenize_data_full.py`, training scripts (indirectly via `FlowDataset`), and `evaluation/rhythm_visualizer.py`.
    *   **LLM Context:** This is a **critical and complex component**. The design of its vocabulary and the logic for `segment_ids` and `intra_line_pos_ids` are fundamental to how the Transformer learns and generates structured flow. The `config_params` in its JSON define quantization strategies.

*   **`flow_model/transformer_model.py` (`FlowTransformerDecoder`, `FlowGPTConfig`)**:
    *   **Purpose:** Implements the main Decoder-only Transformer architecture for generating rap flow sequences.
    *   **Key Components:**
        *   `FlowGPTConfig`: A dataclass-like configuration object holding hyperparameters for the model (e.g., `vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`, `max_segment_types`, `max_intra_line_positions`, `dropout`, `pad_token_id`).
        *   `CausalSelfAttention`: Standard multi-head self-attention with causal masking (supports Flash Attention if available).
        *   `MLP`: Feed-forward network component of a Transformer block.
        *   `Block`: A single Transformer decoder block (self-attention + MLP, with layer norms and residual connections).
        *   `FlowTransformerDecoder`: The main model class.
            *   Embeddings: Includes separate embedding layers for token IDs (`wte`), absolute positions (`wpe`), segment type IDs (`wse`), and intra-line/intra-segment position IDs (`wipe`). These are summed to form the input to the Transformer blocks.
            *   `forward()`: Takes `idx` (token IDs), `segment_ids`, `intra_line_pos_ids`, and optional `targets`. Returns logits and computes cross-entropy loss if targets are provided.
            *   `generate()`: Performs autoregressive generation of token sequences given a prompt (initial `idx_prompt`, `segment_ids_prompt`, `intra_line_pos_ids_prompt`). Uses `get_next_context_ids_for_token()` to determine appropriate segment and intra-line position IDs for newly generated tokens. Supports temperature scaling and top-k sampling.
        *   `get_next_context_ids_for_token()`: A crucial helper function used during generation. Given the sequence of previously generated tokens and the new token being added, it infers the correct `segment_id` and `intra_line_pos_id` for the new token based on special tokens like `[BAR_START]`, `[SEP_INPUT_FLOW]`, `[LINE_START]`. This logic mirrors how these contexts are constructed during `FlowTokenizer.encode_song_instance()`.
    *   **Inputs (training):** Batches of `input_ids`, `segment_ids`, `intra_line_pos_ids`, `target_ids` from `FlowDataset`.
    *   **Outputs (inference):** A sequence of generated token IDs.
    *   **Dependencies:** `torch`.
    *   **Role in Pipeline:** The core generative model. Trained by `lite_model_training/train_lite_flow_model.py` and `scripts/train_flow_model.py`. Used for inference by `evaluation/rhythm_visualizer.py`.
    *   **LLM Context:** The architecture's key features are its decoder-only nature and its use of multiple, summed embeddings to provide rich contextual information (what the token is, where it is absolutely, what kind of segment it belongs to, and its position within that micro-sequence). The `generate()` method's interaction with `get_next_context_ids_for_token()` is vital for coherent structured output.

*   **`flow_model/flow_tokenizer_config_v2.json`**:
    *   **Purpose:** JSON file storing the vocabulary (`token_to_id`, `id_to_token`) and configuration parameters (`max_syllables`, `num_offset_bins`, `bpm_bins`, `max_subdivisions`, etc.) for the `FlowTokenizer`.
    *   **LLM Context:** Defines the discrete symbols the model learns to predict. It's generated/updated by `FlowTokenizer.save_vocab()` and read by `FlowTokenizer.__init__()`. The `v2` indicates an evolution of the tokenization scheme.

### `beefai/lyric_generation/`

Modules related to generating lyrical content.

*   **`lyric_generation/__init__.py`**:
    *   **Purpose:** Makes `lyric_generation` a sub-package.

*   **`lyric_generation/agent.py` (`LyricAgent`)**:
    *   **Purpose:** Intended to house the logic for generating rap lyrics, likely by interfacing with a large language model (LLM).
    *   **Key Functionality (Current):** Placeholder. The `generate_verse()` method returns dummy/placeholder lyrics matching the expected syllable counts from `FlowData`. It explicitly states that LLM integration is required.
    *   **LLM Context:** This is a stub component. Actual lyric generation is a significant separate task. The current implementation allows the rest of the pipeline (e.g., a demo in `main.py`) to run without a real LLM.

### `beefai/synthesis/`

Modules related to synthesizing rap vocals from lyrics and flow information.

*   **`synthesis/__init__.py`**:
    *   **Purpose:** Makes `synthesis` a sub-package.

*   **`synthesis/synthesizer.py` (`RapSynthesizer`)**:
    *   **Purpose:** Intended to synthesize audible rap performances from lyrics and `FlowData`. This would typically involve Text-to-Speech (TTS) or Singing Voice Synthesis (SVS) technology.
    *   **Key Functionality (Current):** Placeholder.
        *   `synthesize_line()`: Returns silent audio of the expected duration based on `FlowDatum` and BPM.
        *   `synthesize_verse()`: Combines silent/stub lines according to `FlowData` and `BeatInfo` timing. If timing info is insufficient, it concatenates lines with fixed pauses. Marks line starts with a small pulse for audibility in the otherwise silent stub output.
    *   **LLM Context:** This is a stub component. Real voice synthesis is a complex, separate ML task. The placeholder allows testing the timing and structuring aspects of a full pipeline.

### `beefai/utils/`

Utility modules and data type definitions.

*   **`utils/__init__.py`**:
    *   **Purpose:** Makes `utils` a sub-package.

*   **`utils/data_types.py`**:
    *   **Purpose:** Defines common Python `TypedDict` structures used throughout the project for type hinting and ensuring data consistency.
    *   **Key Content:**
        *   `AudioData`: `Tuple[np.ndarray, int]` (waveform, sample_rate)
        *   `BeatInfo`: Dict for BPM, beat times, downbeat times.
        *   `BarBeatFeatures`: Dict for detailed per-bar instrumental features (bpm, time_sig, kick/snare/hihat/bass events as lists of subdivision indices).
        *   `SongBeatFeatures`: `List[BarBeatFeatures]`.
        *   `WordTiming`: Dict for word, start_time, end_time.
        *   `LyricsData`: `List[WordTiming]`.
        *   `FlowDatum`: Dict for per-line flow information (`bar_index`, `line_index_in_bar`, `syllables`, `start_offset_beats`, `duration_beats`, `syllable_start_subdivisions`).
        *   `FlowData`: `List[FlowDatum]`.
        *   `TrainingInstance`: Dict combining `song_id`, `beat_features` (`SongBeatFeatures`), and `flow_targets` (`FlowData`) for a single song, used as the primary data structure saved by `preprocess_dataset.py`.
    *   **LLM Context:** **Crucial for understanding the data structures passed between different modules.** This file acts as a schema for much of the project's data.

### `beefai/webapp/`

Files for a potential web application interface.

*   `index.html`, `static/css/style.css`, `static/js/main.js`: Standard web frontend files.
*   **LLM Context:** Relates to the user interface aspect of the project, not directly to the core model training or data processing pipelines unless the web app is used to trigger those.

---

## `data/` Directory

Central storage for all datasets, intermediate processing outputs, and model inputs/outputs.

*   **`acapellas/`**: Stores acapella audio files (vocals only), typically `.mp3` or `.wav`. Output of source separation.
*   **`instrumentals/`**: Stores instrumental audio files (no vocals), typically `.mp3` or `.wav`. Output of source separation or directly provided.
*   **`lyrics/`**: Raw lyric text files (`.txt`), with filenames matching corresponding songs.
*   **`raw_songs_full/`**: Original, complete song audio files (input for source separation).
*   **`alignments_json/`**: Contains JSON files with word-level (and potentially phoneme-level) timestamps. These are typically outputs from a forced aligner like `whisper-timestamped`. Filenames match songs (e.g., `A.D.H.D.json`). **Crucial input for `FlowDataExtractor`**.
*   **`alignments_textgrid/`**: Likely for storing Praat TextGrid alignment files, an alternative or intermediate format for alignments.
*   **`checkpoints/`**: Directory for saving trained model checkpoints.
    *   **`flow_model_full/`**: Checkpoints for the "full" version of the `FlowTransformerDecoder`. Contains `.pt` files (e.g., `full_ckpt_epoch_X.pt`, `full_final_model.pt`).
        *   `runs/`: Subdirectory for TensorBoard logs related to full model training (e.g., `full_experiment_YYYYMMDD-HHMMSS/events.out.tfevents...`).
    *   **`flow_model_lite/`**: Checkpoints and TensorBoard logs for the "lite" version of the model.
*   **`processed_for_transformer/`**: **Key directory for data that has been feature-extracted but not yet tokenized.**
    *   **`processed_training_data.pt`**: **Primary output of `scripts/preprocess_dataset.py`**. A single `.pt` file containing a list of `TrainingInstance` dictionaries. Each `TrainingInstance` holds the `song_name`, `beat_features` (`SongBeatFeatures`), and `flow_data` (`FlowData`) for one song. This file is the main input to the tokenization scripts (`05a_tokenize_data_lite.py`, `05b_tokenize_data_full.py`).
    *   **`beat_features_cache/`**: Stores cached `SongBeatFeatures` for individual songs (e.g., `A.D.H.D_beat_features.pt`) to speed up reprocessing by `scripts/preprocess_dataset.py`.
    *   **`flow_data_cache/`**: Stores cached `FlowData` for individual songs (e.g., `A.D.H.D_flow_data.pt`) to speed up reprocessing by `scripts/preprocess_dataset.py`.
    *   **`stems_cache/`**: This directory seems to store pre-separated audio stems, organized by a separation model (e.g., `htdemucs_ft`) and then by song ID. Example: `stems_cache/htdemucs_ft/A.D.H.D/bass.wav`. `BeatFeatureExtractor` would use these if available.
*   Other subdirectories like `arpa/` (for language model data like `.dict` files) and various data files (`.pdf`, `.txt`) in the root suggest diverse data sources or experimental components.

---

## `downloaded_songs/`

*   **Purpose:** Default output directory for `scripts/download_youtube_lyrics.py`. Contains downloaded `.mp3` audio files and their corresponding fetched `.txt` lyric files.
*   **LLM Context:** A staging area for newly acquired raw data before it's organized into the main `data/` structure by `scripts/organize_downloaded_songs.py`.

---

## `lite_model_training/`

Files specific to configuring and training a "lite" (smaller, faster to train) version of the flow model. Also contains configs for the "full" model.

*   **`data_config_lite.yaml` / `data_config_full.yaml`**:
    *   **Purpose:** YAML configuration files defining data-related paths and parameters for the "lite" and "full" model training pipelines, respectively.
    *   **Key Content:**
        *   `tokenizer_path`: Path to `flow_tokenizer_config_v2.json`.
        *   `processed_data_source_dir`: Path to the directory containing `processed_training_data.pt` (i.e., `data/processed_for_transformer/`).
        *   Output paths for tokenized data (e.g., `data/tokenized_lite/train_lite.pt`, `data/tokenized_full/val_full.pt`).
        *   `max_songs_for_lite`/`_full`: Max number of songs to use for the dataset (-1 for all).
        *   `val_split_ratio_for_lite`/`_full`: Proportion for validation split.
        *   `checkpoint_dir`: Where to save model checkpoints during training.
    *   **Interactions:** Read by `scripts/05a_tokenize_data_lite.py`, `scripts/05b_tokenize_data_full.py`, `lite_model_training/train_lite_flow_model.py`, and `scripts/train_flow_model.py`.
    *   **LLM Context:** Centralizes data path management for different dataset scales and training runs.

*   **`model_config_lite.yaml` / `model_config_full.yaml`**:
    *   **Purpose:** YAML configuration files specifying the architecture hyperparameters for the "lite" and "full" `FlowTransformerDecoder` models, respectively.
    *   **Key Content:** `block_size`, `n_layer`, `n_head`, `n_embd`, `max_segment_types`, `max_intra_line_positions`, `dropout`, `bias`. May also include training-specific parameters like `batch_size`, `learning_rate`, `epochs`.
    *   **Interactions:** Read by `lite_model_training/train_lite_flow_model.py` and `scripts/train_flow_model.py` to instantiate `FlowGPTConfig` and set up the training loop.
    *   **LLM Context:** Defines the model's capacity and complexity. `max_segment_types` and `max_intra_line_positions` are crucial for the tokenizer and model to agree on context ID ranges.

*   **`train_lite_flow_model.py`**:
    *   **Purpose:** Python script dedicated to training the "lite" version of the `FlowTransformerDecoder` model.
    *   **Key Functionality:**
        *   Loads model and data configurations from `model_config_lite.yaml` and `data_config_lite.yaml`.
        *   Initializes `FlowTokenizer`, `FlowGPTConfig`, `FlowTransformerDecoder`, and `FlowDataset` (for `train_lite.pt` and `val_lite.pt`).
        *   Implements the PyTorch training loop: optimizer (AdamW), learning rate scheduler (OneCycleLR), loss calculation (cross-entropy, respecting `pad_token_id`), gradient accumulation.
        *   Handles model evaluation on a validation set (calculating loss and perplexity).
        *   Manages model checkpointing (saving model state, optimizer state, etc.).
        *   Integrates with TensorBoard for logging metrics (`runs` subdirectories are created under the checkpoint directory).
        *   Supports Automatic Mixed Precision (AMP) and `torch.compile()` if available.
    *   **Inputs:** Tokenized "lite" data files, tokenizer config, "lite" model config.
    *   **Outputs:** Trained model checkpoints (e.g., in `data/checkpoints/flow_model_lite/`) and TensorBoard logs.
    *   **LLM Context:** The primary script for training the smaller example model. Demonstrates a full training pipeline.

---

## `output/`

General-purpose directory for outputs generated by the project that aren't raw data or checkpoints.

*   **`beefai_default_instrumental.wav`**: A default or example instrumental audio file, possibly used by demos or tests.
*   **`flow_visualizations/`**:
    *   **Purpose:** Output directory for `beefai/evaluation/rhythm_visualizer.py`.
    *   **Content:** WAV files containing instrumentals mixed with click tracks representing model-generated flow rhythms.
    *   **LLM Context:** Contains audio results for qualitative evaluation of the flow model.

---

## `scripts/` Directory

Contains utility scripts and scripts that form parts of the main data processing and model training pipelines.

*   **`05a_tokenize_data_lite.py` / `05b_tokenize_data_full.py`**:
    *   **Purpose:** Scripts to tokenize the preprocessed data (output of `scripts/preprocess_dataset.py`) for the "lite" and "full" models, respectively.
    *   **Key Functionality:**
        *   Load `processed_training_data.pt` (which contains `TrainingInstance` dicts).
        *   Load `FlowTokenizer` using the path from the respective `data_config_*.yaml` file.
        *   Load `model_config_*.yaml` to get `max_segment_types` and `max_intra_line_positions` for validation.
        *   Select a subset of songs (for "lite") or all songs (for "full", configurable).
        *   For each song, use `tokenizer.encode_song_instance()` to get `token_ids`, `segment_ids`, `intra_line_pos_ids`.
        *   **Crucially, validate** that the maximum `segment_id` and `intra_line_pos_id` generated for any song do not exceed the limits defined in the model's configuration (`config_max_segment_types`, `config_max_intra_line_positions`). Songs exceeding these limits are skipped.
        *   Split the tokenized songs into training and validation sets.
        *   Save these sets as lists of dictionaries (each dict containing PyTorch tensors for `token_ids`, `segment_ids`, `intra_line_pos_ids`) to `.pt` files (e.g., `train_lite.pt`, `val_full.pt`) in the directory specified by `tokenized_data_output_dir` from the data config.
    *   **Inputs:** `processed_training_data.pt`, `data_config_*.yaml`, `model_config_*.yaml`, tokenizer config.
    *   **Outputs:** Tokenized training and validation `.pt` files (e.g., `data/tokenized_lite/train_lite.pt`).
    *   **Role in Pipeline:** Phase 4 (Tokenization). These scripts produce the final data files consumed by `FlowDataset` during model training.
    *   **LLM Context:** The validation check against model config limits is vital to prevent errors during training if the tokenizer generates context IDs outside the embedding table ranges of the model.

*   **`download_youtube_lyrics.py`**:
    *   **Purpose:** Downloads audio from YouTube URLs (provided in a file) using `yt-dlp` and attempts to fetch corresponding lyrics using `lyricsgenius`.
    *   **Key Functionality:** Sanitizes filenames, attempts to clean lyric metadata, saves audio as MP3 and lyrics as TXT to `downloaded_songs/`. Creates a CSV report of successes/failures.
    *   **Dependencies:** `yt-dlp`, `lyricsgenius`, `python-dotenv` (for `GENIUS_ACCESS_TOKEN`).
    *   **LLM Context:** An optional data acquisition script. Relies on the `GENIUS_ACCESS_TOKEN` environment variable.

*   **`inspect_tokenized_data.py`**:
    *   **Purpose:** A utility script to load and inspect the contents of the `.pt` files generated by `05a_tokenize_data_lite.py` or `05b_tokenize_data_full.py`.
    *   **Key Functionality:**
        *   Loads a specified `.pt` file.
        *   Prints information about the data structure (e.g., number of items, type of items).
        *   For each item (tokenized song), it shows keys, tensor shapes, dtypes, and basic stats (min/max/mean).
        *   If a `FlowTokenizer` config is provided and the tokenizer is available, it can decode and print the first few tokens of each sequence.
        *   Checks for length consistency between `token_ids`, `segment_ids`, and `intra_line_pos_ids`.
    *   **LLM Context:** Useful for debugging the tokenization process and verifying the structure of data being fed to `FlowDataset`.

*   **`organize_downloaded_songs.py`**:
    *   **Purpose:** Moves audio and lyric files from a source directory (e.g., `downloaded_songs/`) into the structured `data/` subdirectories (`data/raw_songs_full/`, `data/lyrics/`).
    *   **LLM Context:** Utility script for managing data after acquisition.

*   **`preprocess_dataset.py`**:
    *   **Purpose:** Core script for Phases 2 & 3 of the data pipeline. It processes raw/separated audio and alignment data to extract `SongBeatFeatures` and `FlowData` for each song.
    *   **Key Functionality:**
        *   Iterates through instrumental audio files and finds corresponding alignment JSON files.
        *   Determines paths to pre-separated stems for each song based on `--separator_tool_used` and `--demucs_model_name` arguments (e.g., looks in `data/temp_demucs_separated/htdemucs_ft/<song_id>/`).
        *   Uses `BeatFeatureExtractor` to get `SongBeatFeatures` (instrumental analysis).
        *   Uses `FlowDataExtractor` to get `FlowData` (acapella/alignment analysis).
        *   Caches extracted `SongBeatFeatures` and `FlowData` for individual songs in subdirectories of `processed_output_dir` (e.g., `beat_features_cache/`, `flow_data_cache/`) to speed up subsequent runs. Can be forced to reprocess.
        *   Aggregates the results for all successfully processed songs into a list of `TrainingInstance` dictionaries.
        *   Saves this list as a single `.pt` file (e.g., `data/processed_for_transformer/processed_training_data.pt`).
    *   **Inputs:** Instrumentals directory, alignments JSON directory, output directory, paths/info for pre-separated stems.
    *   **Outputs:** `processed_training_data.pt` (the main collective output) and individual feature caches.
    *   **LLM Context:** This is a **critical data preparation script**. Its outputs are the direct input to the tokenization scripts. The quality of its feature extraction heavily influences model performance. The script now has an argument `--debug_single_song_id` for easier focused debugging.

*   **`run_full_data_pipeline.py`**:
    *   **Purpose:** Master script designed to orchestrate the entire data preparation pipeline, from raw songs to tokenized data ready for model training.
    *   **Key Functionality:** Sequentially calls or executes other components/scripts:
        1.  Optional data acquisition/organization steps (commented out by default, but structure is present).
        2.  Source separation (e.g., using `demucs` via `subprocess`). Iterates through raw songs, runs Demucs, and moves separated vocals/instrumental to `data/acapellas/` and `data/instrumentals/`.
        3.  Forced alignment (e.g., using `whisper-timestamped` via `subprocess`). Iterates through acapellas and generates alignment JSONs in `data/alignments_json/`.
        4.  Calls `scripts/preprocess_dataset.py` to extract beat and flow features, creating `processed_training_data.pt`.
        5.  Calls `scripts/05a_tokenize_data_lite.py` to create tokenized data for the lite model.
        6.  Calls `scripts/05b_tokenize_data_full.py` to create tokenized data for the full model.
    *   **LLM Context:** Intended as the main user-facing script to automate most of the dataset preparation. It manages dependencies between steps and provides a high-level execution flow. Uses command-line arguments to control which phases are run and to specify key directory paths.

*   **`test.py`**:
    *   **Purpose:** A simple Python script, often used for quick, isolated tests, like checking environment variables (e.g., `GENIUS_ACCESS_TOKEN`).
    *   **LLM Context:** Ad-hoc developer testing, not part of the main pipeline.

*   **`train_flow_model.py`**:
    *   **Purpose:** Python script for training the "full" version of the `FlowTransformerDecoder` model.
    *   **Key Functionality:** Similar to `lite_model_training/train_lite_flow_model.py`, but:
        *   Loads model and data configurations from `model_config_full.yaml` and `data_config_full.yaml`.
        *   Uses tokenized data intended for the full model (e.g., `train_full.pt`, `val_full.pt`).
        *   Saves checkpoints and TensorBoard logs to directories associated with the full model (e.g., `data/checkpoints/flow_model_full/`).
    *   **LLM Context:** The script for training the primary, larger flow generation model.

---

This detailed `CONTEXT.md` should serve as a robust reference for understanding the BeefAI project structure and the roles of its constituent files, particularly for guiding an LLM.