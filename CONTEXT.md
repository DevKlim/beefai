# beefai/CONTEXT.md
# BeefAI Project: File Context for LLM (Master Control Program Document)

This document provides a comprehensive summary of files and directories within the BeefAI project. Its purpose is to serve as a persistent knowledge base for an LLM, outlining each component's purpose, interactions, data flow, and role in the overall AI Rap Battle pipeline, particularly focusing on data processing and model training for flow generation.

## Project Root (`./`)

*   **`.gitignore`**:
    *   **Purpose:** Specifies intentionally untracked files for Git (e.g., `__pycache__/`, `*.pyc`, `.venv/`, `downloaded_songs/`, `downloaded_files/`, `data/temp_*`, `data/checkpoints/flow_model_*/runs/`, `output/`).
    *   **LLM Context:** Defines the boundary between source code/configuration and generated/transient data or user-specific files.

*   **`README.md` (Main Project README)**:
    *   **Purpose:** High-level project documentation. Outlines project goals, the AI rap battle pipeline concept (with an **updated Mermaid diagram reflecting the V5 synthesis flow**), data processing phases, the core Flow Generation Model architecture, and development plans.
    *   **Key Content:** Overall project vision, detailed data engineering pipeline phases (0-7).
    *   **LLM Context:** **Essential for understanding the entire project's architecture, data flow, and the intended functionality of various components.** Provides the "big picture" and strategic direction.

*   **`CONTEXT.md` (This File)**:
    *   **Purpose:** This document itself. Provides detailed context on each file and directory for LLM understanding, acting as a knowledge base to reduce redundant code transmission.
    *   **LLM Context:** The primary reference for file-specific roles and interactions within the project.

*   **`print_struc.py`**:
    *   **Purpose:** A Python utility script likely used by developers to print the directory structure of the project, useful for documentation or understanding the layout.
    *   **LLM Context:** Developer utility; not part of the core data or model pipeline.

*   **`requirements.txt`**:
    *   **Purpose:** Lists all Python package dependencies required to run the project (e.g., `torch`, `librosa`, `pyphen`, `phonemizer`, `whisper-timestamped`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `python-dotenv`, `yt-dlp`, `lyricsgenius`, `soundfile`, `spotipy`, `langdetect`, **`pydub`**, **`google-generativeai`**).
    *   **Role in Pipeline:** Phase 0 (Setup). Critical for environment reproducibility.
    *   **LLM Context:** Defines the software stack. `espeak-ng` is a crucial system dependency for `beefai.synthesis.RapSynthesizer`.

*   **`setup.bat` / `setup.sh`**:
    *   **Purpose:** Shell scripts for automating the setup of the Python virtual environment (`.venv/`) and installing dependencies from `requirements.txt`.
    *   **Role in Pipeline:** Phase 0 (Setup).
    *   **LLM Context:** User's first step to prepare their environment.

*   **`test.ipynb`**:
    *   **Purpose:** A Jupyter Notebook for interactive testing, experimentation, or debugging.
    *   **LLM Context:** Developer tool, content can vary.

*   **`153 _ 253 2025 Assignment 2.pdf`**:
    *   **Purpose:** External reference document, possibly project specifications.
    *   **LLM Context:** May contain requirements or design ideas.

*   **`my_songs.txt`, `rap_data.txt`, `rap_urls.txt`**:
    *   **Purpose:** Text files for input data, likely for song lists or lyric corpora.
    *   **LLM Context:** Source data for populating `data/` or other analyses.

---

## `beefai/` (Main Python Package)

*   **`beefai/__init__.py`**: Standard Python packaging file.

### `beefai/data_processing/`

*   **`data_processing/__init__.py`**: Makes `data_processing` a sub-package.

*   **`data_processing/audio_processor.py` (`AudioProcessor`)**:
    *   **Purpose:** Basic audio loading, resampling, and high-level beat info extraction.
    *   **Key Functionality:** `load_audio()`, `get_beat_info()` (using `librosa.beat.beat_track`).
    *   **LLM Context:** General audio tasks, used by `rhythm_visualizer.py` and `generate_rap_battle_response.py`.

*   **`data_processing/beat_feature_extractor.py` (`BeatFeatureExtractor`)**:
    *   **Purpose:** Extracts detailed, bar-level beat features for conditioning the flow model.
    *   **Key Functionality:** Loads instrumental (and optional stems). Detects tempo, beats, downbeats. Performs onset detection (kick, snare, hi-hat, bass) using `librosa.onset.onset_detect`. Quantizes onsets. Can now process specific audio segments using `audio_offset_sec` and `audio_duration_sec`.
    *   **Outputs:** `SongBeatFeatures` (list of `BarBeatFeatures`). `BarBeatFeatures` now includes `bar_start_time_sec` and `bar_duration_sec`.
    *   **LLM Context:** Feature quality is critical. `bar_start_time_sec` and `bar_duration_sec` are crucial for aligning generated flow with specific audio segments.

*   **`data_processing/flow_data_extractor.py` (`FlowDataExtractor`)**:
    *   **Purpose:** Extracts `FlowData` (syllable counts, timings, **stress**) from acapella alignments. This is target data for flow model training.
    *   **Key Functionality:** Parses alignment JSON. Segments words into phrases. `_estimate_syllable_details_for_word()` uses `TextProcessor` for orthographic syllables AND **stress information**. `_create_flow_datum_for_bar_segment()` calculates flow attributes, quantizes syllable starts, **adds `syllable_stresses` list to `FlowDatum`**, and uses injected `FlowTokenizer` for syllable duration quantization.
    *   **Outputs:** `FlowData` (list of `FlowDatum`). `FlowDatum` now includes `syllable_stresses`.
    *   **LLM Context:** Stress information is vital for richer prosody. Accuracy depends on alignment, syllable estimation, and stress detection from `TextProcessor`.

*   **`data_processing/text_processor.py` (`TextProcessor`)**:
    *   **Purpose:** Syllable counting, orthographic syllable splitting, phoneme/stress generation.
    *   **Key Functionality:**
        *   `count_syllables_in_word()`: Uses `pyphen` or `phonemizer`.
        *   `get_syllables_from_word()`: **Crucial for V4/V5 synthesis workflow.** Splits words into orthographic syllables (e.g., for user review of text for TTS).
        *   `get_syllables_with_stress()`: Uses `phonemizer` (if available) for orthographic syllables + stress markers. Feeds `FlowDataExtractor`.
        *   `get_phonemes()`: Uses `phonemizer`.
    *   **LLM Context:** `get_syllables_from_word` is key for the user-reviewable text step in `generate_rap_battle_response.py`. Stress detection from `get_syllables_with_stress` is used for training data and synthesis prosody.

### `beefai/evaluation/`

*   **`evaluation/__init__.py`**: Makes `evaluation` a sub-package.

*   **`evaluation/rhythm_visualizer.py`**:
    *   **Purpose:** Generates an audio "click track" to visualize model-predicted syllable rhythms.
    *   **Key Functionality:** Loads model, tokenizer, instrumental. Extracts `SongBeatFeatures` (can use offset/duration). Generates flow tokens, decodes to `FlowData`. `generate_syllable_click_track()` creates audio with percussive sounds at syllable onsets, **using syllable stress from `FlowData` to vary click properties.** `_create_beat_info_from_custom_features()` is robust for creating `BeatInfo`. `_generate_flow_core` centralizes flow generation logic.
    *   **LLM Context:** Key for qualitative assessment. Tests end-to-end generation.

### `beefai/flow_model/`

*   **`flow_model/__init__.py`**: Makes `flow_model` a sub-package.
*   **`flow_model/dataset.py` (`FlowDataset`)**: (As before - prepares tokenized data for training).
*   **`flow_model/model.py` (`FlowModel`)**: (As before - placeholder/legacy model).
*   **`flow_model/tokenizer.py` (`FlowTokenizer`)**:
    *   **Purpose:** Converts `BarBeatFeatures` and `FlowData` (now including stress) to/from token sequences.
    *   **Key Functionality:** Manages vocabulary. `encode_flow_datum()` now also encodes `syllable_stresses` (e.g., `[SYLLABLE_STRESS_0]`). `decode_flow_tokens_to_datum()` decodes stress tokens. Methods `quantize_syllable_duration_to_bin_index` and `dequantize_syllable_duration_bin` are crucial for handling syllable durations.
    *   **LLM Context:** Stress tokens enhance the model's ability to learn prosodic variations.
*   **`flow_model/transformer_model.py` (`FlowTransformerDecoder`, `FlowGPTConfig`)**: (As before - main Transformer architecture).
*   **`flow_model/flow_tokenizer_config_v2.json`**: (As before - tokenizer vocabulary and config).

### `beefai/lyric_generation/`

*   **`lyric_generation/__init__.py`**: Makes `lyric_generation` a sub-package.

*   **`lyric_generation/agent.py` (`LyricAgent`)**:
    *   **Purpose:** Manages interaction with an LLM (e.g., Gemini API via `google-generativeai`) to generate rap lyrics.
    *   **Key Functionality:**
        *   `generate_response_verse_via_api()`:
            *   Constructs a **concise, token-efficient prompt** for the Gemini API. Specifies persona, diss, theme, and **Target Orthographic Syllable Count** per line. Includes brief pacing guidance.
            *   Calls the LLM API.
            *   Parses the response.
            *   **Automated Retry Logic:** If generated lines fail syllable count checks (within tight tolerance), it re-prompts the API to regenerate *only failed lines*, strongly emphasizing syllable adherence.
            *   Uses stubs for lines still failing after retries.
        *   `_parse_llm_lyric_response_lines()`: Parses LLM text, includes `original_flow_datum_index` for retry tracking.
    *   **Output:** List of `[{"lyric": str, "target_syllables": int, "original_flow_datum_index": int}, ...]`.
    *   **LLM Context:** Provides a robust method for direct API calls for lyric generation, aiming for strict syllable adherence. The model specified by user (`gemini-2.5-flash-preview-05-20` or other) is used.

*   **`lyric_generation/prompt_formatter.py` (`PromptFormatter`)**:
    *   **Purpose:** Creates detailed prompts to guide a *human user* in their interaction with a separate chat-based LLM (like Gemini Advanced, Claude, ChatGPT) for lyric generation when direct API calls are not used/preferred.
    *   **Key Functionality (`format_lyric_generation_prompt_V2`):**
        *   Outlines a **two-stage process** for the user: Stage 1 (Creative Verse), Stage 2 (Alignment to orthographic syllable counts from `FlowData`).
    *   **LLM Context:** Used in `generate_rap_battle_response.py` for the manual lyric generation path.

### `beefai/synthesis/`

*   **`synthesis/__init__.py`**: Makes `synthesis` a sub-package.

*   **`synthesis/synthesizer.py` (`RapSynthesizer`)**:
    *   **Purpose:** Synthesizes audible rap. Uses `espeak-ng` for *entire lyric lines*, segments audio based on `FlowData` timing, applies stress-based prosody, and mixes with instrumental.
    *   **Key Functionality:**
        *   `_generate_line_audio_espeak()`: Generates audio for a full lyric line via `espeak-ng`.
        *   `synthesize_vocal_track()` (formerly `synthesize_phonetic_track`):
            *   Input: List of full lyric strings (`lyric_lines_for_tts`), `FlowData`, `BeatInfo`.
            *   For each line:
                1.  Generates audio for the whole line (cached).
                2.  Stretches/compresses line audio to match total `FlowData` duration for that line.
                3.  Segments stretched audio into syllable chunks based on `FlowData` start times/durations.
                4.  **Stress Modulation:** Applies pitch shift (`librosa.effects.pitch_shift`) and gain to syllable segments based on `syllable_stresses` from `FlowData` (controlled by constructor/script parameters).
                5.  Applies fades, overlays onto master vocal track.
        *   **Conceptual RVC/Advanced TTS Integration Point:** Clear comments indicate where the `espeak-ng` based vocals could be passed to a more advanced model.
        *   `synthesize_verse_with_instrumental()`: Mixes vocals and instrumental.
    *   **Dependencies:** `pydub`, `librosa`, `espeak-ng` (system).
    *   **LLM Context:** Aims for improved `espeak-ng` enunciation by synthesizing whole lines first, then precisely timing segments. Stress modulation adds basic prosody.

### `beefai/utils/`

*   **`utils/__init__.py`**: Makes `utils` a sub-package.
*   **`utils/data_types.py`**:
    *   `FlowDatum`: Includes `syllable_stresses: List[int]`.
    *   `BeatInfo`: Can optionally include `sbf_features_for_timing_ref` (SongBeatFeatures) for refined timing in synthesizer.

### `beefai/webapp/`
// ... (As before) ...

---
## (Other Directories: `data/`, `downloaded_songs/`, `downloaded_files/`, `lite_model_training/`, `output/`)
// ... (Structurally same, content reflects pipeline changes) ...
---

## `scripts/` Directory

*   `05a_tokenize_data_lite.py` / `05b_tokenize_data_full.py`: (As before - tokenizes data).
*   `download_youtube_lyrics.py`: (As before - YouTube/Genius downloader).
*   `download_spotify_playlist.py`: (As before - Spotify/Genius downloader).
*   `inspect_tokenized_data.py`: (As before - utility for inspecting tokenized `.pt` files).
*   `organize_downloaded_songs.py`: (As before - utility for moving downloaded data).
*   `preprocess_dataset.py`: (As before - core script for Phases 2 & 3, extracts `SongBeatFeatures` and `FlowData`).
*   `run_full_data_pipeline.py`: (As before - master script for data prep).
*   `test.py`: (As before - ad-hoc testing).
*   `train_flow_model.py`: (As before - trains the "full" model).

*   **`scripts/generate_rap_battle_response.py`**:
    *   **Purpose:** Main script for generating a full rap battle response. Orchestrates flow generation, lyric generation (API or manual), lyric text review for TTS, and synthesis.
    *   **Key Workflow (V5):**
        1.  **Instrumental Analysis & Flow Generation:** (Steps 1 & 2) Uses `_generate_flow_core`.
        2.  **Lyric Acquisition & TTS Text Prep (Step 3):**
            *   **Lyric Acquisition:**
                *   If API key available for `LyricAgent`, calls `lyric_agent.generate_response_verse_via_api()` for direct, automated lyric generation with syllable count retries.
                *   Else (manual mode), uses `_get_lyrics_manually()` which presents `PromptFormatterV2` output to guide user's interaction with their own LLM.
            *   **Lyric Text Review for TTS:** User reviews the obtained lyric text (from API or manual LLM) and can make minor orthographic tweaks for better `espeak-ng` pronunciation.
            *   Outputs `List[str]` of final lyric lines for TTS.
        3.  **Synthesis (Step 4):**
            *   Calls `RapSynthesizer.synthesize_vocal_track()` with the final list of lyric lines and `FlowData`.
            *   (Conceptual RVC step placeholder).
            *   Calls `RapSynthesizer.synthesize_verse_with_instrumental()` for final mixing.
        4.  Saves outputs. Includes new command-line arguments for stress-based pitch/gain modulation.
    *   **LLM Context:** Provides two paths for lyric generation (direct API vs. guided manual). Emphasizes user review of *text* for TTS, rather than phonetic guides. Synthesis incorporates stress for more dynamic (though still `espeak-ng` based) vocals.