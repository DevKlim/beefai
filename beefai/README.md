# beefai: AI Rap Battle Game

## Project Overview

**beefai** is an ambitious project to create an AI-powered rap battle game. The core goal is to develop a system where an AI can generate and perform rap verses in real-time, responding to a user's rap input, all while staying on beat with a provided instrumental track. This involves modeling rap "flow" (rhythm, timing, and articulation), generating contextually relevant and rhythmically fitting lyrics, and synthesizing these lyrics into an audible rap performance.

This project directly addresses the paradigm of **continuous, conditioned generation**:
*   **Prompt-based generation:** The AI generates rap based on the "prompt" of the user's preceding verse and the musical context.
*   **Continuous control:** The AI's generation is continuously conditioned by the beat and the evolving lyrical context of the battle.

## AI Rap Battle Generation Pipeline

The beefai system generates an AI rap response to a user's input, synchronized with a provided instrumental beat. The pipeline operates as follows:

```mermaid
graph TD
    %% Styling
    classDef input fill:#D6EAF8,stroke:#3498DB,stroke-width:2px;
    classDef process fill:#E8DAEF,stroke:#8E44AD,stroke-width:2px;
    classDef model fill:#D5F5E3,stroke:#2ECC71,stroke-width:2px;
    classDef data fill:#FCF3CF,stroke:#F1C40F,stroke-width:2px;
    classDef output fill:#FDEDEC,stroke:#E74C3C,stroke-width:2px;

    %% Inputs
    A1[/Instrumental Track (MP3/WAV)/];
    A2[/User Rap Voice Input (MP3/WAV)/];
    class A1,A2 input;

    %% Initial Processing
    subgraph "Input Processing & Feature Extraction"
        A1 --> B1[Beat Analysis Engine\n(e.g., librosa, madmom)];
        B1 --> C1[Beat Features Data\n- Tempo, Time Signature\n- Beat Positions (Upbeats, Downbeats)\n- Rhythm Characteristics\n- Musical Key/Scale (optional)];
        class B1 process;
        class C1 data;

        A2 --> B2[User Voice Processing Engine\n(ASR, Feature Extraction)];
        B2 --> C2[User Rap Transcript & Features\n- Transcribed User Lyrics (ASR)\n- Emotion Analysis (optional)\n- Vocal Characteristics (optional)];
        class B2 process;
        class C2 data;
    end

    %% Core AI Generation Pipeline
    subgraph "AI Rap Generation Core"
        C1 --> M1[Flow Generation Model\n(Custom ML Model)];
        M1 --> D1[Generated Flow Data\n- Sequence: (start_time, duration_sec, \n pitch_contour_id, syllables_target)\n- Normalized to beat/bar structure];
        class M1 model;
        class D1 data;
        
        CTX_LLM[Lyric Generation LLM\n(e.g., Fine-tuned Transformer)];
        C2 --> CTX_LLM; %% User Transcript to LLM
        C1 --> CTX_LLM; %% Beat Features to LLM (for context)
        D1 --> CTX_LLM; %% Flow Data to LLM (for structure)
        CTX_LLM --> D2[AI Generated Lyrics\n- Contextually relevant\n- Rhyming\n- Fits syllable/rhythm structure from Flow Data];
        class CTX_LLM model;
        class D2 data;

        D2 --> M3[Speech Synthesis Model\n(TTS/SVS, e.g., Vocaloid-based, DiffSVC)];
        D1 --> M3; %% Flow Data to TTS/SVS (for prosody)
        M3 --> O1[Generated AI Rap Audio\n(Waveform)];
        class M3 model;
        class O1 data;
    end

    %% Outputs
    subgraph "Final Outputs & User Feedback"
        O1 --> F1[/AI Rap Audio File (MP3/WAV)/];
        D1 --> F2[/Live Beat/Syllable Counter\n(Visual Feedback to User)/];
        C1 --> F2; 
        class F1,F2 output;
    end
```

**Pipeline Summary:**

1.  **Input Acquisition & Initial Processing:**
    *   The system takes two primary inputs: an **Instrumental Track** (e.g., MP3/WAV) and the **User's Rap Voice Input** (e.g., MP3/WAV).
    *   The Instrumental Track undergoes **Beat Analysis** (using tools like `librosa` or `madmom`) to extract crucial **Beat Features Data**: tempo, time signature, beat positions (upbeats, downbeats), and overall rhythm characteristics.
    *   The User's Rap Voice Input is processed by a **User Voice Processing Engine** to obtain a **User Rap Transcript** (via Automatic Speech Recognition - ASR) and optionally, features like emotion or vocal characteristics.

2.  **Core AI Generation Cascade:**
    *   **Flow Generation:**
        *   A custom **Flow Generation ML Model** takes the extracted Beat Features Data.
        *   It predicts and generates **Flow Data**: a structured sequence representing the target rhythmic and melodic pattern for the AI's upcoming verse. This includes elements like target syllable counts per line, start times, durations, and pitch contour information, all normalized to the beat and bar structure. This defines *how* the AI should rap.
    *   **Lyric Generation (LLM):**
        *   A **Large Language Model** (e.g., a fine-tuned Transformer model) receives multiple inputs:
            *   The User Rap Transcript (for battle context and response).
            *   The Beat Features Data (for musical context like tempo or mood).
            *   The Generated Flow Data (to constrain lyrics to the correct number of syllables and rhythmic placement).
        *   The LLM then generates **AI Rap Lyrics** that are contextually relevant (e.g., a "diss" track), rhyme, and adhere to the syllable and rhythmic structure defined by the Flow Data.
    *   **Speech Synthesis (TTS/SVS):**
        *   A **Text-to-Speech (TTS)** or **Singing Voice Synthesis (SVS)** model (e.g., Vocaloid-based, DiffSVC, VISinger2) takes:
            *   The AI Generated Lyrics.
            *   The detailed Generated Flow Data (which provides precise timing, duration, and pitch information for each syllable).
        *   It synthesizes these inputs into the final **AI Rap Audio Waveform**.

3.  **Output Delivery:**
    *   The system outputs the **Generated AI Rap Audio** as a downloadable file (e.g., MP3/WAV).
    *   Concurrently, it can drive a **Live Beat/Syllable Counter** to provide visual feedback to the user, synchronized with the music and the AI's vocal delivery, using information from both the Beat Features Data and the Generated Flow Data.

This modular pipeline allows each component to specialize, with the "Flow Data" acting as a critical intermediary, guiding both the lyrical content's structure and the vocal performance's prosody to ensure the AI stays on beat and in rhythm.

*(The current implementation uses placeholder models for Flow Generation, Lyric Generation, and Speech Synthesis. The User Voice Processing Engine is simulated by direct text input.)*

## Modeling Strategy & Future Development

This section outlines the approach for developing and training the machine learning components described in the pipeline.

### 1. Exploratory Analysis, Data Collection, Pre-processing

#### Data Sources & Purpose

To train the components of our AI rap battle system, we will require several types of data:

1.  **Rap Acapellas and Instrumentals:**
    *   **Source:** Explore datasets like FMA (Free Music Archive) for royalty-free music, acapella repositories (e.g., Acapellas4U - with caution regarding licensing for model training), and potentially YouTube for rap battle acapellas and instrumentals (again, with strict attention to copyright and fair use for research). Custom recordings or publicly available rap battle datasets (if any) will also be sought.
    *   **Purpose:**
        *   Acapellas are crucial for training the flow model (to extract rhythmic and melodic patterns of vocals) and the TTS/SVS system (to learn to synthesize rap vocals).
        *   Instrumentals are needed for beat analysis and to provide the musical context for generation.
        *   Aligned acapella-instrumental pairs are ideal for understanding how rap flow interacts with beats.

2.  **Rap Lyrics with Timestamps:**
    *   **Source:** Genius API, Kaggle datasets of rap lyrics. We will need to align these lyrics to audio.
    *   **Purpose:**
        *   Training the lyric generation LLM.
        *   If timestamps (word or syllable-level) can be obtained or generated (e.g., via forced alignment), this data is invaluable for training the flow model to associate lyrical units with rhythmic events.

3.  **Flow/Rhythm Annotations (Derived):**
    *   **Source:** This data will likely be *derived* from acapellas and timestamped lyrics.
    *   **Purpose:** To create a structured representation of rap flow (e.g., sequences of onset times, durations, pitches, phonemes/syllables) that our flow generation model will learn to predict.

4.  **Rap Battle Transcripts/Dialogues:**
    *   **Source:** Online rap battle forums, YouTube battle transcripts, existing dialogue datasets adapted for a battle context.
    *   **Purpose:** To fine-tune the LLM for generating responsorial and confrontational lyrics typical of a rap battle.

#### Data Pre-processing

The collected data will undergo significant pre-processing:

1.  **Audio Processing:**
    *   **Beat Detection & Tempo Estimation:** Using libraries like `librosa` to identify beat locations, downbeats, and overall tempo of instrumental tracks. This forms the rhythmic backbone for the AI.
    *   **Vocal Separation (if needed):** If using full songs, tools like Spleeter or Demucs might be used to isolate vocals.
    *   **Audio Feature Extraction:** Mel-spectrograms, MFCCs, pitch tracking (e.g., CREPE, pYIN) from acapellas to analyze vocal delivery.

2.  **Lyric Processing & Alignment:**
    *   **Text Cleaning:** Normalizing lyrics (lowercase, punctuation removal if necessary).
    *   **Phonetic Transcription:** Converting words/syllables into phonemes (e.g., using `CMUdict`) for more accurate TTS and potentially for flow modeling.
    *   **Forced Alignment:** Using tools like Montreal Forced Aligner (MFA) or Penn Forced Aligner (PFA) to align lyrics (word/phoneme level) with the acapella audio, generating precise timestamps. This is critical for learning flow.
    *   **Syllabification:** Breaking words into syllables (e.g., using `pyphen`) to match rhythmic units to lyrical units.

3.  **Flow Data Representation:**
    *   From aligned audio and lyrics, we will extract a "flow data" sequence. This could be a sequence of tuples: `(start_time, end_time, median_pitch, syllable_text, phonemes)`.
    *   This data will be normalized relative to the beat (e.g., time in beats from the start of a bar, duration in fractions of a beat).

#### Tools & Libraries (Examples for Future Development)
*   Audio: `librosa`, `madmom` (beat tracking), `pydub` (audio manipulation).
*   Vocal Separation: `spleeter`, `demucs`.
*   Lyrics: `requests`, `BeautifulSoup` (scraping), `GeniusLyrics` (API access).
*   Alignment: Montreal Forced Aligner.
*   NLP: `nltk`, `spaCy` (text processing), `pyphen` (syllabification).
*   ASR: `Whisper`, `SpeechRecognition` library.
*   Data Management: `pandas`, `numpy`.

### 2. Modeling Approaches

Our system will consist of three main interconnected ML models:

1.  **Flow Generation Model:**
    *   **Input:** Beat information (tempo, beat positions), potentially the previous N bars of flow, or high-level conditioning signals.
    *   **Output:** A sequence representing the target rhythmic and melodic structure (FlowData: syllable timings, durations, pitch contours).
    *   **Approach:** RNNs (LSTMs/GRUs), Transformers, or Conditional VAEs/GANs. Start with an LSTM/Transformer sequence-to-sequence model.

2.  **Lyric Generation Model (LLM):**
    *   **Input:** Current beat/musical context, FlowData constraints, opponent's last rap verse, prompt.
    *   **Output:** Rap lyrics fitting constraints, rhyming, and contextually relevant.
    *   **Approach:** Fine-tuning a pre-trained LLM (e.g., GPT-2, Llama, Mistral). Focus on techniques for constrained text generation.

3.  **Text-to-Speech (TTS) / Singing Voice Synthesis (SVS) Model:**
    *   **Input:** Lyrics from LLM, detailed prosody from FlowData, target voice.
    *   **Output:** Synthesized rap audio waveform.
    *   **Approach:** Explore fine-tuning a pre-trained SVS model (e.g., VISinger2, DiffSVC) or a robust TTS (e.g., FastSpeech 2) conditioned on detailed F0 and duration.

*   **Overall Architecture:** A modular approach (Flow Model -> LLM -> TTS/SVS) is preferred for easier development and debugging.
*   **Challenges:** Ensuring seamless integration and achieving real-time performance.

## Current Project Structure & Implemented Components

The project is organized into several Python packages:

*   **`beefai/`**
    *   **`data_processing/`**: Handles audio and text data.
        *   `audio_processor.py`: Loads audio, extracts beat information (BPM, beat times, downbeats) using `librosa`. (Placeholder for "Beat Analysis Engine")
        *   `text_processor.py`: Counts syllables using `pyphen`; placeholder for advanced NLP like phonetic analysis.
    *   **`flow_model/`**:
        *   `model.py`: Placeholder for the Flow Generation Model. Currently generates a plausible rhythmic structure based on beat information.
    *   **`lyric_generation/`**:
        *   `agent.py`: Placeholder for the Lyric Generation LLM. Simulates lyric creation to fit flow data syllable counts.
    *   **`synthesis/`**:
        *   `synthesizer.py`: Placeholder for the TTS/SVS Model. Generates basic audio tones or uses (commented out) TTS, attempting to match timing from flow data.
    *   **`utils/`**:
        *   `data_types.py`: Defines common data structures like `BeatInfo` and `FlowData`.
    *   **`webapp/`**: (Conceptual - For a fully interactive version)
        *   `index.html`, `static/css/style.css`, `static/js/main.js`: Basic frontend for interaction. The current `main.js` simulates the backend pipeline.
    *   `main.py`: Main application script to run a simulated rap battle, demonstrating the pipeline with placeholder components.
*   `requirements.txt`: Lists Python dependencies.
*   `setup.sh`: Shell script for setting up the development environment.
*   `README.md`: This file.

## Setup and Usage

1.  **Prerequisites:**
    *   Python 3.11+
    *   (For Linux/macOS) `bash` for the setup script.
    *   Ensure `ffmpeg` is installed and accessible in your PATH if you plan to work with various audio formats extensively with `librosa`.

2.  **Setup:**
    *   Clone the repository.
    *   Run the setup script:
        ```bash
        # For Linux/macOS
        bash setup.sh
        # For Windows (using Git Bash or similar, or manually create venv and install reqs)
        # bash setup.sh 
        # (If setup.sh doesn't work on Windows, manually:
        #  python -m venv .venv
        #  .venv\Scripts\activate
        #  pip install -r requirements.txt)
        ```
    *   This script will:
        *   Check your Python version.
        *   Create a virtual environment (`.venv`).
        *   Activate the virtual environment.
        *   Install dependencies from `requirements.txt`.

3.  **Running the Demo:**
    *   Ensure the virtual environment is activated:
        ```bash
        # Linux/macOS
        source .venv/bin/activate
        # Windows
        # .venv\Scripts\activate
        ```
    *   Run the main simulation script:
        ```bash
        python beefai/main.py
        ```
    *   This will run a simulated rap battle. Generated audio files (placeholder quality) will be saved in the `output/` directory.

4.  **Running the Web Interface (Simulated Backend):**
    *   Open `beefai/webapp/index.html` in your web browser.
    *   This interface uses JavaScript to simulate the recording and AI response process. It does not currently make live calls to the Python backend but demonstrates the intended user interaction flow.
    *   For the "Live Beat/Syllable Counter" visual feedback, a placeholder element might exist, but full functionality requires backend integration.

This README now reflects the target pipeline and clarifies the role of current components as placeholders within that larger vision.
```

Next, I'll adjust the Python files.

**1. `beefai/utils/data_types.py`**
The `FlowDatum` example needs to align with what `FlowModel` produces and what `LyricAgent` and `RapSynthesizer` expect. The current output from `FlowModel` is a dictionary including `start_time_sec`, `duration_sec`, `syllables`, `pitch_contour_id`, `bar_index`, etc. This is more useful than just `start_beat`, `duration_beats`.

```python
# file_path: beefai/utils/data_types.py
from typing import List, Dict, Any, Optional, Tuple

# Represents the rhythmic and structural information for a segment of rap
# This data guides the lyric generation and synthesis.
FlowDatum = Dict[str, Any]
# Example:
# {
#     "bar_index": int,               # 1-based index of the bar
#     "line_index_in_bar": int,       # 1-based index of the line within the bar
#     "start_time_sec": float,        # Start time of the segment in seconds from audio start
#     "duration_sec": float,          # Duration of the segment in seconds
#     "syllables": int,               # Target number of syllables for this segment
#     "pitch_contour_id": Optional[str],# Identifier for a predefined pitch contour (e.g., "rising", "flat_mid")
#     "start_beat_global": Optional[float], # Start beat index globally (optional, for reference)
#     "duration_beats": Optional[float]   # Duration in beats (optional, for reference)
# }


FlowData = List[FlowDatum]

# Represents beat information extracted from an audio track
BeatInfo = Dict[str, Any]
# Example:
# {
#     "bpm": float,
#     "beat_times": List[float],       # List of timestamps for each detected beat
#     "downbeat_times": List[float],   # List of timestamps for each detected downbeat
#     "beats_per_bar": int,            # Estimated beats per bar (e.g., 4)
#     "estimated_bar_duration": float  # Estimated duration of a bar in seconds
# }


# Represents lyrical content, potentially with timing if aligned
# This is more of a target for advanced systems. Currently, lyrics are List[str].
LyricsData = List[Dict[str, Any]] # Example: [{"word": str, "start_time": float, "end_time": float}]

# Represents raw audio data
AudioData = Tuple[Any, int] # (waveform_array: np.ndarray, sample_rate: int)

Okay, let's outline a key implementation plan for the Decoder-only Transformer for Flow Generation, focusing on the specified data engineering pipeline.

## Thinking Phase:

1.  **Deconstruct the Request:**
    *   **Model:** Decoder-only Transformer.
    *   **Task:** Generate rap flow (`FlowData` sequence).
    *   **Input Conditioning (Beat Features):**
        *   Tempo.
        *   Time Signature.
        *   Bass/drum events: List of numbers (1-16) indicating active subdivisions in a measure. This needs to be specified per instrument (bass, kick, snare, hi-hat). So, for each bar, we'll have 4 lists of active subdivisions or 4 binary vectors of length 16.
        *   "Events (option b) of significant sources of sound that indicate a pattern": This implies a more symbolic representation of percussive events rather than just a raw grid. For example, `(instrument, subdivision_index)`.
    *   **Output `FlowData` (Target):**
        *   Sequence of `FlowDatum`: `(bar_index, line_index_in_bar, syllables, start_offset_beats, duration_beats)`.
        *   *Explicitly ignore pitch contours for now.*
    *   **Core Challenge:** Representing the diverse input (continuous tempo, categorical time sig, sparse event lists/grids for percussion) and output (integers for syllables, quantized continuous for timing) in a way a Transformer can consume and produce.

2.  **Data Pipeline - Key Stages:**
    *   **Acquisition:** Acapellas, Instrumentals (ideally stems or good quality for separation), Lyrics.
    *   **Beat Feature Extraction (Model Input):**
        *   Process instrumentals.
        *   Get Tempo, Time Signature.
        *   Percussion Analysis: This is key. Need to get events for bass, kick, snare, hi-hat.
            *   Source separation is almost mandatory here (e.g., Demucs, Spleeter).
            *   For each separated stem (bass, drums):
                *   Onset detection.
                *   For drums, further classification (kick, snare, hi-hat) if possible. This can be tricky; heuristics or simple classifiers might be needed.
                *   Map onsets to 16 subdivisions of a bar.
    *   **Flow Data Extraction (Model Target):**
        *   Process acapellas and lyrics.
        *   Forced alignment (crucial).
        *   Syllabification.
        *   Segmentation into lines/phrases.
        *   Quantization of timing information (start offset, duration) into discrete bins.
    *   **Tokenization & Dataset Assembly:**
        *   Define a vocabulary that covers all input features and output flow elements.
        *   Structure into `(input_sequence, target_sequence)` pairs for the Transformer.

3.  **Decoder-Only Transformer - Key Aspects:**
    *   **Input Format:** A single sequence combining beat features and (for training) target flow elements.
    *   **Output Format:** Autoregressive generation of flow element tokens.
    *   **Architecture:** Standard Transformer decoder blocks.
    *   **Masking:** Causal attention mask.

4.  **Implementation Plan Structure:**
    *   Phase 0: Setup & Prerequisites
    *   Phase 1: Data Collection & Initial Processing
    *   Phase 2: Beat Feature Engineering (Model Input)
    *   Phase 3: Flow Data Engineering (Model Target)
    *   Phase 4: Tokenization & Dataset Assembly
    *   Phase 5: Decoder-Only Transformer Model Implementation
    *   Phase 6: Training & Initial Evaluation
    *   Phase 7: Iteration and Refinement

## Key Implementation Plan: Flow Generation (Decoder-Only Transformer)

This plan focuses on getting a first version of the flow generation model working, ignoring pitch contours and using the specified beat features.

---

**Phase 0: Setup & Prerequisites**

1.  **Environment:**
    *   Confirm Python environment with necessary base libraries (`numpy`, `pandas`).
    *   Install core ML/audio libraries: `torch` (or `tensorflow`), `librosa`, `pydub`, `pyphen`.
    *   Install/setup source separation tools (e.g., `demucs`).
    *   Install/setup a forced aligner (e.g., Montreal Forced Aligner - MFA).
2.  **Version Control:** Initialize Git repository.
3.  **Initial Data Staging Area:** Create directories for raw audio, processed data, etc.

---

**Phase 1: Data Collection & Initial Processing (Iterative)**

1.  **Gather Raw Data:**
    *   **Source:** Acapellas (cleanest possible), corresponding instrumentals, and lyrics.
    *   **Target:** Aim for at least 50-100 songs initially, more is better. Prioritize songs with clear rhythmic structures.
    *   **Organization:** Store systematically (e.g., `data/raw/song_id/acapella.wav`, `data/raw/song_id/instrumental.wav`, `data/raw/song_id/lyrics.txt`).
2.  **Basic Audio Normalization:**
    *   Convert all audio to a consistent format (e.g., WAV, 44.1kHz, mono for acapellas, stereo/mono for instrumentals).
    *   Normalize volume levels if highly variable.
    *   **Tools:** `librosa`, `pydub`, `ffmpeg`.

---

**Phase 2: Beat Feature Engineering (Model Input)**

*Goal: For each instrumental, extract per-bar features: Tempo, Time Signature, and percussive event locations for bass, kick, snare, hi-hat.*

1.  **Audio Pre-processing for Instrumentals:**
    *   Load instrumental audio.
2.  **Global Feature Extraction:**
    *   **Tempo (BPM):** `librosa.beat.tempo`. Store one value per song (or segment if tempo changes).
    *   **Time Signature:** `librosa.beat.beat_track` (can infer from beat groupings). Assume 4/4 if detection is unreliable initially.
3.  **Bar Segmentation:**
    *   Detect beats and downbeats: `librosa.beat.beat_track`.
    *   Group beats into bars based on time signature. Store bar start/end times.
4.  **Percussive Source Separation (per song):**
    *   Apply source separation (e.g., `demucs`) to instrumentals to get:
        *   `bass` stem
        *   `drums` stem
    *   If `demucs` provides `kick`, `snare`, `hihat` directly, use those. Otherwise, the `drums` stem will need further processing.
5.  **Percussive Event Detection & Grid Creation (per bar, per instrument):**
    *   For each bar and for each target instrument stem (`bass`, and from `drums`: `kick`, `snare`, `hi-hat`):
        *   **Onset Detection:** `librosa.onset.onset_detect` on the instrument stem within the bar's time window.
        *   **(If processing generic `drums` stem for K/S/HH):**
            *   *Heuristic Classification (Initial):* Apply simple frequency-based rules or a pre-trained basic drum sound classifier to categorize drum onsets into kick, snare, hi-hat. This is a challenging step; aim for "good enough" initially.
            *   *Alternative:* Use a tool specifically for drum transcription if available and feasible.
        *   **Quantization to 16 Subdivisions:**
            *   For each detected (and classified, if needed) onset time:
                *   Calculate its position within the bar (0.0 to 1.0).
                *   Map this to one of 16 subdivisions (e.g., `subdivision_index = floor(position_in_bar * 16)`).
            *   **Output per bar:**
                *   `kick_events`: List of active subdivision indices (0-15) for kicks.
                *   `snare_events`: List of active subdivision indices for snares.
                *   `hihat_events`: List of active subdivision indices for hi-hats.
                *   `bass_events`: List of active subdivision indices for bass.
6.  **Data Storage:**
    *   Store these extracted features in a structured format (e.g., JSON files, one per song, mapping bar index to its features).
    *   Example per bar:
        ```json
        {
          "bar_index": 0,
          "bpm": 120.0,
          "time_signature": [4, 4],
          "kick_events": [0, 4, 8, 12], // kick on 1st, 5th, 9th, 13th 16th-note
          "snare_events": [4, 12],      // snare on 5th, 13th 16th-note
          "hihat_events": [0, 2, 4, 6, 8, 10, 12, 14], // 8th note hi-hats
          "bass_events": [0, 8]
        }
        ```

---

**Phase 3: Flow Data Engineering (Model Target)**

*Goal: For each acapella, extract a sequence of `FlowDatum` (line/phrase level) containing syllables, start offset, and duration, all normalized to beats.*

1.  **Acapella Pre-processing:**
    *   Load acapella audio.
2.  **Forced Alignment:**
    *   Use MFA (or similar) with acapella audio and corresponding lyrics to get word-level (and ideally phoneme-level) timestamps. *This is critical for accuracy.*
    *   **Output:** Structured alignment data (e.g., Praat TextGrids, JSON).
3.  **Syllabification & Timestamp Propagation:**
    *   For each word from alignment:
        *   Break into syllables using `pyphen`.
        *   Distribute word duration among its syllables (e.g., equally, or more sophisticated heuristics if phoneme timestamps are good). Get start/end time for each syllable.
4.  **Segmentation into Lines/`FlowDatum`:**
    *   **Define "Line":** Decide on a heuristic. E.g.:
        *   Fixed number of bars (e.g., every 2 bars forms a "line" segment for flow).
        *   Silence-based: Group syllables between significant pauses in the acapella.
    *   Iterate through syllables, grouping them into these defined lines/segments.
5.  **`FlowDatum` Creation (per line/segment):**
    *   For each segment:
        *   `bar_index`, `line_index_in_bar`: Determined by the segmentation logic and downbeat times from the *instrumental's beat analysis*.
        *   `syllables`: Count of syllables in this segment.
        *   `segment_start_time_sec`: Start time of the first syllable in the segment.
        *   `segment_end_time_sec`: End time of the last syllable in the segment.
        *   **Beat Normalization (using BPM from Beat Features):**
            *   `beat_duration_sec = 60.0 / bpm`
            *   `bar_start_time_sec`: Get the start time of the bar this segment falls into (from instrumental beat analysis).
            *   `start_offset_beats = (segment_start_time_sec - bar_start_time_sec) / beat_duration_sec`
            *   `duration_beats = (segment_end_time_sec - segment_start_time_sec) / beat_duration_sec`
6.  **Quantization for Transformer:**
    *   **Syllables:** Treat as an integer. Consider if capping at a max (e.g., 24) is needed for vocabulary size.
    *   **`start_offset_beats` & `duration_beats`:** Quantize into discrete bins.
        *   Example: Resolution of 0.25 beats. `start_offset_beats` could range 0 to `beats_per_bar - 0.25`. `duration_beats` could range 0.25 up to, say, 8 beats (if a line can span 2 bars).
        *   Define bin edges and map continuous values to bin indices.
7.  **Data Storage:**
    *   Store as a sequence of `FlowDatum` objects (or dictionaries) per song, linked to the beat features.
    *   Example `FlowDatum` (after quantization, conceptual):
        ```json
        {
          "bar_index": 0, // The bar this line primarily starts in or belongs to
          "line_index_in_bar": 0, // If multiple lines per bar segment
          "syllables_val": 12,
          "start_offset_beats_bin": 0, // e.g., bin for 0.0 beats from bar start
          "duration_beats_bin": 7     // e.g., bin for 2.0 beats duration
        }
        ```

---

**Phase 4: Tokenization & Dataset Assembly**

1.  **Define Vocabulary:**
    *   **Special Tokens:** `[PAD]`, `[BOS]` (Begin Of Sequence), `[EOS]` (End Of Sequence), `[SEP_INPUT_TARGET]` (separates beat input from flow target in the combined sequence), `[BAR_START_MARKER]`, `[LINE_START_MARKER]`.
    *   **Beat Feature Tokens:**
        *   `[BPM_XXX]` (e.g., `[BPM_120]`, `[BPM_121]`, or quantized BPM ranges like `[BPM_118_122]`).
        *   `[TIMESIG_4_4]`, `[TIMESIG_3_4]`.
        *   For each instrument (`kick`, `snare`, `hihat`, `bass`):
            *   A token for each active subdivision: `[KICK_SUBDIV_0]`, ..., `[KICK_SUBDIV_15]`.
            *   Alternatively, to keep sequences shorter: A single token per instrument per bar representing its pattern (e.g., `[KICK_PATTERN_ID_XYZ]`). This requires pre-clustering patterns or handling a very large vocabulary if patterns are diverse. *Let's start with explicit subdivision tokens for presence/absence, e.g., `[KICK_ON_0]`, `[KICK_OFF_1]`, ... or a sequence of active events `[KICK_AT_0] [KICK_AT_4] [KICK_AT_8] [KICK_AT_12] [END_KICK_EVENTS]`.*
            *   **Decision for Plan:** Use event list tokens: `[KICK_AT_0]...[END_KICK_EVENTS]`, `[SNARE_AT_4]...[END_SNARE_EVENTS]`, etc. This is flexible.
    *   **Flow Target Tokens:**
        *   `[SYLLABLES_1]`, `[SYLLABLES_2]`, ..., `[SYLLABLES_MAX]`.
        *   `[START_OFFSET_BIN_0]`, ..., `[START_OFFSET_BIN_MAX]`.
        *   `[DURATION_BIN_0]`, ..., `[DURATION_BIN_MAX]`.
2.  **Create Tokenizer:** Map tokens to integer IDs and vice-versa.
3.  **Construct Training Instances:**
    *   For each song, create a sequence of bars. Each bar will have its beat features followed by its flow data (one or more lines).
    *   **Input Sequence Format for Transformer (autoregressive training):**
        `[BOS]`
        `[BAR_START_MARKER]`
        `[BPM_XXX]` `[TIMESIG_X_X]`
        `[KICK_AT_S1] [KICK_AT_S2] ... [END_KICK_EVENTS]`
        `[SNARE_AT_S1] ... [END_SNARE_EVENTS]`
        `[HIHAT_AT_S1] ... [END_HIHAT_EVENTS]`
        `[BASS_AT_S1] ... [END_BASS_EVENTS]`
        `[SEP_INPUT_TARGET]`
        `[LINE_START_MARKER]` `[SYLLABLES_S]` `[START_OFFSET_BIN_SO]` `[DURATION_BIN_D]` (for line 1 in bar)
        `[LINE_START_MARKER]` `[SYLLABLES_S']` `[START_OFFSET_BIN_SO']` `[DURATION_BIN_D']` (for line 2 in bar, if exists)
        `[BAR_START_MARKER]` (for next bar)
        ...
        `[EOS]`
    *   The model will be trained to predict the next token in this combined sequence. During inference, it predicts tokens after `[SEP_INPUT_TARGET]`.
4.  **Split Data:** Train, validation, test sets.

---

**Phase 5: Decoder-Only Transformer Model Implementation**

1.  **Choose Framework:** PyTorch (Hugging Face Transformers library is excellent here) or TensorFlow/Keras.
2.  **Model Architecture:**
    *   **Embedding Layer:** For the defined vocabulary.
    *   **Positional Encoding:** Sinusoidal or learned.
    *   **Transformer Decoder Blocks:** Stack of N blocks (each with multi-head self-attention, layer norm, feed-forward network, layer norm).
    *   **Output Layer:** Linear layer projecting to vocabulary size, followed by Softmax.
3.  **Masking:**
    *   Implement causal (look-ahead) mask for self-attention.
    *   Implement padding mask if using batching with variable length sequences.
4.  **Configuration:** Hyperparameters (num_layers, num_heads, d_model, d_ff, dropout).

---

**Phase 6: Training & Initial Evaluation**

1.  **Loss Function:** Cross-Entropy Loss.
2.  **Optimizer:** AdamW.
3.  **Learning Rate Scheduler:** E.g., linear warmup and decay.
4.  **Training Loop:**
    *   Feed input sequences to the model.
    *   Calculate loss between predicted tokens and actual next tokens.
    *   Backpropagate and update weights.
5.  **Monitoring:** Track training/validation loss, perplexity, accuracy.
6.  **Generation/Inference:**
    *   Provide beat features as a prompt:
        `[BOS] [BAR_START_MARKER] [BPM_XXX] ... [END_BASS_EVENTS] [SEP_INPUT_TARGET]`
    *   Autoregressively generate flow tokens until `[EOS]` or max length.
    *   Decode generated token IDs back to `FlowDatum` representation.
7.  **Initial Evaluation:**
    *   **Objective:** Perplexity on validation set.
    *   **Subjective (Crucial):**
        *   Take a few input beat patterns.
        *   Generate multiple flow sequences.
        *   Manually inspect: Do syllable counts make sense? Do timings seem plausible for rap? Are they varied?
        *   Use a placeholder TTS (even basic tones with correct timing) to "hear" the rhythm.

---

**Phase 7: Iteration and Refinement**

*Based on evaluation from Phase 6:*

1.  **Data Issues:**
    *   Revisit forced alignment quality.
    *   Improve percussive event detection/classification.
    *   Adjust quantization bins for timing if too coarse/fine.
    *   Augment dataset if sparse in certain rhythmic styles.
2.  **Tokenization:**
    *   Is the vocabulary too large/small?
    *   Are event list tokens (`[KICK_AT_X]`) working well, or is a grid representation better?
3.  **Model Hyperparameters:** Tune learning rate, model size, dropout, etc.
4.  **Error Analysis:** Where does the model fail most? (e.g., wrong syllable counts, awkward timings).
5.  **Consider Adding Pitch:** Once the rhythmic generation is somewhat stable, plan for incorporating pitch contour prediction as another token type in the `FlowDatum`.

---

This detailed plan provides a roadmap. Each step, especially in data engineering, can be complex and iterative. Good luck!