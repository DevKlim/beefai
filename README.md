# beefai: AI Rap Battle Game

## Project Overview

**beefai** is an ambitious project to create an AI-powered rap battle game. The core goal is to develop a system where an AI can generate and perform rap verses in real-time, responding to a user's rap input, all while staying on beat with a provided instrumental track. This involves modeling rap "flow" (rhythm, timing, stress, and articulation), generating contextually relevant and rhythmically fitting lyrics, and synthesizing these lyrics into an audible rap performance.

This project directly addresses the paradigm of **continuous, conditioned generation**:
*   **Prompt-based generation:** The AI generates rap based on the "prompt" of the user's preceding verse and the musical context.
*   **Continuous control:** The AI's generation is continuously conditioned by the beat, musical features (like stress and energy), and the evolving lyrical context of the battle.

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
        A1 --> B1[Beat Analysis Engine\n(beefai.data_processing.BeatFeatureExtractor)];
        B1 --> C1[Beat Features Data\n- Tempo, Time Signature\n- Beat/Bar Timings\n- Quantized Percussive Events (Kick, Snare, Hihat, Bass)];
        class B1 process;
        class C1 data;

        A2 --> B2[User Voice Processing Engine\n(ASR, beefai.data_processing.FlowDataExtractor)];
        B2 --> C2[User Rap Transcript & Flow Features\n- Transcribed User Lyrics (from ASR alignment)\n- Syllable Counts, Timings, Stress (FlowData)];
        class B2 process;
        class C2 data;
    end

    %% Core AI Generation Pipeline
    subgraph "AI Rap Generation Core"
        C1 --> M1[Flow Generation Model\n(beefai.flow_model.FlowTransformerDecoder)];
        M1 --> D1[Generated AI FlowData\n- Sequence: (bar_idx, line_idx, syllables, \n offset_beats, duration_beats, \n syllable_starts, syllable_durations, syllable_stresses)\n- Normalized to beat/bar structure];
        class M1 model;
        class D1 data;
        
        CTX_LLM[Lyric Generation LLM\n(e.g., Fine-tuned Transformer, using beefai.lyric_generation.LyricAgent)];
        C2 --> CTX_LLM; %% User Transcript (or summary) to LLM
        C1 --> CTX_LLM; %% Beat Features to LLM (for context/energy)
        D1 --> CTX_LLM; %% AI Flow Data to LLM (for lyrical structure)
        CTX_LLM --> D2[AI Generated Lyrics\n- Contextually relevant\n- Rhyming\n- Fits syllable/rhythm/stress structure from AI Flow Data];
        class CTX_LLM model;
        class D2 data;

        D2 --> M3[Speech Synthesis Model\n(TTS/SVS, e.g., beefai.synthesis.RapSynthesizer)];
        D1 --> M3; %% AI Flow Data to TTS/SVS (for prosody and timing)
        M3 --> O1[Generated AI Rap Audio\n(Waveform)];
        class M3 model;
        class O1 data;
    end

    %% Outputs
    subgraph "Final Outputs & User Feedback"
        O1 --> F1[/AI Rap Audio File (MP3/WAV)/];
        D1 --> F2[/Live Beat/Syllable Counter & Visualizer\n(Visual Feedback to User, e.g., beefai.evaluation.RhythmVisualizer)/];
        C1 --> F2; 
        class F1,F2 output;
    end