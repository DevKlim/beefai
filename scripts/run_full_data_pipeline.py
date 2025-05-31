import subprocess
import os
import sys
import argparse

# Ensure the beefai module can be found if scripts are run from project root
sys.path.append(os.getcwd())

# Default paths (can be overridden by command-line arguments)
# These should ideally align with defaults in preprocess_dataset.py and 05a/b_tokenize_data.py
DEFAULT_RAW_SONGS_DIR = "data/raw_songs_full/"
DEFAULT_LYRICS_DIR = "data/lyrics/"
DEFAULT_INSTRUMENTALS_DIR = "data/instrumentals/"
DEFAULT_ACAPELLAS_DIR = "data/acapellas/"
DEFAULT_ALIGNMENTS_JSON_DIR = "data/alignments_json/"
DEFAULT_PREPROCESSED_OUTPUT_DIR = "data/processed_for_transformer/" # For preprocess_dataset.py output
DEFAULT_TOKENIZED_LITE_DIR = "data/tokenized_lite/" # For 05a_tokenize_data_lite.py output
DEFAULT_TOKENIZED_FULL_DIR = "data/tokenized_full/" # For 05b_tokenize_data_full.py output

# Stem separation related defaults
DEFAULT_STEM_OUTPUT_BASE_DIR = "data/temp_demucs_separated" # Example, Demucs might put it here
DEFAULT_DEMUCS_MODEL = "htdemucs_ft" # Common high-quality Demucs model


def run_command(command_list, step_name):
    """Helper function to run an external command and handle errors."""
    print(f"\n--- Running Step: {step_name} ---")
    print(f"Executing: {' '.join(command_list)}")
    try:
        # Capture output for better logging
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if stdout: print(f"Stdout from {step_name}:\n{stdout}")
        if stderr: print(f"Stderr from {step_name}:\n{stderr}")

        if process.returncode != 0:
            print(f"ERROR: {step_name} failed with exit code {process.returncode}.")
            print(f"--- {step_name} failed. Halting pipeline. ---")
            return False
        print(f"--- {step_name} completed successfully. ---")
        return True
    except FileNotFoundError:
        print(f"ERROR: Command '{command_list[0]}' not found for step '{step_name}'. Ensure it's installed and in PATH.")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during '{step_name}': {e}")
        return False


def run_python_script(script_path_relative_to_scripts_dir, script_name, script_args=None):
    """Helper function to run a Python script from the 'scripts' directory."""
    full_script_path = os.path.join("scripts", script_path_relative_to_scripts_dir)
    if not os.path.exists(full_script_path):
        print(f"ERROR: Script {full_script_path} not found. Skipping.")
        return False
    
    command = [sys.executable, full_script_path]
    if script_args:
        command.extend(script_args)
    
    return run_command(command, script_name)


def main(args):
    print("====== Starting Full Data Preparation Pipeline ======")

    # --- Phase 0: Setup ---
    print("\nPHASE 0: Setup")
    print("Please ensure you have run 'setup.sh' (Linux/macOS) or 'setup.bat' (Windows)")
    print("to create the virtual environment and install dependencies (PyTorch, whisper-timestamped, etc.).")
    print("Also ensure FFmpeg is installed and in your system's PATH.")
    if not args.skip_setup_prompt:
        input("Press Enter to continue if setup is complete...")

    # --- Phase 1: Data Acquisition & Initial Organization ---
    if not args.skip_data_acquisition:
        print("\nPHASE 1: Data Acquisition & Organization")
        print("This phase includes downloading songs/lyrics and basic organization.")
        print(f"Ensure '{args.raw_songs_dir}' and '{args.lyrics_dir}' are populated, or run acquisition scripts.")
        # run_python_script("download_youtube_lyrics.py", "Download YouTube Lyrics") # Optional
        # run_python_script("organize_downloaded_songs.py", "Organize Downloaded Songs") # Optional
        # run_python_script("remove_first_line_from_lyrics.py", "Clean Lyric Files") # Optional
    else:
        print("\nSkipping PHASE 1: Data Acquisition & Organization")


    # --- Phase 1b: Source Separation (Example using Demucs) ---
    if not args.skip_source_separation:
        print("\nPHASE 1b: Source Separation (Example: Demucs)")
        print("This step separates raw songs into instrumentals and acapellas.")
        print(f"Raw songs are expected in: {args.raw_songs_dir}")
        print(f"Separated stems (and subsequently instrumentals/acapellas) will be processed based on Demucs output.")
        print(f"Instrumentals will be moved to: {args.instrumentals_dir}")
        print(f"Acapellas will be moved to: {args.acapellas_dir}")
        
        os.makedirs(args.instrumentals_dir, exist_ok=True)
        os.makedirs(args.acapellas_dir, exist_ok=True)
        
        # Determine the Demucs output directory structure
        # Demucs typically outputs to: base_output_dir / demucs_model_name / song_name_no_ext / {vocals.wav, no_vocals.wav}
        demucs_output_for_model = os.path.join(args.stem_output_base_dir, args.demucs_model)
        os.makedirs(demucs_output_for_model, exist_ok=True)

        raw_song_files = [f for f in os.listdir(args.raw_songs_dir) if os.path.isfile(os.path.join(args.raw_songs_dir, f)) and f.lower().endswith(('.mp3', '.wav', '.flac'))]
        if not raw_song_files:
            print(f"No raw songs found in {args.raw_songs_dir} to separate. Skipping Demucs.")
        else:
            for song_file in raw_song_files:
                song_path = os.path.join(args.raw_songs_dir, song_file)
                song_name_no_ext = os.path.splitext(song_file)[0]
                
                # Define expected output paths from Demucs
                expected_demucs_song_dir = os.path.join(demucs_output_for_model, song_name_no_ext)
                expected_vocals_path = os.path.join(expected_demucs_song_dir, "vocals.wav")
                expected_no_vocals_path = os.path.join(expected_demucs_song_dir, "no_vocals.wav")

                # Define final destination paths
                final_acapella_path = os.path.join(args.acapellas_dir, f"{song_name_no_ext}.wav") # Assuming wav for consistency
                final_instrumental_path = os.path.join(args.instrumentals_dir, f"{song_name_no_ext}.wav")

                if os.path.exists(final_acapella_path) and os.path.exists(final_instrumental_path) and not args.force_rerun_separation:
                    print(f"Stems for {song_file} already exist at final destinations. Skipping separation.")
                    continue

                # Run Demucs for this song
                # Output directly to the model-specific subdirectory
                demucs_cmd = [
                    sys.executable, "-m", "demucs",
                    "--two-stems=vocals", # Get vocals and accompaniment
                    "-o", args.stem_output_base_dir, # Demucs will create args.demucs_model subdir here
                    "-n", args.demucs_model,
                    song_path
                ]
                if not run_command(demucs_cmd, f"Demucs Separation for {song_file}"):
                    print(f"Demucs failed for {song_file}. Pipeline cannot continue robustly for this song.")
                    continue # Skip to next song or implement stricter error handling

                # Move/rename Demucs outputs
                if os.path.exists(expected_vocals_path) and os.path.exists(expected_no_vocals_path):
                    print(f"Moving separated stems for {song_name_no_ext}...")
                    try:
                        os.rename(expected_vocals_path, final_acapella_path)
                        os.rename(expected_no_vocals_path, final_instrumental_path)
                        print(f"  Moved vocals to: {final_acapella_path}")
                        print(f"  Moved instrumental to: {final_instrumental_path}")
                        # Optionally remove the (now empty) expected_demucs_song_dir if desired
                        # And potentially the demucs_output_for_model if it's the last song, though safer to leave
                    except Exception as e:
                        print(f"  Error moving stems for {song_name_no_ext}: {e}")
                else:
                    print(f"  ERROR: Demucs output not found at expected paths for {song_name_no_ext}:")
                    print(f"    Expected vocals: {expected_vocals_path}")
                    print(f"    Expected no_vocals: {expected_no_vocals_path}")
                    print(f"  Please check Demucs execution and output structure.")
    else:
        print("\nSkipping PHASE 1b: Source Separation")
        print(f"Ensure '{args.instrumentals_dir}' and '{args.acapellas_dir}' are populated.")


    # --- Phase 1c: Forced Alignment (whisper-timestamped) ---
    if not args.skip_forced_alignment:
        print("\nPHASE 1c: Forced Alignment (whisper-timestamped)")
        print(f"Aligning acapellas from: {args.acapellas_dir}")
        print(f"JSON alignments will be saved to: {args.alignments_json_dir}")
        os.makedirs(args.alignments_json_dir, exist_ok=True)

        acapella_files = [f for f in os.listdir(args.acapellas_dir) if os.path.isfile(os.path.join(args.acapellas_dir, f))]
        if not acapella_files:
            print(f"No acapellas found in {args.acapellas_dir} to align. Skipping alignment.")
        else:
            for acapella_file in acapella_files:
                acapella_path = os.path.join(args.acapellas_dir, acapella_file)
                song_name_no_ext = os.path.splitext(acapella_file)[0]
                output_json_path = os.path.join(args.alignments_json_dir, f"{song_name_no_ext}.json")

                if os.path.exists(output_json_path) and not args.force_rerun_alignment:
                    print(f"Alignment for {acapella_file} already exists. Skipping.")
                    continue
                
                # Basic check if whisper_timestamped is available
                try:
                    subprocess.run([sys.executable, "-m", "whisper_timestamped", "--help"], capture_output=True, check=True, text=True)
                except (subprocess.CalledProcessError, FileNotFoundError) :
                    print("ERROR: whisper_timestamped command not found or not runnable. Please ensure it's installed correctly (pip install whisper-timestamped).")
                    print("Halting pipeline.")
                    return

                align_cmd = [
                    sys.executable, "-m", "whisper_timestamped",
                    acapella_path,
                    "--model", "small", # Or "base", "medium", "large" - adjust as needed
                    "--output_dir", args.alignments_json_dir,
                    "--output_format", "json" # Ensures .json output
                    # Add other whisper_timestamped flags as desired, e.g., --language en
                ]
                if not run_command(align_cmd, f"Forced Alignment for {acapella_file}"):
                    print(f"Alignment failed for {acapella_file}. This may impact downstream processing for this song.")
                    # Decide if to halt or continue with other songs
    else:
        print("\nSkipping PHASE 1c: Forced Alignment")
        print(f"Ensure '{args.alignments_json_dir}' is populated with alignment JSONs.")


    # --- Phase 2 & 3: Feature Extraction & Flow Data Engineering ---
    print("\nPHASE 2 & 3: Beat Feature and Flow Data Extraction")
    # This runs scripts/preprocess_dataset.py
    preprocess_args = [
        "--instrumentals_dir", args.instrumentals_dir,
        "--alignments_dir", args.alignments_json_dir,
        "--processed_output_dir", args.preprocessed_output_dir,
        # Pass stem separation info to preprocess_dataset.py
        "--pre_separated_stems_root_dir", args.stem_output_base_dir, # e.g., data/temp_demucs_separated
        "--separator_tool_used", "demucs", # Assuming demucs was used above
        "--demucs_model_name", args.demucs_model
    ]
    if args.force_reprocess_features:
        preprocess_args.append("--force_reprocess")

    if not run_python_script("preprocess_dataset.py", "Preprocess Dataset (Features & Flow)", script_args=preprocess_args):
        return # Halt if this critical step fails


    # --- Phase 4a: Tokenization for Lite Model ---
    print("\nPHASE 4a: Tokenization for Lite Model")
    # 05a_tokenize_data_lite.py reads its config from lite_model_training/data_config_lite.yaml
    # That YAML should point to args.preprocessed_output_dir for its input
    # and args.tokenized_lite_dir for its output (or configure via YAML)
    if not run_python_script("05a_tokenize_data_lite.py", "Tokenize Lite Data"):
        return


    # --- Phase 4b: Tokenization for Full Model ---
    print("\nPHASE 4b: Tokenization for Full Model")
    # 05b_tokenize_data_full.py reads its config from lite_model_training/data_config_full.yaml
    # Similar to lite, YAML should point to correct input/output dirs.
    if not run_python_script("05b_tokenize_data_full.py", "Tokenize Full Data"):
        return

    print("\n====== Full Data Preparation Pipeline Potentially Complete ======")
    print("If all steps were successful, your data should be ready for training.")
    print("Next steps:")
    print("  - To train the LITE model: python lite_model_training/train_lite_flow_model.py")
    print("  - To train the FULL model: python scripts/train_flow_model.py")
    print("  - To visualize flow: python beefai/evaluation/rhythm_visualizer.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full BeefAI data preparation pipeline.")
    parser.add_argument("--skip_setup_prompt", action="store_true", help="Skip the initial setup prompt.")
    parser.add_argument("--skip_data_acquisition", action="store_true", help="Skip data acquisition/organization phase.")
    parser.add_argument("--skip_source_separation", action="store_true", help="Skip source separation (Demucs).")
    parser.add_argument("--skip_forced_alignment", action="store_true", help="Skip forced alignment (whisper-timestamped).")
    
    parser.add_argument("--force_rerun_separation", action="store_true", help="Force re-running source separation even if outputs exist.")
    parser.add_argument("--force_rerun_alignment", action="store_true", help="Force re-running alignment even if outputs exist.")
    parser.add_argument("--force_reprocess_features", action="store_true", help="Force re-running feature extraction in preprocess_dataset.py (ignores caches).")

    # Path configurations
    parser.add_argument("--raw_songs_dir", default=DEFAULT_RAW_SONGS_DIR, help="Directory for raw song files.")
    parser.add_argument("--lyrics_dir", default=DEFAULT_LYRICS_DIR, help="Directory for lyric text files.")
    parser.add_argument("--instrumentals_dir", default=DEFAULT_INSTRUMENTALS_DIR, help="Output directory for instrumentals.")
    parser.add_argument("--acapellas_dir", default=DEFAULT_ACAPELLAS_DIR, help="Output directory for acapellas.")
    parser.add_argument("--alignments_json_dir", default=DEFAULT_ALIGNMENTS_JSON_DIR, help="Output directory for alignment JSONs.")
    parser.add_argument("--preprocessed_output_dir", default=DEFAULT_PREPROCESSED_OUTPUT_DIR, help="Output directory for preprocess_dataset.py.")
    # Tokenized data dirs are usually configured in data_config_lite/full.yaml, so not direct args here.

    # Stem separation specific
    parser.add_argument("--stem_output_base_dir", default=DEFAULT_STEM_OUTPUT_BASE_DIR, help="Base output directory for Demucs (before model name subdir).")
    parser.add_argument("--demucs_model", default=DEFAULT_DEMUCS_MODEL, help="Demucs model to use (e.g., htdemucs_ft).")
    
    args = parser.parse_args()
    main(args)