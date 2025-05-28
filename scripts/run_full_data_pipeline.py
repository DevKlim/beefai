import os
import shutil
import subprocess
import argparse
import glob
from tqdm import tqdm
import sys

PYTHON_EXECUTABLE = sys.executable

def ensure_dir(directory_path: str):
    os.makedirs(directory_path, exist_ok=True)

def run_command(command_list: list, cwd: str = None, step_name: str = ""):
    print(f"\n--- Running {step_name if step_name else 'command'}: {' '.join(command_list)} ---")
    try:
        process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, cwd=cwd)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"Error: {step_name if step_name else 'Command'} failed with return code {process.returncode}")
            return False
        print(f"--- {step_name if step_name else 'Command'} completed successfully ---")
        return True
    except FileNotFoundError:
        print(f"Error: Command not found for '{step_name}'. Is '{command_list[0]}' in your PATH or is it the correct path?")
        print(f"Attempted to run: {' '.join(command_list)}")
        return False
    except Exception as e:
        print(f"Exception during {step_name if step_name else 'command'}: {e}")
        return False

def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found in PATH. Cannot convert MP3 to WAV for MFA.")
        return False
    command = ["ffmpeg", "-i", mp3_path, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "44100", wav_path, "-y", "-hide_banner", "-loglevel", "error"]
    return run_command(command, step_name=f"ffmpeg conversion for {os.path.basename(mp3_path)}")

def main(args):
    base_dataset_dir = args.dataset_dir
    
    raw_songs_full_dir = os.path.join(base_dataset_dir, "raw_songs_full")
    instrumentals_dir = os.path.join(base_dataset_dir, "instrumentals")
    acapellas_dir = os.path.join(base_dataset_dir, "acapellas")
    lyrics_dir = os.path.join(base_dataset_dir, "lyrics") 
    
    # For audio-separator, output is directly to final acapellas/instrumentals
    # For Demucs, we use a temp directory
    temp_demucs_separated_dir = os.path.join(base_dataset_dir, "temp_demucs_separated") 
    
    mfa_input_corpus_dir = os.path.join(base_dataset_dir, "temp_mfa_input_corpus") 
    alignments_textgrid_dir = os.path.join(base_dataset_dir, "alignments_textgrid")
    
    processed_transformer_dir = os.path.join(base_dataset_dir, "processed_for_transformer")
    final_tokenized_output_file = os.path.join(processed_transformer_dir, "tokenized_flow_dataset.pt")
    tokenizer_config_path = os.path.join(os.getcwd(), "beefai", "flow_model", "flow_tokenizer_config_v2.json")

    ensure_dir(instrumentals_dir)
    ensure_dir(acapellas_dir)
    if args.separator_tool == "demucs":
        ensure_dir(temp_demucs_separated_dir)
    ensure_dir(mfa_input_corpus_dir)
    ensure_dir(alignments_textgrid_dir)
    ensure_dir(processed_transformer_dir)

    song_basenames = sorted([os.path.splitext(f)[0] for f in os.listdir(raw_songs_full_dir) if f.lower().endswith(('.mp3', '.wav', '.flac'))])
    if not song_basenames:
        print(f"No audio files found in {raw_songs_full_dir}. Exiting.")
        return

    print(f"Found {len(song_basenames)} songs to process: {', '.join(song_basenames[:5])}...")
    print(f"Using Python executable: {PYTHON_EXECUTABLE} for sub-scripts.")
    print(f"Selected separator tool: {args.separator_tool}")

    # --- Step 1: Source Separation ---
    if not args.skip_separator:
        print("\n=== STEP 1: Source Separation ===")
        for song_name in tqdm(song_basenames, desc=f"{args.separator_tool.capitalize()} Processing"):
            raw_song_path = None
            for ext in ['.mp3', '.wav', '.flac']: 
                p = os.path.join(raw_songs_full_dir, song_name + ext)
                if os.path.exists(p):
                    raw_song_path = p
                    break
            
            if not raw_song_path:
                print(f"  Skipping separation for {song_name}: Raw audio file not found.")
                continue

            final_acapella_path = os.path.join(acapellas_dir, f"{song_name}.mp3") # audio-separator can output mp3
            final_instrumental_path = os.path.join(instrumentals_dir, f"{song_name}.mp3")

            if not args.force_separator and os.path.exists(final_acapella_path) and os.path.exists(final_instrumental_path):
                print(f"  Skipping separation for {song_name}: Separated files already exist in final destination.")
                continue
            
            success = False
            if args.separator_tool == "demucs":
                demucs_output_song_dir = os.path.join(temp_demucs_separated_dir, args.demucs_model, song_name)
                expected_vocals_demucs_path = os.path.join(demucs_output_song_dir, "vocals.mp3")
                expected_no_vocals_demucs_path = os.path.join(demucs_output_song_dir, "no_vocals.mp3")
                
                demucs_cmd = [
                    PYTHON_EXECUTABLE, "-m", "demucs",
                    "--mp3", "--two-stems=vocals", "-n", args.demucs_model,
                    "-o", temp_demucs_separated_dir, raw_song_path
                ]
                if run_command(demucs_cmd, step_name=f"Demucs for {song_name}"):
                    if os.path.exists(expected_vocals_demucs_path) and os.path.exists(expected_no_vocals_demucs_path):
                        try:
                            shutil.move(expected_vocals_demucs_path, final_acapella_path)
                            shutil.move(expected_no_vocals_demucs_path, final_instrumental_path)
                            print(f"  Moved Demucs output for {song_name} to final destinations.")
                            success = True
                        except Exception as e:
                            print(f"  Error moving Demucs output for {song_name}: {e}")
                    else:
                        print(f"  Demucs output not found at expected paths for {song_name}.")
            
            elif args.separator_tool == "audio_separator":
                # audio-separator usually outputs: output_dir/song_name (Vocals).mp3 and song_name (Instrumental).mp3
                # We want to control output names more directly.
                # Let's output to a temporary song-specific folder then move.
                temp_audio_sep_output_dir = os.path.join(base_dataset_dir, "temp_audio_sep_output", song_name)
                ensure_dir(temp_audio_sep_output_dir)

                audio_sep_cmd = [
                    PYTHON_EXECUTABLE, "-m", "audio_separator", raw_song_path,
                    "--model_name", args.audio_separator_model,
                    "--output_dir", temp_audio_sep_output_dir,
                    "--output_format", "MP3", # Outputting as MP3
                    # "--log_level", "DEBUG" # For more verbose output if needed
                ]
                if args.audio_separator_gpu:
                     audio_sep_cmd.append("--use_cuda") # Or other GPU flags like --gpu_device if needed

                if run_command(audio_sep_cmd, step_name=f"Audio-Separator for {song_name}"):
                    # Expected output names from audio-separator might vary slightly based on its version
                    # Common pattern: {filename_without_ext} ({stem_name}).{ext}
                    original_filename_no_ext = os.path.splitext(os.path.basename(raw_song_path))[0]
                    
                    # Try common naming patterns for vocals and instrumental stems
                    # This might need adjustment based on the exact output of your audio-separator version
                    # and the specific MDX-Net model used.
                    # Some models might output "primary_vocals" vs "vocals".
                    # The model "MDX23C-InstVoc_HQ_1.ckpt" is good for instrumental/vocals.
                    # It usually outputs "Instrumental" and "Vocals".
                    
                    # Let's assume the output files are named based on the input filename + stem type
                    # e.g. "SongTitle (Vocals).mp3", "SongTitle (Instrumental).mp3"
                    # Or if it's simpler, it might just be "vocals.mp3" and "instrumental.mp3"
                    # within the temp_audio_sep_output_dir.
                    # The most robust way is to list files in temp_audio_sep_output_dir.

                    found_vocals = None
                    found_instrumental = None
                    
                    # List files in the temp output directory for this song
                    # audio-separator typically creates files like:
                    # {output_dir}/{model_name}/{input_filename_stem}_(Vocals).{ext}
                    # {output_dir}/{model_name}/{input_filename_stem}_(Instrumental).{ext}
                    # The output_dir for audio-separator command is temp_audio_sep_output_dir
                    # So the files will be in temp_audio_sep_output_dir directly (or in a model-named subfolder)

                    # Let's assume audio-separator places them in temp_audio_sep_output_dir/{model_name}/
                    # If not, adjust this path.
                    # If audio-separator --output_dir is set, it creates files like:
                    # output_dir/filename_Vocals.mp3, output_dir/filename_Instrumental.mp3
                    # So, they should be directly in temp_audio_sep_output_dir

                    # Try to find vocals and instrumental based on common naming conventions
                    for f_name in os.listdir(temp_audio_sep_output_dir):
                        f_path = os.path.join(temp_audio_sep_output_dir, f_name)
                        if "vocals" in f_name.lower() and f_name.lower().endswith(".mp3"): # Case-insensitive check
                            found_vocals = f_path
                        elif "instrumental" in f_name.lower() and f_name.lower().endswith(".mp3"):
                            found_instrumental = f_path
                        # Fallback for 'no_vocals' if 'instrumental' isn't found
                        elif "no_vocals" in f_name.lower() and f_name.lower().endswith(".mp3") and not found_instrumental:
                            found_instrumental = f_path

                    if found_vocals and found_instrumental:
                        try:
                            shutil.move(found_vocals, final_acapella_path)
                            shutil.move(found_instrumental, final_instrumental_path)
                            print(f"  Moved Audio-Separator output for {song_name} to final destinations.")
                            success = True
                        except Exception as e:
                            print(f"  Error moving Audio-Separator output for {song_name}: {e}")
                    else:
                        print(f"  Audio-Separator output stems not found as expected in '{temp_audio_sep_output_dir}'. Vocals: {found_vocals}, Instrumental: {found_instrumental}")
                
                # Clean up temporary audio-separator output directory for this song
                if os.path.exists(temp_audio_sep_output_dir):
                    shutil.rmtree(temp_audio_sep_output_dir)

            if not success:
                print(f"  Separation failed for {song_name}. Skipping further processing for this song.")
                # continue # This would skip MFA and preprocess for this song
    else:
        print("\n=== SKIPPING STEP 1: Source Separation ===")

    mfa_corpus_prepared_count = 0
    if not args.skip_mfa:
        print("\n=== STEP 2a: Preparing Corpus for MFA ===")
        ensure_dir(mfa_input_corpus_dir)
        if args.force_mfa and os.path.exists(mfa_input_corpus_dir):
            print(f"  Clearing existing MFA input corpus directory: {mfa_input_corpus_dir}")
            for item in os.listdir(mfa_input_corpus_dir):
                item_path = os.path.join(mfa_input_corpus_dir, item)
                if os.path.isfile(item_path) or os.path.islink(item_path): os.unlink(item_path)
                elif os.path.isdir(item_path): shutil.rmtree(item_path)

        for song_name in tqdm(song_basenames, desc="MFA Corpus Prep"):
            acapella_mp3_path = os.path.join(acapellas_dir, f"{song_name}.mp3")
            lyric_txt_path = os.path.join(lyrics_dir, f"{song_name}.txt")
            acapella_wav_path_mfa = os.path.join(mfa_input_corpus_dir, f"{song_name}.wav")
            lyric_txt_path_mfa = os.path.join(mfa_input_corpus_dir, f"{song_name}.txt")

            if not os.path.exists(acapella_mp3_path):
                print(f"  Skipping MFA prep for {song_name}: Acapella MP3 not found at {acapella_mp3_path}.")
                continue
            if not os.path.exists(lyric_txt_path):
                print(f"  Skipping MFA prep for {song_name}: Lyric TXT not found at {lyric_txt_path}.")
                continue

            if args.force_mfa or not os.path.exists(acapella_wav_path_mfa):
                if not convert_mp3_to_wav(acapella_mp3_path, acapella_wav_path_mfa):
                    print(f"  Failed to convert {song_name} acapella to WAV. Skipping MFA for this song.")
                    continue
            
            if args.force_mfa or not os.path.exists(lyric_txt_path_mfa):
                try: shutil.copy(lyric_txt_path, lyric_txt_path_mfa)
                except Exception as e:
                    print(f"  Failed to copy lyric file for {song_name} to MFA corpus: {e}")
                    continue
            mfa_corpus_prepared_count += 1
        
        if mfa_corpus_prepared_count == 0: print("  No files prepared for MFA. MFA step will be skipped.")
        else: print(f"  Prepared {mfa_corpus_prepared_count} songs for MFA in {mfa_input_corpus_dir}")

    if not args.skip_mfa and mfa_corpus_prepared_count > 0:
        print("\n=== STEP 2b: Forced Alignment (MFA) ===")
        run_mfa_align = True
        if not args.force_mfa and os.path.exists(alignments_textgrid_dir) and os.listdir(alignments_textgrid_dir):
            expected_textgrids = len([f for f in os.listdir(mfa_input_corpus_dir) if f.endswith(".wav")])
            actual_textgrids = len(glob.glob(os.path.join(alignments_textgrid_dir, "*.TextGrid")))
            if actual_textgrids >= expected_textgrids and expected_textgrids > 0:
                print(f"  Skipping MFA alignment: Output directory '{alignments_textgrid_dir}' seems populated and not forcing.")
                run_mfa_align = False
        
        if run_mfa_align:
            mfa_executable = args.mfa_executable_path if args.mfa_executable_path else "mfa"
            if not args.mfa_acoustic_model or not args.mfa_dictionary:
                print("Error: MFA acoustic model or dictionary path not provided. Skipping MFA.")
            elif not os.path.exists(args.mfa_acoustic_model):
                print(f"Error: MFA acoustic model not found at {args.mfa_acoustic_model}. Skipping MFA.")
            elif not os.path.exists(args.mfa_dictionary):
                print(f"Error: MFA dictionary not found at {args.mfa_dictionary}. Skipping MFA.")
            else:
                mfa_align_cmd = [
                    mfa_executable, "align", mfa_input_corpus_dir,
                    args.mfa_dictionary, args.mfa_acoustic_model,
                    alignments_textgrid_dir, "--clean", "--overwrite", "-j", str(args.mfa_jobs)
                ]
                if not run_command(mfa_align_cmd, step_name="MFA Alignment"):
                    print("  MFA alignment process failed. Check MFA logs.")
                else:
                    print(f"  MFA alignment complete. TextGrids should be in {alignments_textgrid_dir}")
    elif args.skip_mfa:
        print("\n=== SKIPPING STEP 2: Forced Alignment (MFA) & Corpus Prep ===")
    elif mfa_corpus_prepared_count == 0 and not args.skip_mfa:
        print("\n=== SKIPPING STEP 2: Forced Alignment (MFA) - No files were prepared for the corpus ===")

    if not args.skip_preprocess:
        print("\n=== STEP 3: Preprocessing for Transformer Model ===")
        run_preprocess_script = True
        if not args.force_preprocess and os.path.exists(final_tokenized_output_file) and os.path.getsize(final_tokenized_output_file) > 100:
            print(f"  Skipping preprocess_dataset.py: Final tokenized file '{final_tokenized_output_file}' already exists and is valid.")
            run_preprocess_script = False

        if run_preprocess_script:
            preprocess_cmd = [
                PYTHON_EXECUTABLE, os.path.join(os.getcwd(), "scripts", "preprocess_dataset.py"),
                "--dataset_base_dir", base_dataset_dir, 
                "--raw_songs_dir", instrumentals_dir, 
                "--lyrics_dir", lyrics_dir, 
                "--alignments_dir", alignments_textgrid_dir,
                "--processed_output_dir", processed_transformer_dir, 
                "--tokenizer_config", tokenizer_config_path,
                "--output_tokenized_file", final_tokenized_output_file,
                "--sample_rate", str(args.sample_rate_preprocess)
            ]
            if args.force_preprocess_script_caches: 
                preprocess_cmd.append("--force_reprocess")
            if not run_command(preprocess_cmd, step_name="Dataset Preprocessing Script"):
                print("  Dataset preprocessing script failed. Check its logs.")
            else:
                print(f"  Dataset preprocessing script complete. Tokenized data should be at {final_tokenized_output_file}")
    else:
        print("\n=== SKIPPING STEP 3: Preprocessing for Transformer Model ===")

    print("\n--- Full Data Pipeline Automation Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated data preparation pipeline for beefai flow model.")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Base directory for the dataset.")
    
    # Separator Args
    parser.add_argument("--separator_tool", type=str, default="demucs", choices=["demucs", "audio_separator"], help="Source separation tool to use.")
    parser.add_argument("--force_separator", action="store_true", help="Force re-run separation even if separated files exist.")
    parser.add_argument("--skip_separator", action="store_true", help="Skip the source separation step.")
    # Demucs specific
    parser.add_argument("--demucs_model", type=str, default="htdemucs_ft", help="Demucs model name.")
    # Audio-separator specific
    parser.add_argument("--audio_separator_model", type=str, default="MDX23C-InstVoc_HQ_1", help="Model name for audio-separator (e.g., from UVR MDX-Net models).")
    parser.add_argument("--audio_separator_gpu", action="store_true", help="Use GPU for audio-separator if available (requires CUDA setup).")


    # MFA Args
    parser.add_argument("--mfa_executable_path", type=str, default=None, help="Optional full path to the 'mfa' executable if not in system PATH.")
    parser.add_argument("--mfa_acoustic_model", type=str, required=False, help="Path to MFA acoustic model (.zip file). REQUIRED if not skipping MFA.")
    parser.add_argument("--mfa_dictionary", type=str, required=False, help="Path to MFA pronunciation dictionary (.dict file). REQUIRED if not skipping MFA.")
    parser.add_argument("--mfa_jobs", type=int, default=4, help="Number of parallel jobs for MFA.")
    parser.add_argument("--force_mfa", action="store_true", help="Force re-run MFA alignment and corpus prep even if outputs exist.")
    parser.add_argument("--skip_mfa", action="store_true", help="Skip the MFA alignment step (and its corpus prep).")

    # Preprocess_dataset.py Args
    parser.add_argument("--sample_rate_preprocess", type=int, default=44100, help="Sample rate for preprocess_dataset.py feature extraction.")
    parser.add_argument("--force_preprocess", action="store_true", help="Force re-run of preprocess_dataset.py script even if its final output exists.")
    parser.add_argument("--force_preprocess_script_caches", action="store_true", help="Pass --force_reprocess to preprocess_dataset.py to ignore its internal caches.")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip running the scripts/preprocess_dataset.py script.")
    
    args = parser.parse_args()

    if not args.skip_mfa and (not args.mfa_acoustic_model or not args.mfa_dictionary):
        parser.error("--mfa_acoustic_model and --mfa_dictionary are required if --skip_mfa is not set.")

    main(args)