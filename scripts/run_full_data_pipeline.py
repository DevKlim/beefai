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

def find_uvr_output_stems(uvr_output_dir: str, original_basename: str) -> tuple[str | None, str | None]:
    """
    Finds vocal and instrumental stems in UVR's output directory.
    UVR often names files like "basename (Vocals).ext" and "basename (Instrumental).ext".
    The exact naming can vary slightly based on UVR version and settings.
    """
    vocals_path = None
    instrumental_path = None
    
    # Common naming patterns. Adjust if your UVR CLI version outputs differently.
    # Note: UVR might put outputs in a subdirectory named after the model within uvr_output_dir.
    # This function assumes uvr_output_dir is the *final* directory containing the stems.
    
    possible_vocals_suffixes = [f" (Vocals).mp3", f"_Vocals.mp3", f".Vocals.mp3"]
    possible_instrumental_suffixes = [f" (Instrumental).mp3", f"_Instrumental.mp3", f".Instrumental.mp3", f" (Instruments).mp3", f"_Instruments.mp3"]

    for suffix in possible_vocals_suffixes:
        p_test = os.path.join(uvr_output_dir, original_basename + suffix)
        if os.path.exists(p_test):
            vocals_path = p_test
            break
            
    for suffix in possible_instrumental_suffixes:
        p_test = os.path.join(uvr_output_dir, original_basename + suffix)
        if os.path.exists(p_test):
            instrumental_path = p_test
            break
            
    # Fallback: Check for files just named "vocals.mp3" or "instrumental.mp3" if UVR places them in a song-specific subfolder
    if not vocals_path and os.path.exists(os.path.join(uvr_output_dir, "vocals.mp3")):
        vocals_path = os.path.join(uvr_output_dir, "vocals.mp3")
    if not instrumental_path and os.path.exists(os.path.join(uvr_output_dir, "instrumental.mp3")): # Or "instruments.mp3"
        instrumental_path = os.path.join(uvr_output_dir, "instrumental.mp3")
    if not instrumental_path and os.path.exists(os.path.join(uvr_output_dir, "instruments.mp3")):
        instrumental_path = os.path.join(uvr_output_dir, "instruments.mp3")


    return vocals_path, instrumental_path


def main(args):
    base_dataset_dir = args.dataset_dir
    
    raw_songs_full_dir = os.path.join(base_dataset_dir, "raw_songs_full")
    instrumentals_dir = os.path.join(base_dataset_dir, "instrumentals")
    acapellas_dir = os.path.join(base_dataset_dir, "acapellas")
    lyrics_dir = os.path.join(base_dataset_dir, "lyrics") 
    
    # Temp directories for separation tools
    temp_demucs_separated_dir = os.path.join(base_dataset_dir, "temp_demucs_separated") 
    temp_uvr_separated_dir = os.path.join(base_dataset_dir, "temp_uvr_separated") # UVR specific temp output

    mfa_input_corpus_dir = os.path.join(base_dataset_dir, "temp_mfa_input_corpus") 
    alignments_textgrid_dir = os.path.join(base_dataset_dir, "alignments_textgrid")
    
    processed_transformer_dir = os.path.join(base_dataset_dir, "processed_for_transformer")
    final_tokenized_output_file = os.path.join(processed_transformer_dir, "tokenized_flow_dataset.pt")
    tokenizer_config_path = os.path.join(os.getcwd(), "beefai", "flow_model", "flow_tokenizer_config_v2.json")

    ensure_dir(instrumentals_dir)
    ensure_dir(acapellas_dir)
    if args.separator_tool == "demucs": ensure_dir(temp_demucs_separated_dir)
    if args.separator_tool == "uvr": ensure_dir(temp_uvr_separated_dir)
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
    if args.separator_tool == "uvr":
        if not args.uvr_cli_path or not os.path.exists(args.uvr_cli_path):
            print(f"Error: UVR CLI path '{args.uvr_cli_path}' not found or not specified. UVR separation cannot proceed.")
            return
        if not args.uvr_model_name:
            print(f"Error: UVR model name (--uvr_model_name) not specified. UVR separation cannot proceed.")
            return
        print(f"UVR CLI Path: {args.uvr_cli_path}")
        print(f"UVR Model: {args.uvr_model_name}")


    # --- Step 1: Source Separation ---
    if not args.skip_separator:
        print(f"\n=== STEP 1: Source Separation ({args.separator_tool.upper()}) ===")
        for song_name in tqdm(song_basenames, desc=f"{args.separator_tool.upper()} Processing"):
            raw_song_path = None
            raw_song_extension = None
            for ext in ['.mp3', '.wav', '.flac']: 
                p = os.path.join(raw_songs_full_dir, song_name + ext)
                if os.path.exists(p):
                    raw_song_path = p
                    raw_song_extension = ext
                    break
            
            if not raw_song_path:
                print(f"  Skipping separation for {song_name}: Raw audio file not found.")
                continue

            final_acapella_path = os.path.join(acapellas_dir, f"{song_name}.mp3") # Standardize to mp3 for pipeline
            final_instrumental_path = os.path.join(instrumentals_dir, f"{song_name}.mp3")

            if not args.force_separator and os.path.exists(final_acapella_path) and os.path.exists(final_instrumental_path):
                print(f"  Skipping {args.separator_tool} for {song_name}: Separated files already exist in final destination.")
                continue
            
            # --- Demucs Logic ---
            if args.separator_tool == "demucs":
                demucs_output_song_dir = os.path.join(temp_demucs_separated_dir, args.demucs_model, song_name)
                expected_vocals_demucs_path = os.path.join(demucs_output_song_dir, "vocals.mp3")
                expected_no_vocals_demucs_path = os.path.join(demucs_output_song_dir, "no_vocals.mp3")
                
                demucs_cmd = [
                    PYTHON_EXECUTABLE, "-m", "demucs",
                    "--mp3", "--mp3-bitrate", "320", # Ensure good quality MP3 output
                    "--two-stems=vocals", 
                    "-n", args.demucs_model,
                    "-o", temp_demucs_separated_dir, # Demucs creates subdirs by model then songname
                    raw_song_path
                ]
                if not run_command(demucs_cmd, step_name=f"Demucs for {song_name}"):
                    print(f"  Demucs failed for {song_name}.")
                    continue

                if os.path.exists(expected_vocals_demucs_path) and os.path.exists(expected_no_vocals_demucs_path):
                    try:
                        shutil.move(expected_vocals_demucs_path, final_acapella_path)
                        shutil.move(expected_no_vocals_demucs_path, final_instrumental_path)
                        print(f"  Moved Demucs output for {song_name} to final destinations.")
                    except Exception as e:
                        print(f"  Error moving Demucs output for {song_name}: {e}")
                else:
                    print(f"  Demucs output not found at expected paths for {song_name}. Check Demucs logs.")

            # --- UVR Logic ---
            elif args.separator_tool == "uvr":
                # UVR CLI often outputs to a subfolder within the specified -o path,
                # or directly into -o with specific naming. This needs careful handling.
                # Let's assume UVR outputs into a song-specific subfolder within temp_uvr_separated_dir
                # or directly into temp_uvr_separated_dir with names like "song_name (Vocals).mp3"
                
                # The output path for UVR CLI. It might create subdirectories here.
                # For simplicity, let's target a general temp UVR output, then find files.
                # Some UVR CLIs might output directly to the specified path with modified names.
                # A common pattern is `uvr -i <input> -o <output_dir> -p <model_name>`
                # The output files appear in `<output_dir>` or `<output_dir>/<model_name_subfolder>/`
                # with names like `song_name (Vocals).mp3`
                
                # Let's make a song-specific temp output for UVR to avoid clashes if it doesn't make its own subdirs
                uvr_song_output_temp_dir = os.path.join(temp_uvr_separated_dir, song_name + "_uvr_out")
                ensure_dir(uvr_song_output_temp_dir)

                uvr_cmd = [
                    args.uvr_cli_path, # Path to the UVR CLI executable/script
                    "-i", raw_song_path,
                    "-o", uvr_song_output_temp_dir, # Output directory for this specific song
                    "-m", args.uvr_model_name, # The MDX-Net model file name (e.g., UVR-MDX-NET-Main.onnx)
                    # Add other necessary UVR CLI flags:
                    # Example flags (these VARY GREATLY between UVR CLI versions - CHECK YOUR UVR DOCUMENTATION)
                    # "--model_path", "/path/to/uvr_models_folder/", # If model isn't found by name
                    "--output_format", "MP3", # Or "FLAC" or "WAV"
                    # "--gpu", "0", # To use GPU if available and supported
                    # "--mdx_segment_size", "256", # Common MDX-Net param
                    # "--mdx_overlap", "0.25",     # Common MDX-Net param
                    # "--mdx_stems", "vocals", "instrumental" # Or how your CLI specifies 2-stem
                ]
                # Add common MDX-Net parameters if your CLI supports them directly
                if args.mdx_segment_size: uvr_cmd.extend(["--mdx_segment_size", str(args.mdx_segment_size)])
                if args.mdx_overlap: uvr_cmd.extend(["--mdx_overlap", str(args.mdx_overlap)])
                if args.uvr_gpu_conversion is not None: uvr_cmd.extend(["--gpu_conversion", str(args.uvr_gpu_conversion)])


                if not run_command(uvr_cmd, step_name=f"UVR for {song_name}"):
                    print(f"  UVR failed for {song_name}.")
                    # shutil.rmtree(uvr_song_output_temp_dir, ignore_errors=True) # Clean up temp dir
                    continue
                
                # Find the separated stems. UVR output naming can vary.
                # The original basename for UVR output might not include the original extension.
                uvr_original_basename_no_ext = song_name 
                
                # Try finding stems directly in uvr_song_output_temp_dir
                vocals_stem_path, instrumental_stem_path = find_uvr_output_stems(uvr_song_output_temp_dir, uvr_original_basename_no_ext)

                # If not found, UVR might have created another subfolder (e.g., model name)
                if not vocals_stem_path or not instrumental_stem_path:
                    # List subdirectories in uvr_song_output_temp_dir
                    subdirs = [d for d in os.listdir(uvr_song_output_temp_dir) if os.path.isdir(os.path.join(uvr_song_output_temp_dir, d))]
                    if subdirs: # If there's a subdirectory, assume stems are there
                        # This heuristic takes the first subdirectory. Might need refinement if multiple exist.
                        potential_stem_dir = os.path.join(uvr_song_output_temp_dir, subdirs[0])
                        print(f"  Searching for UVR stems in subdirectory: {potential_stem_dir}")
                        vocals_stem_path, instrumental_stem_path = find_uvr_output_stems(potential_stem_dir, uvr_original_basename_no_ext)


                if vocals_stem_path and instrumental_stem_path:
                    try:
                        shutil.move(vocals_stem_path, final_acapella_path)
                        shutil.move(instrumental_stem_path, final_instrumental_path)
                        print(f"  Moved UVR output for {song_name} to final destinations.")
                    except Exception as e:
                        print(f"  Error moving UVR output for {song_name}: {e}")
                else:
                    print(f"  UVR output stems not found for {song_name} in {uvr_song_output_temp_dir} or its subdirectories. Check UVR CLI output naming and paths.")
                
                # Clean up the song-specific UVR temp directory
                # shutil.rmtree(uvr_song_output_temp_dir, ignore_errors=True)
            else:
                print(f"Error: Unknown separator tool '{args.separator_tool}'")
                continue
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
                try:
                    shutil.copy(lyric_txt_path, lyric_txt_path_mfa)
                except Exception as e:
                    print(f"  Failed to copy lyric file for {song_name} to MFA corpus: {e}")
                    continue
            mfa_corpus_prepared_count +=1
        
        if mfa_corpus_prepared_count == 0:
            print("  No files prepared for MFA. Check acapella and lyric files. MFA step will be skipped.")
        else:
            print(f"  Prepared {mfa_corpus_prepared_count} songs for MFA in {mfa_input_corpus_dir}")

    if not args.skip_mfa and mfa_corpus_prepared_count > 0: 
        print("\n=== STEP 2b: Forced Alignment (MFA) ===")
        run_mfa_align = True
        if not args.force_mfa and os.path.exists(alignments_textgrid_dir) and os.listdir(alignments_textgrid_dir):
            expected_textgrids = len([f for f in os.listdir(mfa_input_corpus_dir) if f.endswith(".wav")])
            actual_textgrids = len(glob.glob(os.path.join(alignments_textgrid_dir, "*.TextGrid")))
            if actual_textgrids >= expected_textgrids and expected_textgrids > 0 : 
                print(f"  Skipping MFA alignment: Output directory '{alignments_textgrid_dir}' seems populated and not forcing.")
                run_mfa_align = False
        
        if run_mfa_align:
            if not args.mfa_acoustic_model or not args.mfa_dictionary:
                print("Error: MFA acoustic model or dictionary path not provided. Skipping MFA.")
            elif not os.path.exists(args.mfa_acoustic_model):
                print(f"Error: MFA acoustic model not found at {args.mfa_acoustic_model}. Skipping MFA.")
            elif not os.path.exists(args.mfa_dictionary):
                 print(f"Error: MFA dictionary not found at {args.mfa_dictionary}. Skipping MFA.")
            else:
                mfa_align_cmd = [
                    "mfa", "align",
                    mfa_input_corpus_dir,
                    args.mfa_dictionary,
                    args.mfa_acoustic_model,
                    alignments_textgrid_dir,
                    "--clean", 
                    "--overwrite", 
                    "-j", str(args.mfa_jobs) 
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
    
    # Separator tool choice
    parser.add_argument("--separator_tool", type=str, choices=["demucs", "uvr"], default="demucs", help="Source separation tool to use.")
    parser.add_argument("--force_separator", action="store_true", help="Force re-run selected separator even if files exist.")
    parser.add_argument("--skip_separator", action="store_true", help="Skip the source separation step.")

    # Demucs Args (only used if separator_tool is demucs)
    parser.add_argument("--demucs_model", type=str, default="htdemucs_ft", help="Demucs model name.")

    # UVR Args (only used if separator_tool is uvr)
    parser.add_argument("--uvr_cli_path", type=str, help="Path to the UVR CLI executable/script (e.g., path/to/uvr/python.exe or path/to/uvr_cli.py).")
    parser.add_argument("--uvr_model_name", type=str, help="Name of the MDX-Net model file for UVR (e.g., UVR-MDX-NET-Main.onnx). Must be in UVR's model path.")
    parser.add_argument("--mdx_segment_size", type=int, default=256, help="Segment size for MDX-Net models in UVR (if supported by CLI).")
    parser.add_argument("--mdx_overlap", type=float, default=0.25, help="Overlap for MDX-Net models in UVR (if supported by CLI).")
    parser.add_argument("--uvr_gpu_conversion", type=str, default=None, help="GPU ID for UVR conversion (e.g., '0'), or None/empty for CPU. Check your UVR CLI docs.")


    # MFA Args
    parser.add_argument("--mfa_acoustic_model", type=str, required=False, help="Path to MFA acoustic model (.zip file).")
    parser.add_argument("--mfa_dictionary", type=str, required=False, help="Path to MFA pronunciation dictionary (.dict file).")
    parser.add_argument("--mfa_jobs", type=int, default=4, help="Number of parallel jobs for MFA.")
    parser.add_argument("--force_mfa", action="store_true", help="Force re-run MFA alignment and corpus prep.")
    parser.add_argument("--skip_mfa", action="store_true", help="Skip the MFA alignment step.")

    # Preprocess_dataset.py Args
    parser.add_argument("--sample_rate_preprocess", type=int, default=44100, help="Sample rate for preprocess_dataset.py.")
    parser.add_argument("--force_preprocess", action="store_true", help="Force re-run of preprocess_dataset.py script.")
    parser.add_argument("--force_preprocess_script_caches", action="store_true", help="Pass --force_reprocess to preprocess_dataset.py.")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip running scripts/preprocess_dataset.py.")
    
    args = parser.parse_args()

    if not args.skip_mfa and (not args.mfa_acoustic_model or not args.mfa_dictionary):
        parser.error("--mfa_acoustic_model and --mfa_dictionary are required if --skip_mfa is not set.")
    
    if args.separator_tool == "uvr" and (not args.uvr_cli_path or not args.uvr_model_name):
        parser.error("--uvr_cli_path and --uvr_model_name are required if --separator_tool is 'uvr'.")


    main(args)