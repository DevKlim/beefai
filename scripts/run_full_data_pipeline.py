import subprocess
import os
import sys
import argparse
import shutil 
import multiprocessing
from functools import partial
from tqdm import tqdm
import time 
import re 
import traceback

sys.path.append(os.getcwd())

# --- Virtual Environment Check ---
try:
    project_root_for_venv_check = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    expected_venv_dir_name = ".venv" 
    expected_venv_path = os.path.join(project_root_for_venv_check, expected_venv_dir_name)
    actual_venv_path_env = os.environ.get('VIRTUAL_ENV')
    actual_conda_prefix_env = os.environ.get('CONDA_PREFIX')
    is_in_expected_venv = False
    if actual_venv_path_env: 
        if os.path.abspath(actual_venv_path_env) == os.path.abspath(expected_venv_path): is_in_expected_venv = True
    elif actual_conda_prefix_env: 
        if os.path.abspath(actual_conda_prefix_env) == os.path.abspath(expected_venv_path): is_in_expected_venv = True
    if not actual_venv_path_env and not actual_conda_prefix_env and not is_in_expected_venv:
        if sys.prefix == os.path.abspath(expected_venv_path): is_in_expected_venv = True
    if not is_in_expected_venv:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        print("! WARNING: This script may not be running from the project's intended      !", flush=True)
        print("! virtual environment (.venv). Some commands (like whisper_timestamped,    !", flush=True)
        print("! demucs, audio-separator) might not be found or might use global versions.!", flush=True)
        print(f"! Expected venv: {expected_venv_path}", flush=True)
        print(f"! Current sys.executable: {sys.executable}", flush=True)
        print("! If issues occur, please activate the project's .venv and re-run.        !", flush=True)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        skip_prompt_local = False # Default to not skipping
        if 'parsed_args_for_main' in globals() and parsed_args_for_main is not None and hasattr(parsed_args_for_main, 'skip_venv_warning_prompt'):
             skip_prompt_local = parsed_args_for_main.skip_venv_warning_prompt
        if not skip_prompt_local:
            print("Press Enter to attempt to continue, or Ctrl+C to exit (auto-proceeds in 5s)...", flush=True)
            try: time.sleep(5); print("Proceeding automatically after timeout.", flush=True)
            except KeyboardInterrupt: print("Exiting due to user interruption.", flush=True); sys.exit(1)
except Exception as e_venv_check:
    print(f"Note: Could not perform robust virtual environment check: {e_venv_check}", flush=True)


# Default paths
DEFAULT_RAW_SONGS_DIR = "data/raw_songs_full/"
DEFAULT_LYRICS_DIR = "data/lyrics/"
DEFAULT_INSTRUMENTALS_DIR = "data/instrumentals/"
DEFAULT_ACAPELLAS_DIR = "data/acapellas/"
DEFAULT_ALIGNMENTS_JSON_DIR = "data/alignments_json/"
DEFAULT_PREPROCESSED_OUTPUT_DIR = "data/processed_for_transformer/"

DEFAULT_DEMUCS_STEM_OUTPUT_BASE_DIR = "data/temp_demucs_separated"
DEFAULT_AUDIO_SEPARATOR_STEM_OUTPUT_BASE_DIR = "data/temp_audio_separator_output"
DEFAULT_DEMUCS_MODEL = "htdemucs_ft"
DEFAULT_AUDIO_SEPARATOR_MODEL = "UVR_MDXNET_Main.onnx"


def sanitize_filename_for_command(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    name = name.replace("&", "and").replace(" ", "_")
    name = re.sub(r'[^\w\-\.]', '', name) 
    return f"{name}{ext}"


def run_command(command_list, step_name, working_dir=None, suppress_output=False, capture_stderr_on_error=False):
    print(f"DEBUG: run_command called for: {step_name} with command: {' '.join(command_list)}", flush=True)
    try:
        process = subprocess.Popen(
            command_list, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            cwd=working_dir,
            shell=sys.platform == "win32" 
        )
        print(f"DEBUG: run_command: Subprocess for {step_name} started (PID: {process.pid}). Waiting for communication...", flush=True)
        stdout, stderr = process.communicate()
        print(f"DEBUG: run_command: Subprocess for {step_name} finished communication. RC: {process.returncode}", flush=True)
        
        if not suppress_output and stdout and stdout.strip(): 
            print(f"Stdout from {step_name} for {' '.join(command_list[:3])}...:\n{stdout.strip()}", flush=True)

        if process.returncode != 0:
            print(f"ERROR: {step_name} failed for {' '.join(command_list)} with exit code {process.returncode}.", flush=True)
            if stderr and stderr.strip():
                print(f"Stderr from {step_name}:\n{stderr.strip()}", flush=True)
            elif stdout and stdout.strip() and suppress_output : 
                print(f"Stdout (on error, was suppressed) from {step_name}:\n{stdout.strip()}", flush=True)
            return False
        
        if not suppress_output and stderr and stderr.strip():
             print(f"Non-fatal Stderr from {step_name} for {' '.join(command_list[:3])}...:\n{stderr.strip()}", flush=True)
        
        print(f"DEBUG: run_command for {step_name} returning True", flush=True)
        return True
    except FileNotFoundError:
        print(f"ERROR: Command '{command_list[0]}' not found for step '{step_name}'. Ensure it's installed and in PATH for current env ({sys.executable}).", flush=True)
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during '{step_name}' for {' '.join(command_list)}: {e}", flush=True)
        traceback.print_exc() 
        return False

def align_song_worker(acapella_file_info, base_acapellas_dir, output_json_dir, whisper_model_arg, whisper_language_arg, force_rerun_arg):
    original_acapella_filename = acapella_file_info["filename"]
    sanitized_filename_for_processing = sanitize_filename_for_command(original_acapella_filename)
    original_acapella_path = os.path.join(base_acapellas_dir, original_acapella_filename)
    song_name_no_ext_original = os.path.splitext(original_acapella_filename)[0]
    output_json_path = os.path.join(output_json_dir, f"{song_name_no_ext_original}.json")

    if os.path.exists(output_json_path) and not force_rerun_arg:
        return {"filename": original_acapella_filename, "status": "skipped_exists"}

    path_for_whisper_input = original_acapella_path
    temp_sanitized_file_used = False
    temp_sanitized_path = ""

    if original_acapella_filename != sanitized_filename_for_processing:
        temp_sanitized_path = os.path.join(base_acapellas_dir, sanitized_filename_for_processing)
        try:
            shutil.copy2(original_acapella_path, temp_sanitized_path)
            path_for_whisper_input = temp_sanitized_path
            temp_sanitized_file_used = True
        except Exception as e_copy:
            print(f"ERROR: Could not create temporary sanitized copy for {original_acapella_filename}: {e_copy}. Skipping.", flush=True)
            return {"filename": original_acapella_filename, "status": "failed_copy"}

    align_cmd = [
        "whisper_timestamped", 
        path_for_whisper_input, 
        "--model", whisper_model_arg,
        "--output_dir", output_json_dir, 
        "--output_format", "json"
    ]
    if whisper_language_arg:
        align_cmd.extend(["--language", whisper_language_arg])

    success = run_command(align_cmd, f"Forced Alignment for {original_acapella_filename}", suppress_output=True, capture_stderr_on_error=True) 
    
    status_to_return = "failed"
    if success:
        expected_output_from_sanitized = os.path.join(output_json_dir, f"{os.path.splitext(sanitized_filename_for_processing)[0]}.json")
        if temp_sanitized_file_used and os.path.exists(expected_output_from_sanitized):
            try:
                if os.path.exists(output_json_path): 
                    os.remove(output_json_path)
                shutil.move(expected_output_from_sanitized, output_json_path)
                status_to_return = "aligned"
            except Exception as e_rename:
                print(f"ERROR: Failed to rename alignment output for {original_acapella_filename}: {e_rename}", flush=True)
                status_to_return = "failed_rename"
        elif not temp_sanitized_file_used and os.path.exists(output_json_path): 
            status_to_return = "aligned"
        else: 
            # Check if original output name exists (whisper_timestamped might use original name directly)
            original_output_name = f"{song_name_no_ext_original}.json"
            if os.path.exists(os.path.join(output_json_dir, original_output_name)):
                 status_to_return = "aligned"
            else:
                print(f"ERROR: Alignment command succeeded for {original_acapella_filename}, but expected output JSON not found (checked {output_json_path} and {expected_output_from_sanitized}).", flush=True)
                status_to_return = "failed_output_missing"
    
    if temp_sanitized_file_used and os.path.exists(temp_sanitized_path):
        try:
            os.remove(temp_sanitized_path)
        except Exception as e_del_temp:
            print(f"Warning: Could not delete temporary sanitized file {temp_sanitized_path}: {e_del_temp}", flush=True)
            
    return {"filename": original_acapella_filename, "status": status_to_return}


def run_python_script(script_path_relative_to_scripts_dir, script_name, script_args=None):
    print(f"DEBUG: run_python_script: Attempting to run script: {script_path_relative_to_scripts_dir} with name: {script_name}", flush=True)
    full_script_path = os.path.join("scripts", script_path_relative_to_scripts_dir)
    if not os.path.exists(full_script_path):
        print(f"ERROR: Script {full_script_path} not found. Skipping.", flush=True)
        return False
    
    command = [sys.executable, full_script_path] 
    if script_args:
        command.extend(script_args)
    
    print(f"DEBUG: run_python_script: Constructed command: {' '.join(command)}", flush=True)
    success = run_command(command, script_name, suppress_output=False) 
    print(f"DEBUG: run_python_script: {script_name} returned: {success}", flush=True)
    return success


def main(args_main_func): 
    global parsed_args_for_main 
    parsed_args_for_main = args_main_func

    print("====== Starting Full Data Preparation Pipeline ======", flush=True)

    print("\nPHASE 0: Setup", flush=True)
    print("Ensuring Python environment is set up, dependencies installed, and API keys are configured.", flush=True)
    # Add actual checks here if needed, e.g., for API keys from .env
    if not parsed_args_for_main.skip_setup_prompt:
        print("Press Enter to continue if setup is complete (auto-proceeds in 5s)...", flush=True)
        try: time.sleep(5); print("Proceeding automatically after timeout.", flush=True)
        except KeyboardInterrupt: print("Exiting due to user interruption.", flush=True); sys.exit(1)

    if not parsed_args_for_main.skip_data_acquisition:
        print("\nPHASE 1: Data Acquisition & Organization (Conceptual)", flush=True)
        print("  (This phase typically involves manual collection or running download scripts like download_youtube_lyrics.py,", flush=True)
        print("   followed by organize_downloaded_songs.py. These are not auto-run by default here.)", flush=True)
    else: print("\nSkipping PHASE 1: Data Acquisition & Organization", flush=True)

    if not parsed_args_for_main.skip_source_separation:
        print(f"\nPHASE 1b: Source Separation (Using: {parsed_args_for_main.separator_tool})", flush=True)
        os.makedirs(parsed_args_for_main.instrumentals_dir, exist_ok=True)
        os.makedirs(parsed_args_for_main.acapellas_dir, exist_ok=True)
        
        if parsed_args_for_main.separator_tool == "demucs":
            os.makedirs(parsed_args_for_main.demucs_stem_output_base_dir, exist_ok=True)
            sep_check_cmd = ["demucs", "--version"]
        elif parsed_args_for_main.separator_tool == "audio_separator":
            os.makedirs(parsed_args_for_main.audio_separator_stem_output_base_dir, exist_ok=True)
            sep_check_cmd = ["audio-separator", "--help"] # audio-separator --version might not exist
        else:
            print(f"ERROR: Unknown separator tool: {parsed_args_for_main.separator_tool}", flush=True)
            return

        print(f"DEBUG: Checking {parsed_args_for_main.separator_tool} availability with: {' '.join(sep_check_cmd)}", flush=True)
        if not run_command(sep_check_cmd, f"{parsed_args_for_main.separator_tool} check", suppress_output=True, capture_stderr_on_error=True):
            print(f"ERROR: {parsed_args_for_main.separator_tool} command check failed. Halting.", flush=True)
            return

        raw_song_files = [f for f in os.listdir(parsed_args_for_main.raw_songs_dir) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]
        print(f"Found {len(raw_song_files)} songs in {parsed_args_for_main.raw_songs_dir} for separation.", flush=True)

        for filename in tqdm(raw_song_files, desc="Separating Sources"):
            song_path = os.path.join(parsed_args_for_main.raw_songs_dir, filename)
            song_name_no_ext = os.path.splitext(filename)[0]
            
            instrumental_out_path = os.path.join(parsed_args_for_main.instrumentals_dir, f"{song_name_no_ext}.wav")
            acapella_out_path = os.path.join(parsed_args_for_main.acapellas_dir, f"{song_name_no_ext}.wav")

            if (os.path.exists(instrumental_out_path) and os.path.exists(acapella_out_path)) and not parsed_args_for_main.force_rerun_separation:
                print(f"  Skipping separation for {filename}, outputs already exist.", flush=True)
                continue
            
            if parsed_args_for_main.separator_tool == "demucs":
                # Output structure for demucs is typically: out_dir/model_name/song_name_no_ext/{vocals.wav, bass.wav, drums.wav, other.wav}
                demucs_song_output_dir = os.path.join(parsed_args_for_main.demucs_stem_output_base_dir, parsed_args_for_main.demucs_model, song_name_no_ext)
                os.makedirs(demucs_song_output_dir, exist_ok=True) # Demucs will create this if -o is used correctly.
                
                sep_cmd = ["demucs", "-n", parsed_args_for_main.demucs_model, "-o", parsed_args_for_main.demucs_stem_output_base_dir, "--filename", "{track}/{stem}.{ext}", song_path]
                # Demucs --filename expects {track} to be the input track name without extension, not the full path.
                # It will place output in: demucs_stem_output_base_dir / demucs_model / song_name_no_ext / stem.wav
                
                if run_command(sep_cmd, f"Demucs Separation for {filename}", suppress_output=True):
                    # Expected output paths after demucs
                    expected_demucs_vocals = os.path.join(demucs_song_output_dir, "vocals.wav")
                    expected_demucs_no_vocals = os.path.join(demucs_song_output_dir, "no_vocals.wav") # if supported, or sum others
                    
                    # For instrumentals, we often use "no_vocals" if the model provides it, or sum of (bass, drums, other)
                    # For this pipeline, BeatFeatureExtractor can use individual stems like bass.wav, drums.wav.
                    # We need *an* instrumental and *an* acapella for the main data dirs.
                    if os.path.exists(expected_demucs_vocals):
                        shutil.copy2(expected_demucs_vocals, acapella_out_path)
                    else: print(f"Warning: Demucs vocals stem not found for {filename} at {expected_demucs_vocals}", flush=True)

                    # Create instrumental by combining non-vocal stems if 'no_vocals.wav' isn't directly output
                    # Or simply copy 'other.wav' if that's the desired instrumental proxy for simplicity in this script
                    # This part can be complex. For now, let's assume a simpler path or that BFE handles stems.
                    # Here, we just need a file for data/instrumentals. Using 'other.wav' as a proxy if no_vocals isn't available
                    if os.path.exists(expected_demucs_no_vocals):
                        shutil.copy2(expected_demucs_no_vocals, instrumental_out_path)
                    elif os.path.exists(os.path.join(demucs_song_output_dir, "other.wav")): # Fallback
                         shutil.copy2(os.path.join(demucs_song_output_dir, "other.wav"), instrumental_out_path)
                    else: print(f"Warning: Demucs instrumental stem not found for {filename}", flush=True)


            elif parsed_args_for_main.separator_tool == "audio_separator":
                # audio-separator typically outputs directly to specified files or a simple dir
                # Model name is part of the -m argument, output dir can be specified.
                # Let's make it output to a temp song-specific dir first.
                audio_sep_song_temp_dir = os.path.join(parsed_args_for_main.audio_separator_stem_output_base_dir, song_name_no_ext)
                os.makedirs(audio_sep_song_temp_dir, exist_ok=True)

                sep_cmd = [
                    "audio-separator", song_path,
                    "--model_name", parsed_args_for_main.audio_separator_model_filename,
                    "--output_dir", audio_sep_song_temp_dir,
                    "--output_format", "WAV"
                ]
                if parsed_args_for_main.use_autocast_for_separator: sep_cmd.append("--use_cuda_amp")
                
                if run_command(sep_cmd, f"Audio-Separator for {filename}", suppress_output=True):
                    # audio-separator output naming convention: {filename_without_ext}_(Vocals).wav, {filename_without_ext}_(Instrumental).wav
                    # It uses the *original* input filename for these.
                    sanitized_input_song_name_no_ext = os.path.splitext(sanitize_filename_for_command(filename))[0]

                    expected_as_vocals = os.path.join(audio_sep_song_temp_dir, f"{sanitized_input_song_name_no_ext}_(Vocals).wav")
                    expected_as_instrumental = os.path.join(audio_sep_song_temp_dir, f"{sanitized_input_song_name_no_ext}_(Instrumental).wav")
                    
                    if os.path.exists(expected_as_vocals):
                        shutil.copy2(expected_as_vocals, acapella_out_path)
                    else: print(f"Warning: Audio-Separator vocals stem not found for {filename} at {expected_as_vocals}", flush=True)
                    
                    if os.path.exists(expected_as_instrumental):
                        shutil.copy2(expected_as_instrumental, instrumental_out_path)
                    else: print(f"Warning: Audio-Separator instrumental stem not found for {filename} at {expected_as_instrumental}", flush=True)
    else:
        print("\nSkipping PHASE 1b: Source Separation", flush=True)

    if not parsed_args_for_main.skip_forced_alignment:
        print("\nPHASE 1c: Forced Alignment (whisper-timestamped)", flush=True)
        os.makedirs(parsed_args_for_main.alignments_json_dir, exist_ok=True)
        
        wt_check_cmd = ["whisper_timestamped", "--help"] 
        print(f"DEBUG: Checking whisper_timestamped availability with: {' '.join(wt_check_cmd)}", flush=True)
        if not run_command(wt_check_cmd, "whisper_timestamped check", suppress_output=False, capture_stderr_on_error=True):
            print("ERROR: whisper_timestamped command check failed. Halting.", flush=True)
            return 

        acapella_files_for_alignment = [{"filename": f} for f in os.listdir(parsed_args_for_main.acapellas_dir) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]
        print(f"Found {len(acapella_files_for_alignment)} acapellas in {parsed_args_for_main.acapellas_dir} for alignment.", flush=True)

        align_worker_partial = partial(align_song_worker, 
                                       base_acapellas_dir=parsed_args_for_main.acapellas_dir,
                                       output_json_dir=parsed_args_for_main.alignments_json_dir,
                                       whisper_model_arg=parsed_args_for_main.whisper_model,
                                       whisper_language_arg=parsed_args_for_main.whisper_language,
                                       force_rerun_arg=parsed_args_for_main.force_rerun_alignment)
        
        num_align_workers = min(parsed_args_for_main.num_workers_alignment, os.cpu_count() or 1)
        num_align_workers = max(1, num_align_workers) # Ensure at least 1 worker

        aligned_count = 0
        failed_count = 0
        skipped_count = 0

        if num_align_workers > 1 and len(acapella_files_for_alignment) > 1 :
            print(f"Starting parallel alignment with {num_align_workers} workers...", flush=True)
            with multiprocessing.Pool(processes=num_align_workers) as pool:
                for result in tqdm(pool.imap_unordered(align_worker_partial, acapella_files_for_alignment), total=len(acapella_files_for_alignment), desc="Aligning Acapellas"):
                    if result["status"] == "aligned": aligned_count += 1
                    elif result["status"] == "skipped_exists": skipped_count +=1
                    else: failed_count += 1
        else:
            print("Starting serial alignment (1 worker or 1 file)...", flush=True)
            for acapella_info in tqdm(acapella_files_for_alignment, desc="Aligning Acapellas Serially"):
                result = align_worker_partial(acapella_info)
                if result["status"] == "aligned": aligned_count += 1
                elif result["status"] == "skipped_exists": skipped_count +=1
                else: failed_count += 1
        
        print(f"Alignment finished. Aligned: {aligned_count}, Failed: {failed_count}, Skipped (exists): {skipped_count}", flush=True)

    else:
        print("\nSkipping PHASE 1c: Forced Alignment", flush=True)


    print("\nDEBUG: run_full_data_pipeline.py - Entering PHASE 2 & 3 block", flush=True)
    print("PHASE 2 & 3: Beat Feature and Flow Data Extraction", flush=True)
    preprocess_script_args = [
        "--instrumentals_dir", parsed_args_for_main.instrumentals_dir,
        "--alignments_dir", parsed_args_for_main.alignments_json_dir,
        "--processed_output_dir", parsed_args_for_main.preprocessed_output_dir,
        "--separator_tool_used_for_stems", parsed_args_for_main.separator_tool, 
        "--stem_provider_model_name", parsed_args_for_main.demucs_model if parsed_args_for_main.separator_tool == "demucs" else parsed_args_for_main.audio_separator_model_filename, 
        "--pre_separated_stems_root_dir", parsed_args_for_main.demucs_stem_output_base_dir if parsed_args_for_main.separator_tool == "demucs" else parsed_args_for_main.audio_separator_stem_output_base_dir,
        "--num_workers", str(parsed_args_for_main.num_workers_feature_extraction)
    ]
    if parsed_args_for_main.force_reprocess_features: preprocess_script_args.append("--force_reprocess")
    
    print(f"DEBUG: run_full_data_pipeline.py - About to call run_python_script for preprocess_dataset.py with args: {preprocess_script_args}", flush=True)
    if not run_python_script("preprocess_dataset.py", "Preprocess Dataset (Features & Flow)", script_args=preprocess_script_args):
        print("DEBUG: run_full_data_pipeline.py - run_python_script for preprocess_dataset.py returned False. Exiting.", flush=True)
        return 
    print("DEBUG: run_full_data_pipeline.py - run_python_script for preprocess_dataset.py finished.", flush=True)


    print("\nDEBUG: run_full_data_pipeline.py - Entering PHASE 4a block", flush=True)
    print("PHASE 4a: Tokenization for Lite Model", flush=True)
    if not run_python_script("05a_tokenize_data_lite.py", "Tokenize Lite Data"):
        print("DEBUG: run_full_data_pipeline.py - run_python_script for 05a_tokenize_data_lite.py returned False. Exiting.", flush=True)
        return
    print("DEBUG: run_full_data_pipeline.py - run_python_script for 05a_tokenize_data_lite.py finished.", flush=True)


    print("\nDEBUG: run_full_data_pipeline.py - Entering PHASE 4b block", flush=True)
    print("PHASE 4b: Tokenization for Full Model", flush=True)
    if not run_python_script("05b_tokenize_data_full.py", "Tokenize Full Data"):
        print("DEBUG: run_full_data_pipeline.py - run_python_script for 05b_tokenize_data_full.py returned False. Exiting.", flush=True)
        return
    print("DEBUG: run_full_data_pipeline.py - run_python_script for 05b_tokenize_data_full.py finished.", flush=True)

    print("\n====== Full Data Preparation Pipeline Potentially Complete ======", flush=True)

parsed_args_for_main = None 
if __name__ == "__main__":
    multiprocessing.freeze_support() # Important for Windows
    parser = argparse.ArgumentParser(description="Run the full BeefAI data preparation pipeline.")
    
    # Phase skipping flags
    parser.add_argument("--skip_venv_warning_prompt", action="store_true", help="Skip venv warning prompt if not in expected .venv.")
    parser.add_argument("--skip_setup_prompt", action="store_true", help="Skip the initial setup confirmation prompt.")
    parser.add_argument("--skip_data_acquisition", action="store_true", help="Skip the conceptual data acquisition phase (Phase 1).")
    parser.add_argument("--skip_source_separation", action="store_true", help="Skip source separation (Phase 1b).")
    parser.add_argument("--skip_forced_alignment", action="store_true", help="Skip forced alignment (Phase 1c).")

    # Force flags
    parser.add_argument("--force_rerun_separation", action="store_true", help="Force rerun source separation even if outputs exist.")
    parser.add_argument("--force_rerun_alignment", action="store_true", help="Force rerun forced alignment even if outputs exist.")
    parser.add_argument("--force_reprocess_features", action="store_true", help="Force scripts/preprocess_dataset.py to ignore its caches.")

    # Directory paths
    parser.add_argument("--raw_songs_dir", default=DEFAULT_RAW_SONGS_DIR, help=f"Directory for raw song audio files. Default: {DEFAULT_RAW_SONGS_DIR}")
    parser.add_argument("--lyrics_dir", default=DEFAULT_LYRICS_DIR, help=f"Directory for raw lyric text files. Default: {DEFAULT_LYRICS_DIR}")
    parser.add_argument("--instrumentals_dir", default=DEFAULT_INSTRUMENTALS_DIR, help=f"Output directory for instrumental tracks. Default: {DEFAULT_INSTRUMENTALS_DIR}")
    parser.add_argument("--acapellas_dir", default=DEFAULT_ACAPELLAS_DIR, help=f"Output directory for acapella tracks. Default: {DEFAULT_ACAPELLAS_DIR}")
    parser.add_argument("--alignments_json_dir", default=DEFAULT_ALIGNMENTS_JSON_DIR, help=f"Output directory for alignment JSON files. Default: {DEFAULT_ALIGNMENTS_JSON_DIR}")
    parser.add_argument("--preprocessed_output_dir", default=DEFAULT_PREPROCESSED_OUTPUT_DIR, help=f"Output directory for data processed by preprocess_dataset.py. Default: {DEFAULT_PREPROCESSED_OUTPUT_DIR}")

    # Source separation tool configuration
    parser.add_argument("--separator_tool", choices=["demucs", "audio_separator"], default="audio_separator", help=f"Source separation tool to use. Default: audio_separator")
    parser.add_argument("--demucs_stem_output_base_dir", default=DEFAULT_DEMUCS_STEM_OUTPUT_BASE_DIR, help=f"Base output directory for Demucs stems. Default: {DEFAULT_DEMUCS_STEM_OUTPUT_BASE_DIR}")
    parser.add_argument("--demucs_model", default=DEFAULT_DEMUCS_MODEL, help=f"Demucs model name (e.g., htdemucs_ft). Default: {DEFAULT_DEMUCS_MODEL}")
    parser.add_argument("--audio_separator_stem_output_base_dir", default=DEFAULT_AUDIO_SEPARATOR_STEM_OUTPUT_BASE_DIR, help=f"Base output directory for audio-separator stems. Default: {DEFAULT_AUDIO_SEPARATOR_STEM_OUTPUT_BASE_DIR}")
    parser.add_argument("--audio_separator_model_filename", default=DEFAULT_AUDIO_SEPARATOR_MODEL, help=f"Filename of the audio-separator ONNX model. Default: {DEFAULT_AUDIO_SEPARATOR_MODEL}")
    parser.add_argument("--use_autocast_for_separator", action="store_true", help="Enable autocast (mixed precision) for audio-separator if using CUDA.")
    
    # Whisper-timestamped configuration
    parser.add_argument("--whisper_model", default="small", help=f"Whisper model size for alignment (e.g., tiny, base, small, medium, large). Default: small")
    parser.add_argument("--whisper_language", default="en", help="Language for Whisper alignment (e.g., en, es). Default: en")

    # Concurrency
    default_cpu_half = max(1, (os.cpu_count() or 1) // 2)
    parser.add_argument("--num_workers_alignment", type=int, default=max(1, (os.cpu_count() or 1) // 4), help=f"Number of workers for parallel alignment. Default: quarter of CPU cores, min 1.")
    parser.add_argument("--num_workers_feature_extraction", type=int, default=default_cpu_half, help=f"Number of workers for preprocess_dataset.py. Default: half of CPU cores, min 1.")
    
    args = parser.parse_args()
    parsed_args_for_main = args 

    print(f"DEBUG: run_full_data_pipeline.py - Parsed args: {args}", flush=True)

    if not args.skip_source_separation and not os.path.isdir(args.raw_songs_dir):
        print(f"ERROR: Raw songs dir '{args.raw_songs_dir}' not found, but source separation enabled.", flush=True)
        sys.exit(1)
    if not args.skip_forced_alignment and not os.path.isdir(args.acapellas_dir):
        # Create acapellas_dir if it's missing but separation is supposed to run, as separation creates it.
        # Only error if alignment is *enabled* and separation is *skipped* but dir is missing.
        if args.skip_source_separation:
            print(f"ERROR: Acapellas dir '{args.acapellas_dir}' not found, forced alignment is enabled, and source separation is skipped. Cannot proceed.", flush=True)
            sys.exit(1)
        else: # Separation is enabled, so it *should* create this dir. Make sure it exists for later steps.
            os.makedirs(args.acapellas_dir, exist_ok=True)

    
    print("DEBUG: run_full_data_pipeline.py - Calling main function.", flush=True)
    main(args)
    print("DEBUG: run_full_data_pipeline.py - Main function returned.", flush=True)