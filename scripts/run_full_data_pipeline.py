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
# (Keep the venv check as it was)
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
        # ... (rest of warning message, ensuring flush=True for all prints in this block)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        skip_prompt = False
        if 'parsed_args_for_main' in globals() and hasattr(parsed_args_for_main, 'skip_venv_warning_prompt'):
             skip_prompt = parsed_args_for_main.skip_venv_warning_prompt
        if not skip_prompt:
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
        traceback.print_exc() # Print full traceback for unexpected errors
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
            print(f"ERROR: Alignment command succeeded for {original_acapella_filename}, but expected output JSON not found.", flush=True)
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
    success = run_command(command, script_name, suppress_output=False) # Show output for python scripts
    print(f"DEBUG: run_python_script: {script_name} returned: {success}", flush=True)
    return success


def main(args_main_func): 
    global parsed_args_for_main 
    parsed_args_for_main = args_main_func

    print("====== Starting Full Data Preparation Pipeline ======", flush=True)

    print("\nPHASE 0: Setup", flush=True)
    # ... (rest of Phase 0 with flush=True on its prints)
    if not parsed_args_for_main.skip_setup_prompt:
        print("Press Enter to continue if setup is complete (auto-proceeds in 5s)...", flush=True)
        try: time.sleep(5); print("Proceeding automatically after timeout.", flush=True)
        except KeyboardInterrupt: print("Exiting due to user interruption.", flush=True); sys.exit(1)

    if not parsed_args_for_main.skip_data_acquisition:
        print("\nPHASE 1: Data Acquisition & Organization", flush=True)
    else: print("\nSkipping PHASE 1: Data Acquisition & Organization", flush=True)

    if not parsed_args_for_main.skip_source_separation:
        print(f"\nPHASE 1b: Source Separation (Using: {parsed_args_for_main.separator_tool})", flush=True)
        # ... (separation logic with flush=True on its prints) ...
    else:
        print("\nSkipping PHASE 1b: Source Separation", flush=True)

    if not parsed_args_for_main.skip_forced_alignment:
        print("\nPHASE 1c: Forced Alignment (whisper-timestamped)", flush=True)
        # ... (alignment logic with flush=True on its prints) ...
        wt_check_cmd = ["whisper_timestamped", "--help"] 
        print(f"DEBUG: Checking whisper_timestamped availability with: {' '.join(wt_check_cmd)}", flush=True)
        if not run_command(wt_check_cmd, "whisper_timestamped check", suppress_output=False, capture_stderr_on_error=True):
            print("ERROR: whisper_timestamped command check failed. Halting.", flush=True)
            return 
        # ... rest of alignment logic ...
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
    parser = argparse.ArgumentParser(description="Run the full BeefAI data preparation pipeline.")
    # (Keep all argparse definitions as they were)
    parser.add_argument("--skip_venv_warning_prompt", action="store_true", help="Skip venv warning prompt.")
    parser.add_argument("--skip_setup_prompt", action="store_true", help="Skip setup prompt.")
    parser.add_argument("--skip_data_acquisition", action="store_true", help="Skip data acquisition.")
    parser.add_argument("--skip_source_separation", action="store_true", help="Skip source separation.")
    parser.add_argument("--skip_forced_alignment", action="store_true", help="Skip forced alignment.")
    parser.add_argument("--force_rerun_separation", action="store_true", help="Force rerun separation.")
    parser.add_argument("--force_rerun_alignment", action="store_true", help="Force rerun alignment.")
    parser.add_argument("--force_reprocess_features", action="store_true", help="Force reprocess features.")
    parser.add_argument("--raw_songs_dir", default=DEFAULT_RAW_SONGS_DIR, help="Raw songs directory.")
    parser.add_argument("--lyrics_dir", default=DEFAULT_LYRICS_DIR, help="Lyrics directory.")
    parser.add_argument("--instrumentals_dir", default=DEFAULT_INSTRUMENTALS_DIR, help="Instrumentals output dir.")
    parser.add_argument("--acapellas_dir", default=DEFAULT_ACAPELLAS_DIR, help="Acapellas output dir.")
    parser.add_argument("--alignments_json_dir", default=DEFAULT_ALIGNMENTS_JSON_DIR, help="Alignment JSONs output dir.")
    parser.add_argument("--preprocessed_output_dir", default=DEFAULT_PREPROCESSED_OUTPUT_DIR, help="Preprocessed data output dir.")
    parser.add_argument("--separator_tool", choices=["demucs", "audio_separator"], default="audio_separator", help="Source separation tool.") 
    parser.add_argument("--demucs_stem_output_base_dir", default=DEFAULT_DEMUCS_STEM_OUTPUT_BASE_DIR, help="Demucs stems base output dir.")
    parser.add_argument("--demucs_model", default=DEFAULT_DEMUCS_MODEL, help="Demucs model.")
    parser.add_argument("--audio_separator_stem_output_base_dir", default=DEFAULT_AUDIO_SEPARATOR_STEM_OUTPUT_BASE_DIR, help="Audio-separator stems base output dir.")
    parser.add_argument("--audio_separator_model_filename", default=DEFAULT_AUDIO_SEPARATOR_MODEL, help="Audio-separator model ONNX name.")
    parser.add_argument("--use_autocast_for_separator", action="store_true", help="Enable autocast for audio-separator.")
    parser.add_argument("--whisper_model", default="small", help="Whisper model for alignment.")
    parser.add_argument("--whisper_language", default="en", help="Language for Whisper alignment.")
    default_cpu_half = max(1, (os.cpu_count() or 1) // 2)
    parser.add_argument("--num_workers_alignment", type=int, default=max(1, (os.cpu_count() or 1) // 4), help="Num workers for alignment.")
    parser.add_argument("--num_workers_feature_extraction", type=int, default=default_cpu_half, help="Num workers for feature extraction.")

    args = parser.parse_args()
    parsed_args_for_main = args 

    # Ensure all top-level prints in this script also use flush=True
    print(f"DEBUG: run_full_data_pipeline.py - Parsed args: {args}", flush=True)

    if not args.skip_source_separation and not os.path.isdir(args.raw_songs_dir):
        print(f"ERROR: Raw songs dir '{args.raw_songs_dir}' not found, but source separation enabled.", flush=True)
        sys.exit(1)
    if not args.skip_forced_alignment and not os.path.isdir(args.acapellas_dir):
        print(f"ERROR: Acapellas dir '{args.acapellas_dir}' not found, but forced alignment enabled.", flush=True)
        sys.exit(1)
    
    print("DEBUG: run_full_data_pipeline.py - Calling main function.", flush=True)
    main(args)
    print("DEBUG: run_full_data_pipeline.py - Main function returned.", flush=True)