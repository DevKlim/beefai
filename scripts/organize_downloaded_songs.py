import os
import shutil
import argparse

# --- Configuration ---
DEFAULT_SOURCE_DIR = "downloaded_songs" # Directory where your .mp3 and .txt files currently are
DEFAULT_TARGET_BASE_DIR = "data" # The base directory for your organized dataset

TARGET_MP3_SUBDIR = "raw_songs_full" # Subdirectory for .mp3 files (original full tracks)
TARGET_TXT_SUBDIR = "lyrics"         # Subdirectory for .txt files

def organize_files(source_dir: str, target_base_dir: str):
    """
    Moves .mp3 and .txt files from source_dir to structured subdirectories
    within target_base_dir.
    """
    print(f"Starting organization of files from '{source_dir}' to '{target_base_dir}'...")

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found. Please check the path.")
        return

    # Define full paths for target subdirectories
    target_mp3_dir = os.path.join(target_base_dir, TARGET_MP3_SUBDIR)
    target_txt_dir = os.path.join(target_base_dir, TARGET_TXT_SUBDIR)

    # Create target directories if they don't exist
    try:
        os.makedirs(target_mp3_dir, exist_ok=True)
        print(f"Ensured target directory for MP3s exists: '{target_mp3_dir}'")
        os.makedirs(target_txt_dir, exist_ok=True)
        print(f"Ensured target directory for TXTs exists: '{target_txt_dir}'")
    except OSError as e:
        print(f"Error creating target directories: {e}")
        return

    moved_mp3_count = 0
    moved_txt_count = 0
    skipped_count = 0
    error_count = 0

    # Iterate over files in the source directory
    for filename in os.listdir(source_dir):
        source_filepath = os.path.join(source_dir, filename)

        if not os.path.isfile(source_filepath):
            continue # Skip directories

        destination_filepath = None
        file_type = None

        if filename.lower().endswith(".mp3"):
            destination_filepath = os.path.join(target_mp3_dir, filename)
            file_type = "MP3"
        elif filename.lower().endswith(".txt"):
            destination_filepath = os.path.join(target_txt_dir, filename)
            file_type = "TXT"
        else:
            print(f"  Skipping '{filename}': Not an MP3 or TXT file.")
            continue

        if destination_filepath:
            if os.path.exists(destination_filepath):
                print(f"  Skipping '{filename}': File already exists in destination '{destination_filepath}'.")
                skipped_count += 1
            else:
                try:
                    shutil.move(source_filepath, destination_filepath)
                    print(f"  Moved '{filename}' to '{destination_filepath}'.")
                    if file_type == "MP3":
                        moved_mp3_count += 1
                    elif file_type == "TXT":
                        moved_txt_count += 1
                except Exception as e:
                    print(f"  Error moving '{filename}': {e}")
                    error_count += 1
    
    print("\n--- Organization Summary ---")
    print(f"MP3 files moved: {moved_mp3_count}")
    print(f"TXT files moved: {moved_txt_count}")
    print(f"Files skipped (already exist in destination): {skipped_count}")
    print(f"Errors encountered: {error_count}")
    print("Organization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize .mp3 and .txt files from a source directory into a structured dataset directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the .mp3 and .txt files to organize."
    )
    parser.add_argument(
        "--target_base_dir",
        type=str,
        default=DEFAULT_TARGET_BASE_DIR,
        help="Base directory where the 'raw_songs_full' and 'lyrics' subdirectories will be created/used."
    )
    
    args = parser.parse_args()
    
    organize_files(args.source_dir, args.target_base_dir)