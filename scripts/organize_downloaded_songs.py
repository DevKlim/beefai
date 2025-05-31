import os
import shutil
import argparse

# --- Configuration ---
DEFAULT_SOURCE_DIR = "downloaded_songs"
DEFAULT_DATASET_DIR = "data" # Base directory for your organized dataset

TARGET_RAW_SONGS_SUBDIR = "raw_songs_full" # For full mix MP3s/WAVs
TARGET_LYRICS_SUBDIR = "lyrics"

def organize_files(source_dir: str, target_dataset_dir: str):
    """
    Moves .mp3 and .txt files from source_dir to structured subdirectories
    within target_dataset_dir.
    """
    print(f"Starting organization of files from '{source_dir}' to '{target_dataset_dir}'...")

    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' not found. Please check the path.")
        return

    # Define full paths for target subdirectories
    target_raw_songs_dir = os.path.join(target_dataset_dir, TARGET_RAW_SONGS_SUBDIR)
    target_lyrics_dir = os.path.join(target_dataset_dir, TARGET_LYRICS_SUBDIR)

    # Create target directories if they don't exist
    try:
        os.makedirs(target_raw_songs_dir, exist_ok=True)
        print(f"Ensured target directory for raw songs exists: '{target_raw_songs_dir}'")
        os.makedirs(target_lyrics_dir, exist_ok=True)
        print(f"Ensured target directory for lyrics exists: '{target_lyrics_dir}'")
    except OSError as e:
        print(f"Error creating target directories: {e}")
        return

    moved_audio_count = 0
    moved_lyrics_count = 0
    skipped_count = 0
    error_count = 0
    audio_extensions = ('.mp3', '.wav', '.flac', '.m4a')

    # Iterate over files in the source directory
    for filename in os.listdir(source_dir):
        source_filepath = os.path.join(source_dir, filename)

        if not os.path.isfile(source_filepath):
            continue # Skip directories

        destination_filepath = None
        file_type = None

        if filename.lower().endswith(audio_extensions):
            destination_filepath = os.path.join(target_raw_songs_dir, filename)
            file_type = "Audio"
        elif filename.lower().endswith(".txt"):
            destination_filepath = os.path.join(target_lyrics_dir, filename)
            file_type = "Lyrics"
        else:
            print(f"  Skipping '{filename}': Not a recognized audio or TXT file.")
            continue

        if destination_filepath:
            if os.path.exists(destination_filepath):
                print(f"  Skipping '{filename}': File already exists in destination '{destination_filepath}'.")
                skipped_count += 1
            else:
                try:
                    shutil.move(source_filepath, destination_filepath)
                    print(f"  Moved '{filename}' to '{destination_filepath}'.")
                    if file_type == "Audio":
                        moved_audio_count += 1
                    elif file_type == "Lyrics":
                        moved_lyrics_count += 1
                except Exception as e:
                    print(f"  Error moving '{filename}': {e}")
                    error_count += 1
    
    print("\n--- Organization Summary ---")
    print(f"Audio files moved: {moved_audio_count}")
    print(f"Lyrics files moved: {moved_lyrics_count}")
    print(f"Files skipped (already exist in destination): {skipped_count}")
    print(f"Errors encountered: {error_count}")
    print("Organization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize audio and lyric files from a source directory into a structured dataset directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the .mp3, .wav, .txt etc. files to organize."
    )
    parser.add_argument(
        "--target_dataset_dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help="Base directory where the structured subdirectories (e.g., 'raw_songs_full', 'lyrics') will be created/used."
    )
    
    args = parser.parse_args()
    
    organize_files(args.source_dir, args.target_dataset_dir)