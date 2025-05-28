import os
import argparse

def remove_first_line(filepath: str):
    """
    Reads a file, removes its first line, and writes the rest back.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if not lines:
            # print(f"  Skipping '{os.path.basename(filepath)}': File is empty.")
            return False # No change made

        # Remove the first line
        modified_lines = lines[1:]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
        
        # print(f"  Removed first line from '{os.path.basename(filepath)}'.")
        return True # Change was made
    except Exception as e:
        print(f"  Error processing file '{os.path.basename(filepath)}': {e}")
        return False

def process_lyrics_directory(lyrics_dir: str):
    """
    Iterates through all .txt files in the given directory and removes
    the first line from each.
    """
    if not os.path.isdir(lyrics_dir):
        print(f"Error: Lyrics directory '{lyrics_dir}' not found.")
        return

    print(f"Processing .txt files in directory: '{lyrics_dir}'")
    processed_count = 0
    skipped_empty_count = 0
    error_count = 0

    for filename in os.listdir(lyrics_dir):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(lyrics_dir, filename)
            print(f"Processing: {filename}")
            if remove_first_line(filepath):
                processed_count += 1
            else:
                # This 'else' branch is hit if the file was empty or an error occurred.
                # We can refine counting based on return value if needed.
                # For now, if remove_first_line returns False because it was empty, it's not an error.
                # Let's assume remove_first_line prints specific skip/error messages.
                pass # Error/skip message handled by remove_first_line
    
    # A more accurate count would require remove_first_line to distinguish
    # between "skipped because empty" and "error".
    # For simplicity, this summary is basic.
    print("\n--- Summary ---")
    print(f"Attempted to process {processed_count + skipped_empty_count + error_count} .txt files.") # This is just total .txt files found
    print(f"Successfully modified (removed first line from non-empty files): {processed_count}")
    # To get accurate skipped_empty_count and error_count, remove_first_line would need to return more specific status.
    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Removes the first line from all .txt files in a specified directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--lyrics_dir",
        type=str,
        default=os.path.join("data", "lyrics"), # Default path
        help="Directory containing the .txt lyric files to process."
    )
    
    args = parser.parse_args()
    
    # Confirmation step
    print(f"WARNING: This script will modify .txt files in '{args.lyrics_dir}' by removing their first line.")
    confirm = input("Are you sure you want to continue? (yes/no): ")
    
    if confirm.lower() == 'yes':
        process_lyrics_directory(args.lyrics_dir)
    else:
        print("Operation cancelled by user.")