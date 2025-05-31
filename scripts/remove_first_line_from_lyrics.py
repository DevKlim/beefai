import os
import argparse
import re

def remove_problematic_first_line(filepath: str):
    """
    Reads a file, removes its first line if it matches known metadata patterns,
    and writes the rest back. Also removes a "Lyrics" header if present after the metadata line.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if not lines:
            return False # No change made, file empty

        original_first_line = lines[0].strip()
        modified = False

        # Pattern to match "Number Contributors", "Translations", language names
        # Example: "364 ContributorsTranslationsFrançaisTü"
        # More general: Starts with a number, then "Contributor(s)", then optionally "Translations"
        first_line_pattern = r"^\d+\s*Contributor(s)?(Translations)?([A-Za-zÀ-ÖØ-öø-ÿ\s\d\[\]\(\)]+)?[lL]yrics$"
        # Simplified pattern to catch common "Contributors...Lyrics" type lines
        metadata_pattern_genius = r"^\d+\s*Contributor(s)?.*?Lyrics$"


        if re.match(metadata_pattern_genius, original_first_line, re.IGNORECASE):
            print(f"  Removing detected metadata line: '{original_first_line}' from '{os.path.basename(filepath)}'")
            lines.pop(0)
            modified = True
            # Also remove the actual word "Lyrics" if it appears as the new first line
            if lines and lines[0].strip().lower() == "lyrics":
                print(f"  Removing subsequent 'Lyrics' header line from '{os.path.basename(filepath)}'")
                lines.pop(0)
        elif lines and original_first_line.lower() == "lyrics": # If only "Lyrics" is the first line
             print(f"  Removing 'Lyrics' header line from '{os.path.basename(filepath)}'")
             lines.pop(0)
             modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        return False # No problematic line found and removed

    except Exception as e:
        print(f"  Error processing file '{os.path.basename(filepath)}': {e}")
        return False

def process_lyrics_directory(lyrics_dir: str):
    if not os.path.isdir(lyrics_dir):
        print(f"Error: Lyrics directory '{lyrics_dir}' not found.")
        return

    print(f"Processing .txt files in directory: '{lyrics_dir}' to remove metadata lines.")
    modified_count = 0
    
    for filename in os.listdir(lyrics_dir):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(lyrics_dir, filename)
            if remove_problematic_first_line(filepath):
                modified_count += 1
    
    print("\n--- Lyric Cleaning Summary ---")
    print(f"Files modified (metadata line removed): {modified_count}")
    print("Lyric cleaning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Removes the typical first metadata line (e.g., 'Number Contributors...Lyrics') from .txt files in a specified directory, often found in lyricsgenius output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--lyrics_dir",
        type=str,
        default=os.path.join("data", "lyrics"), # Default path after organization
        help="Directory containing the .txt lyric files to process."
    )
    
    args = parser.parse_args()
    
    print(f"This script will attempt to remove known metadata lines from .txt files in '{args.lyrics_dir}'.")
    process_lyrics_directory(args.lyrics_dir)