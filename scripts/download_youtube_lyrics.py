import os
import sys
import argparse
import re
import time
import json # For yt-dlp JSON output
import csv # Added for CSV reporting
from yt_dlp import YoutubeDL
import requests 
from bs4 import BeautifulSoup 
import lyricsgenius 
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
REQUEST_TIMEOUT = 15
DOWNLOAD_DELAY = 1 # Adjusted delay, can be configured
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

# Punctuation pattern from clean_lyric_text.py
CHARS_TO_DELETE_PATTERN = r'[.,!?"():;“”‘’«»–—/[\]{}<>*&^%$#@+=~`|_]'


# --- Helper Functions ---

def sanitize_filename(name: str) -> str:
    if not name:
        name = "unknown_title"
    # Remove invalid characters for filenames
    name = re.sub(r'[\\/*?:"<>|]',"_", name)
    # Replace newlines and carriage returns with underscore
    name = name.replace("\n", "_").replace("\r", "_")
    # Replace multiple spaces/underscores with a single underscore
    name = re.sub(r'\s+', '_', name.strip())
    name = re.sub(r'_+', '_', name) # Consolidate multiple underscores
    return name[:150] # Truncate if too long

def apply_final_lyric_cleaning(text_content: str) -> str:
    """
    Cleans lyric text by removing specified punctuation,
    normalizing spaces within lines, preserving newlines,
    and removing lines that become empty after cleaning.
    Assumes Genius-specific headers/footers are already handled.
    """
    if not text_content:
        return ""

    lines = text_content.splitlines()
    cleaned_lines = []

    for line in lines:
        # 1. Remove specified punctuation from the line
        line_no_punct = re.sub(CHARS_TO_DELETE_PATTERN, '', line)
        
        # 2. Normalize whitespace (multiple spaces/tabs to a single space) WITHIN the line
        line_normalized_space = re.sub(r'[ \t]+', ' ', line_no_punct).strip()
        
        # 3. Only add the line if it's not empty after cleaning
        if line_normalized_space:
            cleaned_lines.append(line_normalized_space)
            
    # Join the cleaned lines back with newlines
    final_text = "\n".join(cleaned_lines)
    return final_text.strip() # Ensure no leading/trailing whitespace on the whole block

def fetch_lyrics_with_genius(song_title: str, artist_name: Optional[str] = None) -> Optional[str]:
    if not GENIUS_ACCESS_TOKEN or GENIUS_ACCESS_TOKEN == "YOUR_GENIUS_CLIENT_ACCESS_TOKEN_HERE" or GENIUS_ACCESS_TOKEN == "":
        print("  GENIUS_ACCESS_TOKEN not set or is placeholder. LyricsGenius fetch will be skipped.")
        return None
    
    try:
        genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, 
                                     verbose=False, 
                                     remove_section_headers=True, 
                                     skip_non_songs=True,
                                     timeout=REQUEST_TIMEOUT,
                                     retries=2)
    except Exception as e:
        print(f"  Failed to initialize LyricsGenius object: {e}")
        return None

    search_term_display = song_title
    if artist_name:
        search_term_display = f"{song_title} by {artist_name}"
        
    print(f"  Searching Genius for: '{search_term_display}'")
    song_result = None
    try:
        if artist_name and song_title:
            song_result = genius.search_song(song_title, artist_name)
        
        if not song_result and song_title : 
            if artist_name: 
                 print(f"  No direct match for '{song_title}' by '{artist_name}'. Trying title only: '{song_title}'")
            song_result = genius.search_song(song_title) 

        if song_result:
            print(f"  Lyrics found via lyricsgenius for '{song_result.title}' (Artist: {song_result.artist})")
            lyrics_text = song_result.lyrics.strip()
            
            lines = lyrics_text.split('\n')
            
            # Pattern for the Genius metadata header line (often the first line)
            # Catches lines like "123 ContributorsTranslationsENDEFR Lyrics" or "XXXContributorsXXX Lyrics"
            metadata_header_pattern = r"^\d*\s*Contributor(?:s)?.*Lyrics\s*$" 
            # A more specific pattern for just "Lyrics" possibly with pyong count
            lyrics_only_header_pattern = r"^(?:\d+Pyong)?\s*Lyrics\s*$"


            if lines:
                first_line_stripped = lines[0].strip()
                # Remove the first line if it's the Genius metadata header
                if re.match(metadata_header_pattern, first_line_stripped, re.IGNORECASE):
                    print(f"  Removing detected Genius metadata header (first line): '{first_line_stripped}'")
                    lines.pop(0)
                # If the first line wasn't the full metadata header, check if it's just "Lyrics" (potentially with pyong)
                # and there's more content afterwards.
                elif lines and re.match(lyrics_only_header_pattern, first_line_stripped, re.IGNORECASE):
                    if len(lines) > 1 or (len(lines) == 1 and lines[0].strip() != first_line_stripped): # ensure it's not the *only* content
                        print(f"  Removing detected 'Lyrics' type header (first line): '{first_line_stripped}'")
                        lines.pop(0)
            
            lyrics_text = "\n".join(lines).strip()
            
            # lyricsgenius.Genius(remove_section_headers=True) handles most "[Verse]", "[Chorus]"
            # Additional regex to clean up any remaining section headers if they are at the very start of a line.
            lyrics_text = re.sub(r"^\s*\[[^\]]+\]\s*\n?", "", lyrics_text, flags=re.MULTILINE).strip()
            
            # Remove common trailing footers
            lyrics_text = re.sub(r"\d*EmbedShare URLCopyEmbedCopy$", "", lyrics_text, flags=re.MULTILINE | re.IGNORECASE).strip()
            lyrics_text = re.sub(r"\d*Embed$", "", lyrics_text, flags=re.IGNORECASE).strip() 
            lyrics_text = re.sub(r"You might also like.*?(\n|$)", "", lyrics_text, flags=re.IGNORECASE | re.DOTALL).strip() # Original DOTALL was broad
            lyrics_text = re.sub(r"\s*See .*? LiveGet tickets.*(?:\n|$)", "", lyrics_text, flags=re.IGNORECASE | re.DOTALL).strip()


            # Apply final punctuation cleaning and line formatting
            fully_cleaned_lyrics = apply_final_lyric_cleaning(lyrics_text)
            
            return fully_cleaned_lyrics if fully_cleaned_lyrics else None # Return None if cleaning resulted in empty string
        else:
            print(f"  No song found on Genius for '{search_term_display}'.")
            return None
    except requests.exceptions.Timeout:
        print(f"  Timeout fetching lyrics from Genius for '{search_term_display}'.")
        return None
    except Exception as e: # Catching general exceptions from lyricsgenius or processing
        print(f"  Error fetching/processing lyrics from Genius for '{search_term_display}': {type(e).__name__} - {e}")
        return None

def process_video_entry(video_info: Dict[str, Any], output_dir: str, skip_existing: bool) -> Dict[str, Any]:
    original_title = video_info.get('title', 'Unknown_Title')
    video_url = video_info.get('webpage_url') or video_info.get('original_url') or video_info.get('url')
    
    base_filename_sanitized = sanitize_filename(original_title)
    
    report_entry = {
        'original_title': original_title,
        'sanitized_filename': base_filename_sanitized,
        'mp3_downloaded': False,
        'lyrics_found_and_saved': False,
        'mp3_path': os.path.join(output_dir, f"{base_filename_sanitized}.mp3"),
        'lyrics_path': os.path.join(output_dir, f"{base_filename_sanitized}.txt")
    }

    if not video_url:
        print(f"  Could not determine video URL for '{original_title}'. Skipping entry.")
        return report_entry

    mp3_final_filepath = report_entry['mp3_path']
    txt_filepath = report_entry['lyrics_path']

    print(f"\nProcessing: {original_title} (URL: {video_url})")
    print(f"  Sanitized base: {base_filename_sanitized}")
    print(f"  MP3 target: {mp3_final_filepath}")
    print(f"  TXT target: {txt_filepath}")

    # --- MP3 Download ---
    mp3_exists_and_valid = os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) > 1000

    if skip_existing and mp3_exists_and_valid:
        print(f"  MP3 already exists and is valid: {mp3_final_filepath}. Skipping download.")
        report_entry['mp3_downloaded'] = True
    # If it exists but overwrite is allowed, yt-dlp will handle it. We proceed to download.
    
    if not report_entry['mp3_downloaded']: # If not skipped due to existing
        output_template_for_yt_dlp = os.path.join(output_dir, base_filename_sanitized)
        ydl_opts_single = {
            'format': 'bestaudio/best',
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'outtmpl': output_template_for_yt_dlp + '.%(ext)s',
            'quiet': False, 'ignoreerrors': True, 'restrictfilenames': False,
            'overwrites': not skip_existing, 
            'nocheckcertificate': True, 'keepvideo': False,
        }
        try:
            with YoutubeDL(ydl_opts_single) as ydl_single:
                error_code = ydl_single.download([video_url])
                if os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) > 1000:
                    print(f"  Successfully downloaded/updated audio to: {mp3_final_filepath}")
                    report_entry['mp3_downloaded'] = True
                else:
                    print(f"  yt-dlp download process finished. Final MP3 '{mp3_final_filepath}' not found or too small (Error code: {error_code}).")
                    if os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) <= 1000:
                        try: os.remove(mp3_final_filepath); print(f"  Removed small/invalid file: {mp3_final_filepath}")
                        except OSError: pass # Ignore if removal fails
        except Exception as e:
            print(f"  Exception during audio download for {video_url}: {e}")

    # --- Lyrics Fetching and Saving ---
    if report_entry['mp3_downloaded']:
        lyrics_exist_and_valid = os.path.exists(txt_filepath) and os.path.getsize(txt_filepath) > 1

        if skip_existing and lyrics_exist_and_valid:
            print(f"  Lyrics file already exists and is valid: {txt_filepath}. Skipping fetch.")
            report_entry['lyrics_found_and_saved'] = True
        else:
            artist = video_info.get('artist') or video_info.get('creator') or video_info.get('uploader') or video_info.get('channel')
            cleaned_title_for_lyrics = original_title
            common_video_terms = [
                r"official video", r"music video", r"lyrics video", r"lyric video", r"official visualizer",
                r"official audio", r"audio", r"hd", r"hq", r"4k", r"\(visualizer\)",
                r"\((?:official|music|lyric|audio|video|visualizer|live|explicit|clean|extended|remix|acoustic|album version|radio edit)[^)]*\)",
                r"\[(?:official|music|lyric|audio|video|visualizer|live|explicit|clean|extended|remix|acoustic|album version|radio edit)[^\]]*\]",
                r"\s*-\s*Lyrics", r"ft\.", r"feat\.", r"prod\.", r"w/", r"&"
            ]
            for term in common_video_terms:
                cleaned_title_for_lyrics = re.sub(term, "", cleaned_title_for_lyrics, flags=re.IGNORECASE).strip()
            if artist:
                 escaped_artist = re.escape(artist)
                 cleaned_title_for_lyrics = re.sub(r"\s*-\s*" + escaped_artist + r"\s*$", "", cleaned_title_for_lyrics, flags=re.IGNORECASE).strip()
                 cleaned_title_for_lyrics = re.sub(r"^" + escaped_artist + r"\s*-\s*", "", cleaned_title_for_lyrics, flags=re.IGNORECASE).strip()
            cleaned_title_for_lyrics = re.sub(r'\s{2,}', ' ', cleaned_title_for_lyrics).strip()
            cleaned_title_for_lyrics = cleaned_title_for_lyrics.replace('_', ' ') # Replace underscores from sanitize with spaces for search

            lyrics_content = fetch_lyrics_with_genius(cleaned_title_for_lyrics, artist)
            
            if lyrics_content and lyrics_content.strip(): # Ensure content is not just whitespace
                try:
                    with open(txt_filepath, 'w', encoding='utf-8') as f:
                        f.write(lyrics_content)
                    print(f"  Lyrics saved to: {txt_filepath}")
                    report_entry['lyrics_found_and_saved'] = True
                except IOError as e:
                    print(f"  Error saving lyrics to {txt_filepath}: {e}")
            else:
                print(f"  No valid lyrics found or fetched for '{original_title}' (searched as: '{cleaned_title_for_lyrics}', artist: '{artist}'). No lyric file will be created/updated.")
                # If a previous (e.g. placeholder or outdated) .txt file exists, remove it.
                if os.path.exists(txt_filepath):
                    try:
                        print(f"  Removing existing lyrics file as no new lyrics found: {txt_filepath}")
                        os.remove(txt_filepath)
                    except OSError as e:
                        print(f"  Error removing existing lyrics file {txt_filepath}: {e}")
    else:
         print(f"  Skipping lyrics for '{original_title}' due to audio download failure or skip.")
    
    time.sleep(DOWNLOAD_DELAY)
    return report_entry

def main():
    parser = argparse.ArgumentParser(description="Download YouTube audio and fetch lyrics using LyricsGenius. Creates a CSV report.")
    parser.add_argument("url_file", help="Path to a text file containing YouTube video/playlist URLs (one per line).")
    parser.add_argument("-o", "--output_dir", default="downloaded_songs", help="Directory to save downloaded MP3s, lyrics TXTs, and the report. Default: %(default)s")
    parser.add_argument("--skip_existing", action="store_true", help="Skip downloading/fetching if MP3/TXT files already exist and are valid.")
    parser.add_argument("--genius_token", default=os.environ.get("GENIUS_ACCESS_TOKEN"), help="Genius API Client Access Token. Can also be set via GENIUS_ACCESS_TOKEN environment variable.")
    parser.add_argument("--report_file", default="download_report.csv", help="Filename for the CSV report of downloads. Will be saved in the output_dir. Default: %(default)s")

    args = parser.parse_args()

    global GENIUS_ACCESS_TOKEN 
    if args.genius_token:
        GENIUS_ACCESS_TOKEN = args.genius_token
    
    if not GENIUS_ACCESS_TOKEN or GENIUS_ACCESS_TOKEN == "YOUR_GENIUS_CLIENT_ACCESS_TOKEN_HERE" or GENIUS_ACCESS_TOKEN == "":
        print("WARNING: Genius API token is not set or is a placeholder. Lyrics fetching via LyricsGenius will be skipped.")
        print("You can provide it via the --genius_token argument, a GENIUS_ACCESS_TOKEN environment variable, or a .env file.")

    if not os.path.exists(args.url_file):
        print(f"Error: URL file not found at '{args.url_file}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    report_csv_path = os.path.join(args.output_dir, args.report_file)
    download_records: List[Dict[str, Any]] = []

    with open(args.url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not urls:
        print("No URLs found in the input file.")
        sys.exit(0)

    print(f"Found {len(urls)} URLs/Playlists to process. Report will be saved to: {report_csv_path}")

    ydl_opts_info = {
        'quiet': True, 'ignoreerrors': True, 'extract_flat': False, 
        'skip_download': True, 'nocheckcertificate': True,
        'playlist_items': '1-10000' # Generous playlist item limit
    }

    with YoutubeDL(ydl_opts_info) as ydl_info_extractor:
        for i, url_or_playlist in enumerate(urls):
            print(f"\n--- Processing source {i+1}/{len(urls)}: {url_or_playlist} ---")
            try:
                result_info_data = ydl_info_extractor.extract_info(url_or_playlist, download=False)
                if not result_info_data:
                    print(f"Could not fetch info for URL: {url_or_playlist}. Skipping.")
                    download_records.append({
                        'original_title': f"Failed_Info_Fetch_{url_or_playlist[:50]}",
                        'sanitized_filename': sanitize_filename(f"Failed_Info_Fetch_{url_or_playlist[:50]}"),
                        'mp3_downloaded': False, 'lyrics_found_and_saved': False,
                        'mp3_path': "", 'lyrics_path': "Info extraction failed"
                    })
                    continue

                actual_videos: List[Dict[str, Any]] = []
                if result_info_data.get('_type') == 'playlist' and 'entries' in result_info_data:
                    actual_videos = [entry for entry in result_info_data.get('entries', []) if entry and isinstance(entry, dict)]
                elif isinstance(result_info_data, dict) : # Single video
                    actual_videos = [result_info_data]
                
                if not actual_videos:
                    print(f"  No valid video items found for URL/playlist: {url_or_playlist}")
                    download_records.append({
                        'original_title': f"No_Videos_In_Source_{url_or_playlist[:50]}",
                        'sanitized_filename': sanitize_filename(f"No_Videos_In_Source_{url_or_playlist[:50]}"),
                        'mp3_downloaded': False, 'lyrics_found_and_saved': False,
                        'mp3_path': "", 'lyrics_path': "No video entries found"
                    })
                    continue
                
                print(f"  Found {len(actual_videos)} video(s) to process from this source.")
                for vid_idx, video_data in enumerate(actual_videos):
                    print(f"  Processing video {vid_idx+1}/{len(actual_videos)} from source '{url_or_playlist}'...")
                    entry_report = process_video_entry(video_data, args.output_dir, args.skip_existing)
                    download_records.append(entry_report)

            except Exception as e:
                print(f"  Major error processing URL {url_or_playlist}: {type(e).__name__} - {e}")
                import traceback
                traceback.print_exc()
                download_records.append({
                    'original_title': f"Error_Processing_URL_{url_or_playlist[:50]}",
                    'sanitized_filename': sanitize_filename(f"Error_Processing_URL_{url_or_playlist[:50]}"),
                    'mp3_downloaded': False, 'lyrics_found_and_saved': False,
                    'mp3_path': "", 'lyrics_path': str(e)
                })

    # Write CSV report
    print(f"\n--- Writing download report to {report_csv_path} ---")
    if download_records:
        fieldnames = ['original_title', 'sanitized_filename', 'mp3_downloaded', 'lyrics_found_and_saved', 'mp3_path', 'lyrics_path']
        try:
            with open(report_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for record in download_records: # Ensure all keys are present
                    writer.writerow({k: record.get(k, "") for k in fieldnames})
            print(f"Successfully wrote {len(download_records)} records to {report_csv_path}")
        except IOError as e:
            print(f"Error writing CSV report: {e}")
    else:
        print("No download records to write to CSV.")

    print("\n--- All processing finished. ---")

if __name__ == "__main__":
    main()