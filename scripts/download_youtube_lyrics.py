import os
import sys
import argparse
import re
import time
import json # For yt-dlp JSON output
from yt_dlp import YoutubeDL
import requests # Keep for potential fallback or other sites if lyricsgenius fails broadly
from bs4 import BeautifulSoup # Keep for potential fallback
# from duckduckgo_search import DDGS # Can be removed if only using lyricsgenius
import lyricsgenius # Added
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
REQUEST_TIMEOUT = 15
DOWNLOAD_DELAY = 2
# LYRIC_SEARCH_RESULTS = 3 # Not used if only lyricsgenius
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

# --- Helper Functions ---

def sanitize_filename(name: str) -> str:
    if not name:
        name = "unknown_title"
    name = re.sub(r'[\\/*?:"<>|]',"_", name)
    name = name.replace("\n", "_").replace("\r", "_")
    name = re.sub(r'\s+', '_', name.strip())
    return name[:150]

# Keep fetch_url_content and bs4 extractors if you want a fallback to scraping.
# For now, primary focus is lyricsgenius. If GENIUS_ACCESS_TOKEN is not set, it will skip.

def fetch_lyrics_with_genius(song_title: str, artist_name: Optional[str] = None) -> Optional[str]:
    if not GENIUS_ACCESS_TOKEN or GENIUS_ACCESS_TOKEN == "YOUR_GENIUS_CLIENT_ACCESS_TOKEN_HERE" or GENIUS_ACCESS_TOKEN == "":
        print("  GENIUS_ACCESS_TOKEN not set or is placeholder. Skipping lyricsgenius fetch.")
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
            song_result = genius.search_song(song_title) # Search by title only if artist combo failed

        if song_result:
            print(f"  Lyrics found via lyricsgenius for '{song_result.title}' (Artist: {song_result.artist})")
            lyrics_text = song_result.lyrics.strip()
            
            # Additional cleaning for common Genius artifacts
            # Pattern to remove lines like "Number Contributors...Lyrics", e.g., "364ContributorsTranslationsFrançaisTüLyrics"
            # This also handles cases where "Lyrics" might be on the next line or part of the metadata line.
            # The main problem is the "Contributors...Lyrics" part.
            
            lines = lyrics_text.split('\n')
            cleaned_lines = []
            metadata_header_pattern = r"^\d+\s*Contributor(s)?.*?Lyrics$" # Catches "123 Contributors...Lyrics"
            lyrics_header_pattern = r"^\s*Lyrics\s*$" # Catches a line that is just "Lyrics"

            # Remove initial metadata/header lines
            temp_lines = list(lines) # Work on a copy
            
            if temp_lines:
                # Check first line for "Number Contributors...Lyrics"
                if re.match(metadata_header_pattern, temp_lines[0].strip(), re.IGNORECASE):
                    print(f"  Removing detected Genius metadata header: '{temp_lines[0].strip()}'")
                    temp_lines.pop(0)
                
                # Check (potentially new) first line for just "Lyrics"
                if temp_lines and re.match(lyrics_header_pattern, temp_lines[0].strip(), re.IGNORECASE):
                    print(f"  Removing detected 'Lyrics' header: '{temp_lines[0].strip()}'")
                    temp_lines.pop(0)
            
            lyrics_text = "\n".join(temp_lines).strip()

            # Remove [Verse], [Chorus] etc. if they are at the very start of a line
            lyrics_text = re.sub(r"^\[[^\]]+\]\s*\n?", "", lyrics_text, flags=re.MULTILINE)
            
            # Remove common trailing footers
            lyrics_text = re.sub(r"\d*EmbedShare URLCopyEmbedCopy$", "", lyrics_text, flags=re.MULTILINE | re.IGNORECASE).strip()
            lyrics_text = re.sub(r"\d*Embed$", "", lyrics_text, flags=re.IGNORECASE).strip() 
            lyrics_text = re.sub(r"You might also like.*?(\n|$)", "", lyrics_text, flags=re.IGNORECASE | re.DOTALL).strip()


            return lyrics_text
        else:
            print(f"  No song found on Genius for '{search_term_display}'.")
            return None
    except Exception as e:
        print(f"  Error fetching lyrics from Genius for '{search_term_display}': {type(e).__name__} - {e}")
        return None


def process_video_entry(video_info: Dict[str, Any], output_dir: str, skip_existing: bool):
    video_url = video_info.get('webpage_url') or video_info.get('original_url') or video_info.get('url')
    if not video_url:
        print("  Could not determine video URL from metadata. Skipping entry.")
        return

    original_title = video_info.get('title', 'Unknown_Title')
    # Sanitize filename from the original title to ensure consistency
    base_filename_sanitized = sanitize_filename(original_title) 
    
    # Define final paths using the sanitized base filename
    mp3_final_filepath = os.path.join(output_dir, f"{base_filename_sanitized}.mp3")
    txt_filepath = os.path.join(output_dir, f"{base_filename_sanitized}.txt")

    print(f"\nProcessing: {original_title} (URL: {video_url})")
    print(f"  Sanitized base: {base_filename_sanitized}")
    print(f"  MP3 target: {mp3_final_filepath}")
    print(f"  TXT target: {txt_filepath}")

    audio_downloaded_successfully = False
    mp3_exists_and_valid = os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) > 1000

    if skip_existing and mp3_exists_and_valid:
        print(f"  MP3 already exists and is valid: {mp3_final_filepath}. Skipping download.")
        audio_downloaded_successfully = True
    elif mp3_exists_and_valid: 
        print(f"  MP3 {mp3_final_filepath} already exists. Download behavior depends on 'overwrites' yt-dlp option.")
        audio_downloaded_successfully = True # Assume it's good if it exists, overwrite handles staleness
    
    if not audio_downloaded_successfully:
        # Use the sanitized base filename for the output template to avoid double extension issues.
        # yt-dlp will add its own .mp3 (or .webm then converted to .mp3)
        output_template_for_yt_dlp = os.path.join(output_dir, base_filename_sanitized) # NO extension here

        ydl_opts_single = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_template_for_yt_dlp + '.%(ext)s', # yt-dlp adds extension
            'quiet': False,
            'ignoreerrors': True,
            'restrictfilenames': False, # Let sanitize_filename handle it before ytdlp
            'overwrites': not skip_existing, # If skip_existing is false, then overwrite is true
            'nocheckcertificate': True,
            'keepvideo': False,
        }
        try:
            with YoutubeDL(ydl_opts_single) as ydl_single:
                error_code = ydl_single.download([video_url])
                
                # After download, check if the mp3_final_filepath exists and is valid
                if os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) > 1000:
                    print(f"  Successfully downloaded and converted audio to: {mp3_final_filepath}")
                    audio_downloaded_successfully = True
                else:
                    # This case should be less common if outtmpl is correct, but check just in case
                    # for temp files or if preferredcodec was not available and it saved as something else.
                    # For now, assume if mp3_final_filepath isn't there, it failed.
                    print(f"  yt-dlp download process finished (code: {error_code}). Final MP3 '{mp3_final_filepath}' not found or too small.")
                    if os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) <= 1000:
                        try: os.remove(mp3_final_filepath); print(f"  Removed small/invalid file: {mp3_final_filepath}")
                        except: pass
                            
        except Exception as e:
            print(f"  Exception during audio download for {video_url}: {e}")

    lyrics_exist_and_valid = os.path.exists(txt_filepath) and os.path.getsize(txt_filepath) > 10

    if skip_existing and lyrics_exist_and_valid:
        print(f"  Lyrics file already exists and is valid: {txt_filepath}. Skipping fetch.")
    elif audio_downloaded_successfully: 
        # Try to get artist info from metadata
        artist = video_info.get('artist') or video_info.get('creator') or video_info.get('uploader') or video_info.get('channel')
        
        # Clean title for lyric search (remove "official video", "(lyrics)", etc.)
        cleaned_title_for_lyrics = original_title
        common_video_terms = [
            r"official video", r"music video", r"lyrics video", r"lyric video",
            r"official audio", r"audio", r"hd", r"hq", r"4k", 
            r"\((?:official|music|lyric|audio|video|visualizer|live|explicit|clean|extended|remix|acoustic)[^)]*\)", # More specific parenthesized terms
            r"\[(?:official|music|lyric|audio|video|visualizer|live|explicit|clean|extended|remix|acoustic)[^\]]*\]", # More specific bracketed terms
            r"\s*-\s*Lyrics", # Remove "- Lyrics"
            r"ft\.", r"feat\.", r"prod\." # Common abbreviations
        ]
        for term in common_video_terms:
            cleaned_title_for_lyrics = re.sub(term, "", cleaned_title_for_lyrics, flags=re.IGNORECASE).strip()
        
        # If artist was found in title, try to remove it as well to get a cleaner song title
        if artist:
             # Escape artist name for regex, then try to remove " - ARTIST" or "ARTIST - " patterns
             escaped_artist = re.escape(artist)
             cleaned_title_for_lyrics = re.sub(r"\s*-\s*" + escaped_artist + r"\s*$", "", cleaned_title_for_lyrics, flags=re.IGNORECASE).strip()
             cleaned_title_for_lyrics = re.sub(r"^" + escaped_artist + r"\s*-\s*", "", cleaned_title_for_lyrics, flags=re.IGNORECASE).strip()

        cleaned_title_for_lyrics = re.sub(r'\s{2,}', ' ', cleaned_title_for_lyrics).strip() # Consolidate multiple spaces

        lyrics_content = fetch_lyrics_with_genius(cleaned_title_for_lyrics, artist)
        if lyrics_content:
            try:
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(lyrics_content)
                print(f"  Lyrics saved to: {txt_filepath}")
            except IOError as e:
                print(f"  Error saving lyrics to {txt_filepath}: {e}")
        else:
            print(f"  No lyrics found via lyricsgenius for '{original_title}' (searched as: '{cleaned_title_for_lyrics}', artist: '{artist}'). Creating placeholder TXT.")
            try:
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"[Lyrics for '{original_title}' (search term: '{cleaned_title_for_lyrics}', artist: '{artist}') not found via lyricsgenius.]")
            except IOError as e:
                print(f"  Error saving placeholder lyrics text to {txt_filepath}: {e}")
    elif not audio_downloaded_successfully:
         print(f"  Skipping lyrics for '{original_title}' due to audio download failure.")
    
    time.sleep(DOWNLOAD_DELAY)


def main():
    parser = argparse.ArgumentParser(description="Download YouTube audio and fetch lyrics using LyricsGenius.")
    parser.add_argument("url_file", help="Path to a text file containing YouTube video/playlist URLs (one per line).")
    parser.add_argument("-o", "--output_dir", default="downloaded_songs", help="Directory to save downloaded MP3s and lyrics TXTs.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip downloading/fetching if MP3/TXT files already exist and are valid.")
    parser.add_argument("--genius_token", default=os.environ.get("GENIUS_ACCESS_TOKEN"), help="Genius API Client Access Token. Can also be set via GENIUS_ACCESS_TOKEN environment variable.")

    args = parser.parse_args()

    global GENIUS_ACCESS_TOKEN 
    if args.genius_token:
        GENIUS_ACCESS_TOKEN = args.genius_token
    
    if not GENIUS_ACCESS_TOKEN or GENIUS_ACCESS_TOKEN == "YOUR_GENIUS_CLIENT_ACCESS_TOKEN_HERE" or GENIUS_ACCESS_TOKEN == "":
        print("WARNING: Genius API token is not set. Lyrics fetching via LyricsGenius will be skipped.")
        print("Please provide it via --genius_token argument or GENIUS_ACCESS_TOKEN environment variable, or create a .env file with it.")


    if not os.path.exists(args.url_file):
        print(f"Error: URL file not found at '{args.url_file}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not urls:
        print("No URLs found in the input file.")
        sys.exit(0)

    print(f"Found {len(urls)} URLs/Playlists to process.")

    ydl_opts_info = {
        'quiet': True,
        'ignoreerrors': True,
        'extract_flat': False, # Set to False to get individual video info from playlists
        'skip_download': True, 
        'nocheckcertificate': True,
        'playlist_items': '1-500' # Limit playlist items if needed
    }

    with YoutubeDL(ydl_opts_info) as ydl_info_extractor:
        for i, url_or_playlist in enumerate(urls):
            print(f"\n--- Processing source {i+1}/{len(urls)}: {url_or_playlist} ---")
            try:
                result_info_data = ydl_info_extractor.extract_info(url_or_playlist, download=False)
                
                if not result_info_data:
                    print(f"Could not fetch info for URL: {url_or_playlist}. Skipping.")
                    continue

                actual_videos: List[Dict[str, Any]] = []
                # Check if it's a playlist or a single video with entries
                if result_info_data.get('_type') == 'playlist' or 'entries' in result_info_data:
                    actual_videos = [entry for entry in result_info_data.get('entries', []) if entry]
                else: # Single video
                    actual_videos = [result_info_data]
                
                if not actual_videos:
                    print(f"  No video items found for URL/playlist: {url_or_playlist}")
                    continue
                
                print(f"  Found {len(actual_videos)} video(s) to process from this source.")
                for vid_idx, video_data in enumerate(actual_videos):
                    if not video_data: 
                        print(f"  Skipping null video data at index {vid_idx} for {url_or_playlist}")
                        continue
                    if not isinstance(video_data, dict):
                        print(f"  Skipping non-dictionary video_data at index {vid_idx}: {type(video_data)}")
                        continue
                    print(f"  Processing video {vid_idx+1}/{len(actual_videos)} from source '{url_or_playlist}'...")
                    process_video_entry(video_data, args.output_dir, args.skip_existing)

            except Exception as e:
                print(f"  Major error processing URL {url_or_playlist}: {e}")
                import traceback
                traceback.print_exc()

    print("\n--- All processing finished. ---")

if __name__ == "__main__":
    main()