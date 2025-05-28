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
    if not GENIUS_ACCESS_TOKEN or GENIUS_ACCESS_TOKEN == "YOUR_GENIUS_CLIENT_ACCESS_TOKEN_HERE":
        print("  GENIUS_ACCESS_TOKEN not set in the script. Skipping lyricsgenius fetch.")
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
                 print(f"  No direct match for '{song_title}' by '{artist_name}'. Trying title only.")
            song_result = genius.search_song(song_title)

        if song_result:
            print(f"  Lyrics found via lyricsgenius for '{song_result.title}' (Artist: {song_result.artist})")
            lyrics_text = song_result.lyrics.strip()
            
            # Additional cleaning for common Genius artifacts
            lyrics_text = re.sub(r"^\d*EmbedShare URLCopyEmbedCopy$", "", lyrics_text, flags=re.MULTILINE | re.IGNORECASE).strip()
            lyrics_text = re.sub(r"\d*Embed$", "", lyrics_text, flags=re.IGNORECASE).strip() 
            lyrics_text = re.sub(r"You might also like", "", lyrics_text, flags=re.IGNORECASE).strip() 
            lyrics_text = re.sub(r"^\[[^\]]+\]\s*\n?", "", lyrics_text, flags=re.MULTILINE) # Remove [Verse], [Chorus] at start of lines

            # Specifically remove the "ContributorsTranslations..." line if it's the first line
            lines = lyrics_text.split('\n')
            if lines:
                # Pattern to match "Number Contributors", "Translations", language names
                # Example: "364 ContributorsTranslationsFrançaisTü"
                # More general: Starts with a number, then "Contributor(s)", then optionally "Translations", then language names / special chars
                # Regex: r"^\d+\s*Contributors?(Translations)?([A-Za-zÀ-ÖØ-öø-ÿ\s]+)?$"
                # Simpler check for the observed pattern:
                first_line_pattern = r"^\d+\s*Contributor(s)?(Translations)?([A-Za-zÀ-ÖØ-öø-ÿ\s]+)?$"
                if re.match(first_line_pattern, lines[0].strip(), re.IGNORECASE):
                    print(f"  Removing detected metadata line: '{lines[0]}'")
                    lines.pop(0)
                # Also remove the actual word "Lyrics" if it appears as the first line after metadata
                if lines and lines[0].strip().lower() == "lyrics":
                    print(f"  Removing detected 'Lyrics' header line: '{lines[0]}'")
                    lines.pop(0)

            lyrics_text = "\n".join(lines).strip()
            
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
    base_filename_sanitized = sanitize_filename(original_title)
    
    mp3_final_filepath = os.path.join(output_dir, f"{base_filename_sanitized}.mp3")
    txt_filepath = os.path.join(output_dir, f"{base_filename_sanitized}.txt")

    print(f"\nProcessing: {original_title} (URL: {video_url})")
    print(f"  MP3 target: {mp3_final_filepath}")
    print(f"  TXT target: {txt_filepath}")

    audio_downloaded_successfully = False
    mp3_exists_and_valid = os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) > 1000

    if skip_existing and mp3_exists_and_valid:
        print(f"  MP3 already exists and is valid: {mp3_final_filepath}. Skipping download.")
        audio_downloaded_successfully = True
    elif mp3_exists_and_valid: 
        print(f"  MP3 {mp3_final_filepath} already exists. Download behavior depends on 'overwrites' yt-dlp option.")
        audio_downloaded_successfully = True 
    
    if not audio_downloaded_successfully: 
        output_template_base = os.path.join(output_dir, base_filename_sanitized)

        ydl_opts_single = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_template_base + '.%(ext)s',
            'quiet': False,
            'ignoreerrors': True,
            'restrictfilenames': False,
            'overwrites': not skip_existing,
            'nocheckcertificate': True,
            'keepvideo': False,
        }
        try:
            with YoutubeDL(ydl_opts_single) as ydl_single:
                error_code = ydl_single.download([video_url])
                
                if os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) > 1000:
                    print(f"  Successfully downloaded and converted audio to: {mp3_final_filepath}")
                    audio_downloaded_successfully = True
                else:
                    double_ext_filepath = mp3_final_filepath + ".mp3" 
                    if os.path.exists(double_ext_filepath) and os.path.getsize(double_ext_filepath) > 1000:
                        print(f"  Found file with double extension: {double_ext_filepath}. Renaming to {mp3_final_filepath}")
                        try:
                            if os.path.exists(mp3_final_filepath):
                                os.remove(mp3_final_filepath)
                            os.rename(double_ext_filepath, mp3_final_filepath)
                            if os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) > 1000:
                                print(f"  Successfully renamed. Final audio: {mp3_final_filepath}")
                                audio_downloaded_successfully = True
                        except Exception as rename_e:
                            print(f"  Error renaming {double_ext_filepath} to {mp3_final_filepath}: {rename_e}")
                    
                    if not audio_downloaded_successfully:
                        print(f"  yt-dlp potentially failed (error code: {error_code}). Final MP3 '{mp3_final_filepath}' not found or too small.")
                        if os.path.exists(mp3_final_filepath) and os.path.getsize(mp3_final_filepath) < 1000:
                            try: os.remove(mp3_final_filepath)
                            except: pass
                            
        except Exception as e:
            print(f"  Exception during audio download for {video_url}: {e}")

    lyrics_exist_and_valid = os.path.exists(txt_filepath) and os.path.getsize(txt_filepath) > 10

    if skip_existing and lyrics_exist_and_valid:
        print(f"  Lyrics file already exists and is valid: {txt_filepath}. Skipping fetch.")
    elif audio_downloaded_successfully: 
        artist = video_info.get('artist') or video_info.get('uploader') or video_info.get('channel')
        
        cleaned_title_for_lyrics = original_title
        common_video_terms = [
            r"official video", r"music video", r"lyrics video", r"lyric video",
            r"official audio", r"audio", r"hd", r"hq", r"4k", r"\(.*?\)", r"\[.*?\]",
            r"ft\.", r"feat\."
        ]
        for term in common_video_terms:
            cleaned_title_for_lyrics = re.sub(term, "", cleaned_title_for_lyrics, flags=re.IGNORECASE).strip()
        if artist:
             cleaned_title_for_lyrics = re.sub(r"\s*-\s*" + re.escape(artist) + r"\s*$", "", cleaned_title_for_lyrics, flags=re.IGNORECASE).strip()

        cleaned_title_for_lyrics = re.sub(r'\s{2,}', ' ', cleaned_title_for_lyrics)

        lyrics_content = fetch_lyrics_with_genius(cleaned_title_for_lyrics, artist)
        if lyrics_content:
            try:
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(lyrics_content)
                print(f"  Lyrics saved to: {txt_filepath}")
            except IOError as e:
                print(f"  Error saving lyrics to {txt_filepath}: {e}")
        else:
            print(f"  No lyrics found via lyricsgenius for '{original_title}'. Creating placeholder TXT.")
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
    
    if not GENIUS_ACCESS_TOKEN or GENIUS_ACCESS_TOKEN == "YOUR_GENIUS_CLIENT_ACCESS_TOKEN_HERE":
        print("WARNING: Genius API token is not set. Lyrics fetching via LyricsGenius will be skipped.")
        print("Please provide it via --genius_token argument or GENIUS_ACCESS_TOKEN environment variable, or edit it directly in the script.")


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
        'extract_flat': False, 
        'skip_download': True, 
        'nocheckcertificate': True,
        'playlist_items': '1-500' 
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
                if result_info_data.get('_type') == 'playlist':
                    actual_videos = [entry for entry in result_info_data.get('entries', []) if entry]
                elif 'entries' in result_info_data : 
                    actual_videos = [entry for entry in result_info_data.get('entries', []) if entry] 
                else: 
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