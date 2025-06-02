# scripts/download_spotify_playlist.py
import os
import re
import argparse
import csv
from pathlib import Path
import subprocess
import time
import random

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import requests # For AZLyrics

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

DEFAULT_OUTPUT_DIR = "downloaded_files"
DOWNLOAD_REPORT_FILENAME = "spotify_download_report.csv"

# --- Helper Functions ---

def sanitize_filename(name: str) -> str:
    """Remove characters that are problematic for filenames."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_") # Replace spaces with underscores
    return name[:200]

def clean_lyrics_text(text: str) -> str:
    """Clean lyrics text: remove punctuation, ensure lowercase, remove annotations."""
    if not text:
        return ""

    # Remove content in brackets (like [Chorus], [Verse], [Outro], etc.)
    # and parentheses (often instrumental breaks or ad-libs notes)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)

    # Convert to lowercase
    text = text.lower()

    # Preserve intra-word apostrophes but replace other punctuation with a space.
    punctuation_to_replace_with_space = r"[.,;:?!(){}\[\]<>\"“”‘’«»‹›„“”‘‚’‛‟〃‶〝«»\-_–—=/\\|]"
    text = re.sub(punctuation_to_replace_with_space, " ", text)
    
    text = text.replace("'", "_APOSTROPHE_PLACEHOLDER_")
    text = re.sub(r"[^a-z0-9_APOSTROPHE_PLACEHOLDER_\s]", " ", text) # Keep letters, numbers, apostrophe placeholder, spaces
    text = text.replace("_APOSTROPHE_PLACEHOLDER_", "'")

    text = re.sub(r'\s+', ' ', text).strip() # Normalize multiple spaces and strip
    
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    text = "\n".join(cleaned_lines)

    lines = text.split('\n') # Re-split after potential changes
    # Remove very short lines or lines that are just apostrophes after cleaning
    final_lines = [line for line in lines if len(line.replace("'", "").strip()) > 0]
    return "\n".join(final_lines)


def is_english(text: str, min_length_for_detection: int = 20) -> bool:
    """Check if the provided text is in English."""
    if not text or len(text.strip()) < min_length_for_detection:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def download_audio_yt_dlp(track_name: str, artist_name: str, output_dir: Path, filename_base: str) -> (bool, Path | None):
    """Downloads audio for a track using yt-dlp."""
    search_query = f"{artist_name} - {track_name} audio"
    output_template = output_dir / f"{filename_base}.%(ext)s"
    mp3_output_path = output_dir / f"{filename_base}.mp3"

    if mp3_output_path.exists():
        print(f"    Audio already exists: {mp3_output_path.name}")
        return True, mp3_output_path

    command = [
        'yt-dlp', '-x', '--audio-format', 'mp3', '--audio-quality', '0',
        '-o', str(output_template),
        '--ffmpeg-location', os.getenv('FFMPEG_PATH', 'ffmpeg'),
        '--no-playlist', '--default-search', 'ytsearch1', search_query
    ]
    try:
        print(f"    Downloading audio for: {track_name} by {artist_name}...")
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"    Audio downloaded successfully: {mp3_output_path.name}")
        
        if mp3_output_path.exists():
             return True, mp3_output_path
        else:
            for f in output_dir.glob(f"{filename_base}.*"):
                if f.suffix in ['.mp3', '.m4a', '.ogg', '.wav']:
                    if f.suffix != '.mp3': print(f"    Warning: Audio downloaded as {f.suffix}, not mp3. Path: {f}")
                    return True, f
            print(f"    Error: MP3 file not found after download: {mp3_output_path}")
            return False, None
    except subprocess.CalledProcessError as e:
        print(f"    Error downloading audio for '{track_name}': {e.stderr}")
        return False, None
    except FileNotFoundError:
        print("    Error: yt-dlp or ffmpeg not found. Ensure they are installed and in PATH or FFMPEG_PATH is set.")
        return False, None
    except Exception as e:
        print(f"    An unexpected error occurred during audio download: {e}")
        return False, None

def sanitize_for_azlyrics_url(name: str, is_artist: bool = False) -> str:
    s = name.lower()
    if is_artist:
        if s.startswith("the "):
            s = s[4:]
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def get_lyrics_azlyrics(track_name: str, artist_name: str) -> str | None:
    """Fetches lyrics from AZLyrics.com."""
    print(f"    Attempting AZLyrics fallback for: {track_name} by {artist_name}")
    sanitized_artist = sanitize_for_azlyrics_url(artist_name, is_artist=True)
    sanitized_track = sanitize_for_azlyrics_url(track_name)

    if not sanitized_artist or not sanitized_track:
        print("    AZLyrics: Could not sanitize artist/track name for URL.")
        return None

    url = f"https://www.azlyrics.com/lyrics/{sanitized_artist}/{sanitized_track}.html"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        time.sleep(random.uniform(1, 2.5)) # Be respectful
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status() # Will raise an HTTPError for bad responses (4XX or 5XX)
        
        html_content = response.text
        
        # Lyrics are typically between this comment and the next div.
        # Sometimes there's a <!-- MxM banner --> comment in between.
        lyrics_regex = r"Usage of azlyrics\.com content by any third-party lyrics provider is prohibited by our licensing agreement\. Sorry about that\. -->\s*([\s\S]*?)\s*</div>"
        match = re.search(lyrics_regex, html_content)
        
        if match:
            lyrics_html_block = match.group(1)
            # Convert <br> to newlines
            lyrics_text = re.sub(r'<br\s*/?>', '\n', lyrics_html_block, flags=re.IGNORECASE)
            # Remove any other HTML tags (like <i>, <b>, <a>, <div> inside if any)
            lyrics_text = re.sub(r'<[^>]+>', '', lyrics_text)
            # Remove potential ad JS script lines sometimes embedded
            lyrics_text = re.sub(r'^\s*\(function\(\)\s*{[\s\S]*?}\)\(\);\s*$', '', lyrics_text, flags=re.MULTILINE)

            lyrics_text = lyrics_text.strip()
            if lyrics_text:
                print("    AZLyrics: Lyrics found and parsed.")
                return lyrics_text
            else:
                print("    AZLyrics: Parsed block is empty.")
                return None
        else:
            print("    AZLyrics: Lyrics pattern not found in HTML.")
            return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"    AZLyrics: Page not found (404) at {url}")
        else:
            print(f"    AZLyrics: HTTP error {e.response.status_code} for {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"    AZLyrics: Request error for {url}: {e}")
        return None
    except Exception as e:
        print(f"    AZLyrics: Unexpected error: {e}")
        return None


def get_lyrics_genius(track_name: str, artist_name: str, genius_api) -> str | None:
    """Fetches lyrics using LyricsGenius."""
    try:
        print(f"    Fetching lyrics from Genius for: {track_name} by {artist_name}...")
        song = genius_api.search_song(track_name, artist_name, get_full_info=False)
        if song and song.lyrics:
            raw_lyrics = song.lyrics
            # Clean common Genius headers/footers
            lines = raw_lyrics.split('\n')
            if len(lines) > 1:
                first_line_lower = lines[0].lower()
                # Check if first line strongly matches "Track Title Lyrics" or includes artist
                # This is a heuristic.
                if (f"{track_name.lower()} lyrics" in first_line_lower or 
                    artist_name.lower() in first_line_lower) and \
                   (lines[0].strip().endswith("Lyrics") or lines[0].strip().endswith("lyrics")):
                    # A more specific pattern for common Genius titles
                    potential_header_pattern = rf".*?{re.escape(track_name)}.*Lyrics(\s*\[.*\])?$"
                    if re.match(potential_header_pattern, lines[0], re.IGNORECASE):
                       raw_lyrics = '\n'.join(lines[1:])
                       print(f"    Genius: Removed potential header: '{lines[0]}'")
            
            # Remove common footers
            raw_lyrics = re.sub(r'\d*EmbedShare URLCopyEmbedCopy$', '', raw_lyrics).strip()
            raw_lyrics = re.sub(r'\d*Embed$', '', raw_lyrics).strip()
            raw_lyrics = re.sub(r'\S+\sComposers:.*', '', raw_lyrics, flags=re.IGNORECASE)
            raw_lyrics = re.sub(r'\S+\sLyrics:.*', '', raw_lyrics, flags=re.IGNORECASE)
            raw_lyrics = re.sub(r'You might also like.*', '', raw_lyrics, flags=re.IGNORECASE | re.DOTALL) # Remove "You might also like" and subsequent content
            
            print("    Genius: Lyrics found.")
            return raw_lyrics.strip()
        else:
            print("    Genius: Lyrics not found.")
            return None
    except Exception as e:
        print(f"    Genius: Error fetching lyrics for '{track_name}': {e}")
        return None

def get_lyrics(track_name: str, artist_name: str, genius_api) -> (str | None, str):
    """Tries to fetch lyrics from Genius, then AZLyrics as a fallback."""
    lyrics, source = None, "None"

    # Try Genius first
    lyrics = get_lyrics_genius(track_name, artist_name, genius_api)
    if lyrics:
        source = "Genius"
        return lyrics, source

    # If Genius fails, try AZLyrics
    print("    Genius failed, trying AZLyrics fallback...")
    lyrics = get_lyrics_azlyrics(track_name, artist_name)
    if lyrics:
        source = "AZLyrics"
    
    return lyrics, source


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Download a Spotify playlist's tracks (audio and lyrics).")
    parser.add_argument("playlist_url", type=str, help="The URL of the Spotify playlist.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save downloaded files. Default: '{DEFAULT_OUTPUT_DIR}'")
    parser.add_argument("--spotify_client_id", type=str, default=SPOTIFY_CLIENT_ID, help="Spotify Client ID.")
    parser.add_argument("--spotify_client_secret", type=str, default=SPOTIFY_CLIENT_SECRET, help="Spotify Client Secret.")
    parser.add_argument("--genius_token", type=str, default=GENIUS_ACCESS_TOKEN, help="Genius API Access Token.")
    parser.add_argument("--force_redownload", action="store_true", help="Force re-download of audio and lyrics even if files exist.")

    args = parser.parse_args()

    if not args.spotify_client_id or not args.spotify_client_secret:
        print("Spotify API credentials are required. Set environment variables or pass as arguments.")
        return
    if not args.genius_token:
        print("Genius API token is required. Set environment variable or pass as argument.")
        return

    try:
        spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=args.spotify_client_id,
                                                                       client_secret=args.spotify_client_secret))
        print("Successfully authenticated with Spotify API.")
    except Exception as e:
        print(f"Error authenticating with Spotify: {e}"); return

    try:
        # Genius API: verbose=False, remove_section_headers=True (though we do more), skip_non_songs=True
        genius = lyricsgenius.Genius(args.genius_token, verbose=False, remove_section_headers=False, 
                                     skip_non_songs=True, timeout=15, retries=2)
        print("Successfully initialized LyricsGenius API.")
    except Exception as e:
        print(f"Error initializing LyricsGenius: {e}"); return
        
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Downloads will be saved to: {output_path.resolve()}")

    report_data = []
    try:
        playlist_id = args.playlist_url.split("/")[-1].split("?")[0]
        results = spotify.playlist_items(playlist_id)
        tracks = results['items']
        while results['next']:
            results = spotify.next(results)
            tracks.extend(results['items'])
        print(f"Found {len(tracks)} tracks in the playlist.")
    except Exception as e:
        print(f"Error fetching playlist details from Spotify: {e}"); return

    for i, item in enumerate(tracks):
        track_info = item['track']
        if not track_info:
            print(f"Skipping item {i+1} (not a track or unavailable).")
            report_data.append({
                'track_number': i + 1, 'track_name': "N/A (skipped item)", 'artist_name': "",
                'audio_downloaded': "No", 'audio_path': "N/A",
                'lyrics_source': "None", 'lyrics_english_and_saved': "No", 'lyrics_path': "N/A",
                'status': "Skipped - Invalid track item"
            })
            continue

        track_name = track_info['name']
        artists = [artist['name'] for artist in track_info['artists']]
        artist_name_primary = artists[0] if artists else "Unknown Artist"
        artist_name_str = ", ".join(artists)

        print(f"\nProcessing track {i+1}/{len(tracks)}: {track_name} by {artist_name_str}")

        sanitized_track_name = sanitize_filename(track_name)
        sanitized_artist_name = sanitize_filename(artist_name_primary)
        filename_base = f"{sanitized_artist_name}_{sanitized_track_name}"

        audio_file_path = output_path / f"{filename_base}.mp3"
        lyrics_file_path = output_path / f"{filename_base}.txt"

        audio_downloaded_flag = False
        final_audio_path = None
        if args.force_redownload or not audio_file_path.exists():
            audio_downloaded_flag, final_audio_path = download_audio_yt_dlp(track_name, artist_name_primary, output_path, filename_base)
        else:
            print(f"    Audio already exists (skipping download): {audio_file_path.name}")
            audio_downloaded_flag = True
            final_audio_path = audio_file_path
        
        lyrics_source = "None"
        lyrics_english_saved_flag = False
        raw_lyrics = None

        if args.force_redownload or not lyrics_file_path.exists():
            raw_lyrics, lyrics_source = get_lyrics(track_name, artist_name_primary, genius)
            if raw_lyrics:
                cleaned_lyrics = clean_lyrics_text(raw_lyrics)
                if is_english(cleaned_lyrics):
                    try:
                        with open(lyrics_file_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_lyrics)
                        print(f"    Cleaned English lyrics (from {lyrics_source}) saved: {lyrics_file_path.name}")
                        lyrics_english_saved_flag = True
                    except Exception as e:
                        print(f"    Error saving lyrics to file: {e}")
                else:
                    print(f"    Lyrics found (from {lyrics_source}) but not in English or too short. Skipping save.")
            else:
                 print(f"    No lyrics found from any source.")
        elif lyrics_file_path.exists():
            print(f"    Lyrics file already exists (skipping fetch): {lyrics_file_path.name}")
            # Try to determine source if needed, or just mark as existing
            lyrics_english_saved_flag = True # Assume it's good if it exists and not forced
            lyrics_source = "Existing File" # Can't know original source easily without re-parsing
            try: # Quick check if existing file is english
                with open(lyrics_file_path, 'r', encoding='utf-8') as f:
                    if not is_english(f.read()):
                        lyrics_english_saved_flag = False 
                        print(f"    Warning: Existing lyrics file {lyrics_file_path.name} may not be English.")
            except Exception: pass


        report_data.append({
            'track_number': i + 1, 'track_name': track_name, 'artist_name': artist_name_str,
            'audio_downloaded': "Yes" if audio_downloaded_flag and final_audio_path and final_audio_path.exists() else "No",
            'audio_path': str(final_audio_path.resolve()) if final_audio_path and final_audio_path.exists() else "N/A",
            'lyrics_source': lyrics_source if (args.force_redownload or not lyrics_file_path.exists()) else "Existing File",
            'lyrics_english_and_saved': "Yes" if lyrics_english_saved_flag and lyrics_file_path.exists() else "No",
            'lyrics_path': str(lyrics_file_path.resolve()) if lyrics_english_saved_flag and lyrics_file_path.exists() else "N/A",
            'status': "Success" if audio_downloaded_flag and lyrics_english_saved_flag else "Partial or Failed"
        })

    report_file_path = output_path / DOWNLOAD_REPORT_FILENAME
    if report_data:
        try:
            with open(report_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['track_number', 'track_name', 'artist_name', 'audio_downloaded', 'audio_path',
                              'lyrics_source', 'lyrics_english_and_saved', 'lyrics_path', 'status']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report_data)
            print(f"\nDownload report saved to: {report_file_path.resolve()}")
        except Exception as e:
            print(f"\nError saving download report: {e}")

    print("\nPlaylist processing finished.")

if __name__ == "__main__":
    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET and GENIUS_ACCESS_TOKEN):
        print("Warning: API credentials (SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, GENIUS_ACCESS_TOKEN) missing from environment.")
        print("Pass as command-line arguments or set in .env file.")
    main()