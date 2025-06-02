# beefai/scripts/fetch_song_data.py
import os
import sys
import argparse
import re
import time
import json
import csv
# import requests # Not strictly needed if all web access is via spotdl/spotipy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

# Updated spotdl imports for v4.x
try:
    from spotdl import Downloader
    from spotdl.types.song import Song
    # SpotifyClient is not directly called for init, Downloader handles it internally.
    SPOTDL_AVAILABLE = True
except ImportError:
    SPOTDL_AVAILABLE = False
    print("WARNING: spotdl library not found or version incompatible. MP3 downloading and spotdl lyrics fetching will be skipped. Install/update with 'pip install spotdl'")

load_dotenv()

# --- Configuration ---
REQUEST_TIMEOUT = 20
DOWNLOAD_DELAY_PER_REQUEST = 1.5
DEFAULT_INTER_SONG_DELAY = 3
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
CHARS_TO_DELETE_PATTERN = r'[.,!?"():;“”‘’«»–—/[\]{}<>*&^%$#@+=~`|_]'

# --- Lyric Cleaning Helper Functions ---
def pre_clean_raw_lyrics(lyrics_text: str, source: str = "spotdl") -> str:
    if not lyrics_text: return ""
    lyrics_text = re.sub(r"^\s*\[[^\]]+\]\s*\n?", "", lyrics_text, flags=re.MULTILINE).strip()
    if source.lower() == "spotdl":
        lyrics_text = re.sub(r"\d*EmbedShare URLCopyEmbedCopy$", "", lyrics_text, flags=re.MULTILINE | re.IGNORECASE).strip()
        lyrics_text = re.sub(r"\d*Embed$", "", lyrics_text, flags=re.IGNORECASE).strip()
        lyrics_text = re.sub(r"You might also like.*?(?=\n\n|\Z)", "", lyrics_text, flags=re.IGNORECASE | re.DOTALL).strip()
        lyrics_text = re.sub(r"\s*\*+\s*This Lyrics is NOT for Commercial use\s*\*+\s*", "", lyrics_text, flags=re.IGNORECASE | re.DOTALL).strip()
        lyrics_text = re.sub(r"\s*\(\d+\)\s*$", "", lyrics_text).strip()
        lyrics_text = re.sub(r"Lyrics for .*? by .*", "", lyrics_text, flags=re.IGNORECASE).strip()
        lyrics_text = re.sub(r"Source: Musixmatch", "", lyrics_text, flags=re.IGNORECASE).strip()
        lyrics_text = re.sub(r"Writer\(s\):.*", "", lyrics_text, flags=re.IGNORECASE | re.DOTALL).strip()
        lyrics_text = re.sub(r"******* This Lyrics is NOT for Commercial use *******", "", lyrics_text, flags=re.IGNORECASE | re.DOTALL).strip()
    lyrics_text = re.sub(r'\[[^\]]+\]', '', lyrics_text).strip()
    lyrics_text = re.sub(r'\((?:SOUND|MUSIC|LAUGHTER|APPLAUSE|NOISE|BLEEP)[^\)]*\)', '', lyrics_text, flags=re.IGNORECASE).strip()
    lyrics_text = re.sub(r'>>\s*[^:]+:\s*', '', lyrics_text).strip()
    lyrics_text = re.sub(r'^\s*\.\.\.\s*$', '', lyrics_text, flags=re.MULTILINE).strip()
    lines = lyrics_text.split('\n')
    lines = [line for line in lines if line.strip()]
    return "\n".join(lines).strip()

def apply_final_lyric_cleaning(text_content: str) -> str:
    if not text_content: return ""
    lines = text_content.splitlines()
    cleaned_lines = []
    for line in lines:
        line_no_punct = re.sub(CHARS_TO_DELETE_PATTERN, '', line)
        line_normalized_space = re.sub(r'[ \t]+', ' ', line_no_punct).strip()
        if line_normalized_space: cleaned_lines.append(line_normalized_space)
    return "\n".join(cleaned_lines).strip()

# --- Spotify Functions (spotipy specific for playlist fetching) ---
def init_spotipy_client() -> Optional[spotipy.Spotify]:
    if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET:
        print("  Spotipy API: Client ID or Secret not found for playlist fetching. Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET env vars.")
        return None
    try:
        client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        print("  Spotipy API: Client for playlist fetching initialized successfully.")
        return sp
    except Exception as e:
        print(f"  Spotipy API: Failed to initialize client for playlist fetching: {e}")
        return None

def get_tracks_from_spotify_playlist(sp: spotipy.Spotify, playlist_url: str) -> List[Dict[str, Any]]:
    tracks_to_process = []
    if not sp: return tracks_to_process
    try:
        playlist_id = playlist_url.split('/')[-1].split('?')[0]
        print(f"  Spotipy API: Fetching tracks from playlist ID: {playlist_id}")
        results = sp.playlist_items(playlist_id, fields='items(track(name,artists(name),id,uri,type)),next')
        if not results: return tracks_to_process
        for item in results['items']:
            if item and item.get('track') and item['track'].get('type') == 'track':
                track = item['track']
                tracks_to_process.append({
                    'title': track['name'],
                    'artist': track['artists'][0]['name'] if track['artists'] else "Unknown Artist",
                    'spotify_id': track.get('id'),
                    'spotify_uri': track.get('uri')
                })
        while results['next']:
            results = sp.next(results)
            for item in results['items']:
                if item and item.get('track') and item['track'].get('type') == 'track':
                    track = item['track']
                    tracks_to_process.append({
                        'title': track['name'],
                        'artist': track['artists'][0]['name'] if track['artists'] else "Unknown Artist",
                        'spotify_id': track.get('id'),
                        'spotify_uri': track.get('uri')
                    })
        print(f"  Spotipy API: Fetched {len(tracks_to_process)} tracks from playlist '{playlist_id}'.")
    except Exception as e:
        print(f"  Spotipy API: Error fetching playlist tracks from URL '{playlist_url}': {e}")
    return tracks_to_process

# --- CSV Handling ---
CSV_FIELDNAMES = ['spotify_artist', 'spotify_title', 'sanitized_filename_base', 'mp3_path', 'lyrics_path', 'mp3_found', 'lyrics_saved', 'status_notes']
def initialize_csv(csv_filepath: str):
    if not os.path.exists(csv_filepath):
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
        print(f"  CSV report initialized at: {csv_filepath}")
def append_to_csv(csv_filepath: str, data_row: Dict[str, Any]):
    try:
        with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            writer.writerow(data_row)
    except IOError as e: print(f"  ERROR: Could not write to CSV '{csv_filepath}': {e}")

# --- Main Processing Logic ---
def sanitize_filename(name: str) -> str:
    if not name: name = "unknown_title"
    name = str(name)
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = name.replace("\n", "_").replace("\r", "_")
    name = re.sub(r'\s+', '_', name.strip())
    name = re.sub(r'_+', '_', name)
    return name[:150]

def process_song_entry(
    song_query_data: Dict[str, Any],
    output_dir: str,
    skip_existing_lyrics: bool,
    skip_existing_mp3: bool,
    spotdl_downloader: Optional[Downloader] # Pass the Downloader instance
) -> Dict[str, Any]:
    spotify_title = song_query_data.get('title', 'Unknown Title')
    spotify_artist = song_query_data.get('artist', 'Unknown Artist')
    spotify_uri_original = song_query_data.get('spotify_uri')

    report_entry: Dict[str, Any] = {
        'spotify_title': spotify_title, 'spotify_artist': spotify_artist,
        'sanitized_filename_base': '', 'mp3_path': '', 'lyrics_path': '',
        'mp3_found': False, 'lyrics_saved': False, 'status_notes': []
    }
    print(f"\nProcessing: '{spotify_title}' by '{spotify_artist}' (URI: {spotify_uri_original})")
    filename_base_text = f"{spotify_artist} - {spotify_title}"
    sanitized_base = sanitize_filename(filename_base_text)
    report_entry['sanitized_filename_base'] = sanitized_base

    mp3_ext = 'mp3'
    if spotdl_downloader and hasattr(spotdl_downloader, 'settings') and spotdl_downloader.settings:
        mp3_ext = spotdl_downloader.settings.get("audio_format", "mp3")

    mp3_filename = f"{sanitized_base}.{mp3_ext}"
    mp3_target_path = os.path.join(output_dir, mp3_filename)
    report_entry['mp3_path'] = mp3_target_path

    lyrics_filename = f"{sanitized_base}.txt"
    lyrics_filepath = os.path.join(output_dir, lyrics_filename)
    report_entry['lyrics_path'] = lyrics_filepath

    print(f"  Sanitized base for files: {sanitized_base}")
    print(f"  MP3 target path: {mp3_target_path} (format: {mp3_ext})")
    print(f"  Lyrics output path: {lyrics_filepath}")

    processed_song_object: Optional[Song] = None
    mp3_exists_and_valid = os.path.exists(mp3_target_path) and os.path.getsize(mp3_target_path) > 1000
    lyrics_exist_and_valid_txt = os.path.exists(lyrics_filepath) and os.path.getsize(lyrics_filepath) > 10

    url_for_song_lookup = None
    if spotify_uri_original:
        if spotify_uri_original.startswith("spotify:track:"):
            track_id = spotify_uri_original.split(':')[-1]
            url_for_song_lookup = f"https://open.spotify.com/track/{track_id}"
            print(f"  spotdl: Converted URI '{spotify_uri_original}' to URL: {url_for_song_lookup}")
        elif spotify_uri_original.startswith("https://open.spotify.com/track/"):
            url_for_song_lookup = spotify_uri_original
        else: # Could be a search query or other URL type
            url_for_song_lookup = spotify_uri_original
            print(f"  spotdl: Query '{url_for_song_lookup}' is not a Spotify track URI/URL. Will be treated as search/generic URL by spotdl.")
    else: # Fallback to search string if no URI provided
        url_for_song_lookup = f"{spotify_artist} {spotify_title}"
        print(f"  spotdl: No Spotify URI provided, will use search string: '{url_for_song_lookup}'")

    if SPOTDL_AVAILABLE and url_for_song_lookup:
        if not spotdl_downloader: # If Downloader instance is None, client initialization by spotdl likely failed or was skipped.
            print(f"  spotdl: Skipping Song object creation because spotdl Downloader is not available (implies Spotify client not initialized by spotdl).")
            report_entry['status_notes'].append(f"spotdl Song object creation skipped (Downloader not available/initialized).")
        else: # Downloader is available, so SpotifyClient should have been initialized by it.
            try:
                print(f"  spotdl: Attempting to create Song object from: {url_for_song_lookup}")
                processed_song_object = Song.from_url(url_for_song_lookup)
                if processed_song_object:
                     print(f"  spotdl: Successfully created Song object for '{processed_song_object.name}'.")
                else: # Song.from_url can return None if not found
                    print(f"  spotdl: Song.from_url did not find a match for '{url_for_song_lookup}'.")
                    report_entry['status_notes'].append(f"spotdl Song.from_url found no match for query.")
            except Exception as e_song_create:
                print(f"  spotdl: Failed to create Song object from '{url_for_song_lookup}': {type(e_song_create).__name__} - {e_song_create}")
                report_entry['status_notes'].append(f"spotdl Song object creation failed: {e_song_create}")
                if os.path.exists(mp3_target_path) and os.path.getsize(mp3_target_path) > 1000:
                    report_entry['mp3_found'] = True
    elif not SPOTDL_AVAILABLE:
        print("  spotdl: Library not available, skipping Song object creation.")
        report_entry['status_notes'].append("spotdl library not available.")


    if skip_existing_mp3 and mp3_exists_and_valid:
        report_entry['mp3_found'] = True
        print(f"  MP3 already exists: {mp3_target_path}. Skipping download.")
        report_entry['status_notes'].append("MP3 skipped (already exists).")
    elif spotdl_downloader and processed_song_object:
        print(f"  Attempting to download MP3 using spotdl for: {processed_song_object.display_name}")
        try:
            song_obj_after_download, downloaded_file_path_obj = spotdl_downloader.download_song(processed_song_object)
            processed_song_object = song_obj_after_download

            if downloaded_file_path_obj:
                print(f"  spotdl: Download attempt successful, file at: {downloaded_file_path_obj}")
                if os.path.normpath(str(downloaded_file_path_obj)) == os.path.normpath(mp3_target_path) and \
                   os.path.exists(mp3_target_path) and os.path.getsize(mp3_target_path) > 1000:
                    report_entry['mp3_found'] = True
                    print(f"  MP3 downloaded successfully to target path: {mp3_target_path}")
                    report_entry['status_notes'].append(f"MP3 downloaded by spotdl to {mp3_target_path}.")
                elif os.path.exists(str(downloaded_file_path_obj)) and os.path.dirname(str(downloaded_file_path_obj)) == os.path.abspath(output_dir):
                    print(f"  spotdl: MP3 downloaded to '{downloaded_file_path_obj}', expected '{mp3_target_path}'.")
                    try:
                        os.makedirs(os.path.dirname(mp3_target_path), exist_ok=True)
                        if os.path.exists(mp3_target_path) and not skip_existing_mp3:
                            os.remove(mp3_target_path); print(f"  Removed existing '{mp3_target_path}'.")
                        if not os.path.exists(mp3_target_path) or not skip_existing_mp3:
                            os.rename(str(downloaded_file_path_obj), mp3_target_path)
                            print(f"  Renamed '{downloaded_file_path_obj}' to '{mp3_target_path}'.")
                            report_entry['mp3_found'] = True
                        else:
                            report_entry['mp3_found'] = True
                            print(f"  Target '{mp3_target_path}' exists and skip_existing_mp3 is true, did not rename.")
                        report_entry['status_notes'].append(f"MP3 downloaded by spotdl and ensured at {mp3_target_path}.")
                    except OSError as e_rename:
                        print(f"  ERROR: Failed to rename/move downloaded MP3: {e_rename}")
                        report_entry['status_notes'].append(f"MP3 downloaded to {downloaded_file_path_obj}, rename/move failed.")
                        report_entry['mp3_found'] = os.path.exists(str(downloaded_file_path_obj))
                else:
                    report_entry['mp3_found'] = False
                    print(f"  spotdl: Downloaded file '{downloaded_file_path_obj}' not at expected location or invalid.")
                    report_entry['status_notes'].append(f"spotdl downloaded to {downloaded_file_path_obj}, not matching target.")
            else:
                report_entry['mp3_found'] = False
                print(f"  spotdl: MP3 download failed for '{spotify_title}'. No file path returned.")
                report_entry['status_notes'].append("MP3 download via spotdl failed (no path).")
        except Exception as e_spotdl_dl:
            report_entry['mp3_found'] = False
            print(f"  ERROR during spotdl download_song execution: {type(e_spotdl_dl).__name__} - {e_spotdl_dl}")
            report_entry['status_notes'].append(f"MP3 download error: {e_spotdl_dl}")
    elif not spotdl_downloader:
        note = f"MP3 NOT DOWNLOADED (spotdl Downloader unavailable): {mp3_target_path}"
        if not mp3_exists_and_valid: report_entry['status_notes'].append(note)
    elif not processed_song_object:
        note = f"MP3 DOWNLOAD SKIPPED (spotdl Song object not available/found): {mp3_target_path}"
        if not mp3_exists_and_valid: report_entry['status_notes'].append(note)
        print(f"  WARNING: {note}")


    if skip_existing_lyrics and lyrics_exist_and_valid_txt:
        report_entry['lyrics_saved'] = True
        if "Lyrics skipped (.txt exists)." not in report_entry['status_notes']:
             print(f"  Lyrics .txt file already exists: {lyrics_filepath}. Skipping fetch.")
             report_entry['status_notes'].append("Lyrics skipped (.txt exists).")
    elif spotdl_downloader and processed_song_object:
        if not processed_song_object.lyrics:
            print(f"  Song object lyrics attribute is empty. Explicitly fetching with spotdl for: {processed_song_object.display_name}")
            try:
                enriched_song_obj, lyrics_text_spotdl = spotdl_downloader.search_and_get_lyrics(processed_song_object)
                if lyrics_text_spotdl:
                    processed_song_object = enriched_song_obj
                    print(f"  spotdl: Explicit lyric search yielded lyrics for '{processed_song_object.name}'.")
                else:
                    print(f"  spotdl: Explicit lyric search found no lyrics for '{processed_song_object.name}'.")
            except Exception as e_lyric_explicit:
                print(f"  spotdl: Error during explicit lyric fetch attempt: {e_lyric_explicit}")
                report_entry['status_notes'].append(f"spotdl explicit lyric fetch error: {e_lyric_explicit}")

        raw_lyrics_from_spotdl = processed_song_object.lyrics if processed_song_object else None
        if raw_lyrics_from_spotdl:
            print(f"  spotdl: Lyrics found for '{spotify_title}'.")
            cleaned_lyrics_intermediate = pre_clean_raw_lyrics(raw_lyrics_from_spotdl, source="spotdl")
            final_lyrics = apply_final_lyric_cleaning(cleaned_lyrics_intermediate)
            if final_lyrics and final_lyrics.strip():
                try:
                    os.makedirs(os.path.dirname(lyrics_filepath), exist_ok=True)
                    with open(lyrics_filepath, 'w', encoding='utf-8') as f: f.write(final_lyrics)
                    report_entry['lyrics_saved'] = True
                    print(f"  Lyrics saved to: {lyrics_filepath}")
                    if "Lyrics fetched by spotdl and saved." not in report_entry['status_notes']:
                        report_entry['status_notes'].append("Lyrics fetched by spotdl and saved.")
                except IOError as e:
                    report_entry['lyrics_saved'] = False; report_entry['status_notes'].append(f"Error saving lyrics: {e}")
                    print(f"  ERROR: {report_entry['status_notes'][-1]}")
            else:
                report_entry['lyrics_saved'] = False; report_entry['status_notes'].append("No valid lyrics after cleaning (spotdl).")
                print(f"  spotdl: {report_entry['status_notes'][-1]}")
                if os.path.exists(lyrics_filepath):
                    try: os.remove(lyrics_filepath)
                    except OSError: pass
        else:
            report_entry['lyrics_saved'] = False
            no_lyrics_note = "Lyrics not found by spotdl (lyrics attribute empty after all attempts)."
            if no_lyrics_note not in report_entry['status_notes']: report_entry['status_notes'].append(no_lyrics_note)
            print(f"  spotdl: {no_lyrics_note}")
            if os.path.exists(lyrics_filepath):
                try: os.remove(lyrics_filepath)
                except OSError: pass
    elif not spotdl_downloader:
        if not lyrics_exist_and_valid_txt: report_entry['status_notes'].append("Lyrics fetch skipped (spotdl Downloader unavailable).")
    elif not processed_song_object:
        if not lyrics_exist_and_valid_txt: report_entry['status_notes'].append("Lyrics fetch skipped (spotdl Song object not available/found).")

    if not report_entry['mp3_found']:
        if os.path.exists(mp3_target_path) and os.path.getsize(mp3_target_path) > 1000 :
             report_entry['mp3_found'] = True
             if "MP3 skipped (already exists)." not in report_entry['status_notes'] and \
                "MP3 found (pre-existing)." not in report_entry['status_notes']:
                report_entry['status_notes'].append("MP3 found (pre-existing).")
        elif not spotdl_downloader and f"MP3 NOT DOWNLOADED (spotdl Downloader unavailable): {mp3_target_path}" not in report_entry['status_notes']:
             report_entry['status_notes'].append(f"MP3 NOT DOWNLOADED (spotdl Downloader unavailable): {mp3_target_path}")

    return report_entry

def main():
    parser = argparse.ArgumentParser(description="Download MP3s, fetch lyrics (via spotdl), and generate CSV report.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--song_list_file", help="Path to CSV/TSV (cols: 'artist', 'title', optional 'spotify_uri').")
    group.add_argument("--playlist_url", help="URL of a public Spotify playlist.")
    parser.add_argument("--output_dir", default="downloaded_songs", help="Unified output directory. Default: %(default)s")
    parser.add_argument("--skip_existing_lyrics", action="store_true", help="Skip fetching lyrics if .txt exists.")
    parser.add_argument("--skip_existing_mp3", action="store_true", help="Skip downloading MP3 if file exists.")
    parser.add_argument("--report_file", default="song_processing_report.csv", help="CSV report filename. Default: %(default)s")
    parser.add_argument("--inter_song_delay", type=float, default=DEFAULT_INTER_SONG_DELAY, help="Delay between songs. Default: %(default)s s.")
    parser.add_argument("--spotdl_audio_format", default="mp3", help="spotdl audio format (mp3, m4a, etc.). Default: mp3")
    parser.add_argument("--spotdl_bitrate", default="192k", help="spotdl audio bitrate. Default: 192k")
    parser.add_argument("--spotdl_lyrics_providers", nargs='+', default=["musixmatch", "genius", "synced"], help="spotdl lyrics providers. Default: musixmatch genius synced")
    parser.add_argument("--ffmpeg_path", default="ffmpeg", help="Path to ffmpeg executable if not in system PATH. Default: ffmpeg")

    args = parser.parse_args()

    if not (SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET):
        print("CRITICAL: Spotify API credentials (SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET) not set in .env. Exiting.")
        sys.exit(1)
    else:
        print(f"  Found Spotify Credentials: Client ID starts with '{SPOTIPY_CLIENT_ID[:4]}...'")


    os.makedirs(args.output_dir, exist_ok=True)
    report_csv_path = os.path.join(args.output_dir, args.report_file)
    initialize_csv(report_csv_path)

    spotipy_client_for_playlists = init_spotipy_client() # For fetching playlist items via Spotipy
    if not spotipy_client_for_playlists and args.playlist_url:
        print("Failed to initialize Spotipy client for playlist fetching. Exiting."); sys.exit(1)

    spotdl_downloader_instance = None
    if SPOTDL_AVAILABLE:
        try:
            print("  spotdl: Attempting to initialize Downloader (this will also set up spotdl's Spotify client)...")
            settings = {
                "output": os.path.join(args.output_dir, "{artist} - {title}.{output-ext}"),
                "audio_format": args.spotdl_audio_format, "bitrate": args.spotdl_bitrate,
                "lyrics_providers": args.spotdl_lyrics_providers,
                "overwrite": "skip" if args.skip_existing_mp3 else "force",
                "ffmpeg": args.ffmpeg_path, "threads": os.cpu_count() or 2,
                "log_level": "INFO", 
                "headless": True, # Important: Ensures SpotifyClient is initialized for client_credentials
                "client_id": SPOTIPY_CLIENT_ID, 
                "client_secret": SPOTIPY_CLIENT_SECRET, 
                "user_auth": False, 
                "cache_path": os.path.join(args.output_dir, ".spotdl_cache"),
                "archive_file": os.path.join(args.output_dir, ".spotdl_archive.txt"),
                "simple_tui": True,
            }
            spotdl_downloader_instance = Downloader(settings=settings)
            # If Downloader initializes, it means it successfully set up its internal SpotifyClient.
            # Song.from_url() should then be able to use this client.
            print(f"  spotdl: Downloader initialized successfully. Output format: {settings['audio_format']}")
            print(f"  spotdl: Lyrics providers: {settings['lyrics_providers']}")

            try:
                import subprocess
                creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                process = subprocess.run(
                    [args.ffmpeg_path, "-version"],
                    capture_output=True, check=True, text=True,
                    creationflags=creation_flags
                )
                print(f"  FFmpeg seems to be callable at '{args.ffmpeg_path}'. Version info snippet: {process.stdout.splitlines()[0]}")
            except (subprocess.CalledProcessError, FileNotFoundError) as ffmpeg_e:
                print(f"  WARNING: FFmpeg not found at '{args.ffmpeg_path}' or it failed to execute ({ffmpeg_e}). spotdl audio conversion might fail.")
        except Exception as e_spotdl_init:
            print(f"  WARNING: Failed to initialize spotdl Downloader: {type(e_spotdl_init).__name__} - {e_spotdl_init}.")
            print(f"  This will prevent MP3 downloads and lyric fetching via spotdl.")
            spotdl_downloader_instance = None
    else:
        print("INFO: spotdl library is not installed/runnable. MP3 & Lyrics via spotdl will be SKIPPED.")


    songs_to_process: List[Dict[str, Any]] = []
    if args.playlist_url:
        print(f"Fetching tracks from Spotify playlist: {args.playlist_url}")
        if not spotipy_client_for_playlists:
             print("Spotipy client not available for playlist fetching. Exiting."); sys.exit(1)
        songs_to_process = get_tracks_from_spotify_playlist(spotipy_client_for_playlists, args.playlist_url)
        if not songs_to_process: print("No tracks from playlist or failed. Exiting."); sys.exit(1)
    elif args.song_list_file:
        if not os.path.exists(args.song_list_file): print(f"Error: Song list file not found: '{args.song_list_file}'"); sys.exit(1)
        try:
            with open(args.song_list_file, 'r', encoding='utf-8') as f:
                first_line = f.readline(); delimiter = '\t' if '\t' in first_line else ','; f.seek(0)
                reader = csv.DictReader(f, delimiter=delimiter)
                required_cols = ['title', 'artist']
                if not all(col in (reader.fieldnames or []) for col in required_cols): print(f"Error: File '{args.song_list_file}' needs 'title' & 'artist' columns."); sys.exit(1)
                for row in reader:
                    if row.get('title') and row.get('artist'):
                        song_data: Dict[str, Any] = {'title': row['title'].strip(), 'artist': row['artist'].strip()}
                        if 'spotify_uri' in row and row['spotify_uri']: song_data['spotify_uri'] = row['spotify_uri'].strip()
                        songs_to_process.append(song_data)
        except Exception as e: print(f"Error reading song list file '{args.song_list_file}': {e}"); sys.exit(1)

    if not songs_to_process: print("No songs to process."); sys.exit(0)

    print(f"Found {len(songs_to_process)} songs. Output dir: {args.output_dir}")
    if spotdl_downloader_instance:
        print(f"  MP3s & Lyrics will be attempted via spotdl. Ensure FFmpeg is installed and correctly configured.")
    else:
        print("  MP3 downloading & Lyrics via spotdl will be SKIPPED (spotdl not available or Downloader init failed).")
    print(f"Inter-song delay: {args.inter_song_delay}s.")

    for i, song_data_item in enumerate(songs_to_process):
        entry_report_data = process_song_entry(
            song_data_item, args.output_dir,
            args.skip_existing_lyrics, args.skip_existing_mp3,
            spotdl_downloader_instance
        )
        entry_report_data['status_notes'] = "; ".join(entry_report_data['status_notes'])
        append_to_csv(report_csv_path, entry_report_data)
        if i < len(songs_to_process) - 1:
            print(f"  Pausing for {args.inter_song_delay}s before next song...")
            time.sleep(args.inter_song_delay)

    print(f"\n--- All processing finished. Report at: {report_csv_path} ---")
    if not spotdl_downloader_instance:
         print("Reminder: MP3s and lyrics were NOT processed by spotdl because the Downloader failed to initialize. Place MP3s manually if needed.")
    print(f"Lyrics .txt files are in '{args.output_dir}'. Run forced alignment for .json files needed by preprocess_dataset.py.")

if __name__ == "__main__":
    main()