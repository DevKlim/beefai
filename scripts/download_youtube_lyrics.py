import os
import sys
import argparse
import re
import time
import json # For yt-dlp JSON output
from yt_dlp import YoutubeDL
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from typing import Optional, List, Dict, Any # Import Optional and other types

# --- Configuration ---
REQUEST_TIMEOUT = 15  # seconds for HTTP requests
DOWNLOAD_DELAY = 2    # seconds delay between downloads/lyric fetches
LYRIC_SEARCH_RESULTS = 3 # Number of search results to check for lyrics
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# --- Helper Functions ---

def sanitize_filename(name: str) -> str:
    """Basic sanitization for filenames."""
    if not name:
        name = "unknown_title"
    name = re.sub(r'[\\/*?:"<>|]',"_", name) # Remove illegal characters
    name = name.replace("\n", "_").replace("\r", "_")
    name = re.sub(r'\s+', '_', name.strip())   # Replace whitespace with underscore
    return name[:150] # Limit length

def fetch_url_content(url: str) -> Optional[str]:
    """Fetches content from a URL with a user-agent."""
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {url}: {e}")
        return None

def extract_lyrics_azlyrics(soup: BeautifulSoup) -> Optional[str]:
    """Extracts lyrics specifically from an AZLyrics page structure."""
    lyrics_div = None
    main_col = soup.find('div', class_='col-xs-12 col-lg-8 text-center')
    if main_col:
        divs = main_col.find_all('div', recursive=False) 
        if len(divs) > 1: # AZLyrics usually has multiple divs here
            # The lyrics are typically in a div that's not the first few (which are ads/titles)
            # and not the last few (which are also boilerplate or links)
            # Heuristic: find the largest div after skipping a few.
            # This is very fragile. A more robust way involves looking for comments like <!-- Usage of azlyrics.com... -->
            # and taking the div immediately preceding it, or the one after <!-- MxM banner -->.
            
            potential_lyrics_text = ""
            # Search for a div that typically holds lyrics. It's usually unstyled.
            # It often follows the "ringtone" div and precedes the "usage" comment.
            
            # Attempt to find the div directly, assuming it's not the first one.
            # Let's iterate and look for a div with lots of <br> or many lines.
            for i, d in enumerate(divs):
                if i < 2: continue # Skip first few divs usually (title, ads)
                if '<!--' in str(d) and '-->' in str(d): continue
                if d.find('form') or d.find('button') or d.find('script'): continue
                
                current_text = d.get_text(separator='\n').strip()
                if current_text.count('\n') > 5 and len(current_text) > len(potential_lyrics_text): # More than 5 line breaks
                    potential_lyrics_text = current_text
            
            if potential_lyrics_text:
                # Clean up common AZLyrics boilerplate
                lyrics_text = re.sub(r"^\s*\[.*?\]\s*\n", "", potential_lyrics_text, flags=re.MULTILINE) # Remove [Chorus], [Verse], etc.
                lyrics_text = re.sub(r"Thanks to .* for correcting these lyrics\n?", "", lyrics_text, flags=re.IGNORECASE)
                lyrics_text = re.sub(r"Writer\(s\):.*?\n?", "", lyrics_text, flags=re.IGNORECASE)
                lyrics_text = re.sub(r"Submit Corrections\n?", "", lyrics_text, flags=re.IGNORECASE)
                lyrics_text = re.sub(r"Search.+Lyrics", "", lyrics_text, flags=re.IGNORECASE) # Top search bar text
                if "Unfortunately, we are not licensed to display the full lyrics for this song at the moment." in lyrics_text:
                    return None
                return lyrics_text.strip()
    return None


def extract_lyrics_genius(soup: BeautifulSoup) -> Optional[str]:
    """Extracts lyrics specifically from a Genius page structure."""
    lyrics_containers = soup.find_all('div', attrs={"data-lyrics-container": "true"})
    
    if not lyrics_containers:
        # Fallback for newer Genius structures (observed Jan 2024)
        lyrics_elements = soup.select('div[class^="Lyrics__Container"], div[class*="LyricsGaps__Container"]')
        # Genius HTML is complex, let's try to get all text from these elements and join
        if lyrics_elements:
            all_lyrics_parts = []
            for elem in lyrics_elements:
                # Genius uses <br> for line breaks within its containers
                # We want to preserve these.
                for br in elem.find_all("br"):
                    br.replace_with("\n")
                all_lyrics_parts.append(elem.get_text().strip())
            if all_lyrics_parts:
                return "\n".join(all_lyrics_parts).strip()
        return None # If specific selectors not found

    # Original logic for data-lyrics-container
    if lyrics_containers:
        all_lyrics_parts = []
        for container in lyrics_containers:
            for br in container.find_all("br"): # Replace <br> with newlines before get_text
                br.replace_with("\n")
            lyrics_text = container.get_text().strip() # get_text then strips leading/trailing whitespace
            all_lyrics_parts.append(lyrics_text)

        if all_lyrics_parts:
            return "\n\n".join(all_lyrics_parts).strip() # Join parts with double newlines
    return None


def extract_lyrics_generic(soup: BeautifulSoup) -> Optional[str]:
    """A more generic attempt to extract lyrics if specific site parsers fail."""
    best_candidate_text = ""

    # Try looking for elements with classes or IDs that often contain lyrics
    common_selectors = [
        'div[class*="lyrics"]', 'div[id*="lyrics"]',
        'p[class*="lyrics"]', 'td[class*="lyrics"]',
        'div.lyric-body', 'div.song-lyrics', 'div.lyrics_container'
    ]
    for selector in common_selectors:
        elements = soup.select(selector)
        for elem in elements:
            text = elem.get_text(separator='\n', strip=True)
            if text.count('\n') > 3 and len(text) > 50 : # Basic check
                return text # Return first good candidate from common selectors

    # Fallback to broader search if specific selectors fail
    potential_elements = soup.find_all(['div', 'p', 'pre', 'td', 'article'])

    for elem in potential_elements:
        if elem.name in ['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'button', 'input']:
            continue
        if any(parent.name in ['script', 'style', 'nav', 'header', 'footer', 'aside', 'form'] for parent in elem.parents):
            continue
        if 'comment' in str(elem).lower() or 'reply' in str(elem).lower() or 'share' in str(elem).lower() or 'advert' in str(elem).lower():
            continue
        if elem.find(['script', 'style', 'nav', 'header', 'footer', 'form', 'button', 'input']): # Contains unwanted sub-elements
            continue

        text = elem.get_text(separator='\n', strip=True)
        
        # Heuristics for lyrics-like content
        line_breaks = text.count('\n')
        text_length = len(text)
        avg_line_length = text_length / (line_breaks + 1)

        if line_breaks > 4 and text_length > 100 and avg_line_length < 80 : # Many lines, decent length, not overly long lines (like paragraphs)
            if len(text) > len(best_candidate_text):
                 # Check for common non-lyric keywords
                lower_text = text.lower()
                if not any(kw in lower_text for kw in ["advertisement", "related links", "cookie policy", "javascript", "newsletter", "article content", "terms of use", "privacy policy"]):
                    best_candidate_text = text
    
    if best_candidate_text:
        lines = [line.strip() for line in best_candidate_text.split('\n') if line.strip()]
        cleaned_lines = [line for line in lines if len(line) > 1 and not line.lower().startswith(("lyrics for", "artist:", "album:", "song:", "powered by", "track list", "search result"))]
        if len(cleaned_lines) > 4: 
            return "\n".join(cleaned_lines)
    return None


def search_and_extract_lyrics(song_title: str) -> Optional[str]:
    """Searches for lyrics and tries to extract them."""
    print(f"  Searching lyrics for: {song_title}")
    if not song_title:
        return None

    try:
        search_query = f"{song_title} lyrics"
        with DDGS() as ddgs:
            search_results = list(ddgs.text(search_query, max_results=LYRIC_SEARCH_RESULTS))

        if not search_results:
            print(f"  No search results found for '{song_title}'.")
            return None

        for i, result in enumerate(search_results):
            url = result['href']
            print(f"  Trying lyric URL ({i+1}/{LYRIC_SEARCH_RESULTS}): {url}")
            
            html_content = fetch_url_content(url)
            if not html_content:
                time.sleep(DOWNLOAD_DELAY / 2) 
                continue

            soup = BeautifulSoup(html_content, 'html.parser')
            lyrics = None

            if "azlyrics.com" in url:
                lyrics = extract_lyrics_azlyrics(soup)
            elif "genius.com" in url:
                lyrics = extract_lyrics_genius(soup)
            
            if not lyrics: 
                lyrics = extract_lyrics_generic(soup)

            if lyrics and len(lyrics.strip()) > 50: 
                print(f"  Lyrics found from {url}")
                return lyrics.strip()
            else:
                print(f"  Could not extract meaningful lyrics from {url}")
            
            time.sleep(DOWNLOAD_DELAY / 2)

    except Exception as e:
        print(f"  An error occurred during lyric search for '{song_title}': {e}")
    
    print(f"  Could not find lyrics for '{song_title}' after trying {LYRIC_SEARCH_RESULTS} sources.")
    return None


def process_video_entry(video_info: Dict[str, Any], output_dir: str, skip_existing: bool):
    """Downloads audio and fetches lyrics for a single video entry."""
    video_url = video_info.get('webpage_url') or video_info.get('original_url') or video_info.get('url')
    if not video_url:
        print("  Could not determine video URL from metadata. Skipping entry.")
        return

    original_title = video_info.get('title', 'Unknown_Title')
    
    base_filename_sanitized = sanitize_filename(original_title)
    
    mp3_filename = f"{base_filename_sanitized}.mp3"
    txt_filename = f"{base_filename_sanitized}.txt"
    
    mp3_filepath = os.path.join(output_dir, mp3_filename)
    txt_filepath = os.path.join(output_dir, txt_filename)

    print(f"\nProcessing: {original_title} (URL: {video_url})")
    print(f"  MP3 target: {mp3_filepath}")
    print(f"  TXT target: {txt_filepath}")

    audio_downloaded_successfully = False
    if skip_existing and os.path.exists(mp3_filepath) and os.path.getsize(mp3_filepath) > 1000:
        print(f"  MP3 already exists: {mp3_filepath}. Skipping download.")
        audio_downloaded_successfully = True
    else:
        ydl_opts_single = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192', # Standard quality
            }],
            'outtmpl': mp3_filepath, 
            'quiet': False, 
            'ignoreerrors': True,
            'restrictfilenames': False, 
            'overwrites': not skip_existing,
            'nocheckcertificate': True, # Add this if SSL errors occur
        }
        try:
            with YoutubeDL(ydl_opts_single) as ydl_single:
                error_code = ydl_single.download([video_url])
                if error_code == 0 and os.path.exists(mp3_filepath) and os.path.getsize(mp3_filepath) > 1000:
                    print(f"  Successfully downloaded and converted audio to: {mp3_filepath}")
                    audio_downloaded_successfully = True
                else:
                    print(f"  yt-dlp potentially failed to download or convert {video_url} (error code: {error_code}).")
                    if os.path.exists(mp3_filepath) and os.path.getsize(mp3_filepath) < 1000:
                        print("  Downloaded file is too small, likely an error page or incomplete.")
                        try: os.remove(mp3_filepath)
                        except: pass
        except Exception as e:
            print(f"  Exception during audio download for {video_url}: {e}")
    
    if not audio_downloaded_successfully:
        print(f"  Skipping lyrics for '{original_title}' due to audio download failure.")
        return

    if skip_existing and os.path.exists(txt_filepath) and os.path.getsize(txt_filepath) > 10:
        print(f"  Lyrics file already exists: {txt_filepath}. Skipping fetch.")
    else:
        lyrics_content = search_and_extract_lyrics(original_title)
        if lyrics_content:
            try:
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(lyrics_content)
                print(f"  Lyrics saved to: {txt_filepath}")
            except IOError as e:
                print(f"  Error saving lyrics to {txt_filepath}: {e}")
        else:
            print(f"  No lyrics found or extracted for '{original_title}'. Creating empty/placeholder TXT.")
            try:
                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"[Lyrics for '{original_title}' not found or failed to extract.]")
            except IOError as e:
                print(f"  Error saving placeholder lyrics text to {txt_filepath}: {e}")
    
    time.sleep(DOWNLOAD_DELAY)


def main():
    parser = argparse.ArgumentParser(description="Download YouTube audio and fetch lyrics.")
    parser.add_argument("url_file", help="Path to a text file containing YouTube video/playlist URLs (one per line).")
    parser.add_argument("-o", "--output_dir", default="downloaded_songs", help="Directory to save downloaded MP3s and lyrics TXTs.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip downloading/fetching if files already exist.")
    
    args = parser.parse_args()

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

    # Common YDL options for extracting info
    # `dumpjson` is problematic with the Python API for getting structured data sometimes.
    # It's better to let `extract_info` return a Python dictionary directly.
    ydl_opts_info = {
        'quiet': True,
        'ignoreerrors': True,
        'extract_flat': False, # Get full info, including entries for playlists
        # 'dumpjson': True, # REMOVE THIS - we want Python dicts, not JSON strings
        'skip_download': True, # Ensure we only fetch metadata
        'nocheckcertificate': True,
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
                if 'entries' in result_info_data: # It's a playlist
                    actual_videos = [entry for entry in result_info_data['entries'] if entry] # Filter out None entries
                else: # It's a single video
                    actual_videos = [result_info_data]
                
                if not actual_videos:
                    print(f"  No video items found for URL/playlist: {url_or_playlist}")
                    continue
                
                print(f"  Found {len(actual_videos)} video(s) to process from this source.")
                for vid_idx, video_data in enumerate(actual_videos):
                    if not video_data: # Should be filtered by list comprehension above, but double check
                        print(f"  Skipping null video data at index {vid_idx} for {url_or_playlist}")
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