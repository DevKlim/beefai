from typing import List
import pyphen 
import re 
import os 
import sys 
import ctypes.util 

PHONEMIZER_AVAILABLE = False 
PHONEMIZER_BACKEND_INITIALIZED = False # Global flag to track backend status
ESPEAK_NG_PATH_HINT = None # Stores the detected directory of espeak-ng installation

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend 
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    PHONEMIZER_AVAILABLE = True 
except ImportError:
    print("Warning: 'phonemizer' library not found. Phoneme generation will not be functional.")
    print("Please install phonemizer: pip install phonemizer")

if sys.platform == "win32" and PHONEMIZER_AVAILABLE:
    possible_paths = [
        os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "eSpeak NG"),
        os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "eSpeak NG"),
    ]
    env_path = os.environ.get("ESPEAK_NG_PATH") 
    if env_path and os.path.isdir(env_path):
        possible_paths.insert(0, env_path) 

    for path_candidate in possible_paths:
        if os.path.isdir(path_candidate):
            if os.path.exists(os.path.join(path_candidate, "espeak-ng.exe")) or \
               os.path.exists(os.path.join(path_candidate, "libespeak-ng.dll")): 
                ESPEAK_NG_PATH_HINT = path_candidate
                print(f"Found potential espeak-ng installation directory at: {ESPEAK_NG_PATH_HINT}")
                break 
    
    if not ESPEAK_NG_PATH_HINT:
        print("Could not automatically find a common espeak-ng installation directory on Windows.")
        print("If phonemizer fails due to espeak-ng issues, consider the following:")
        print("  1. Ensure eSpeak NG is installed correctly.")
        print("  2. Add the main eSpeak NG installation directory to your system PATH.")
        print("  3. Alternatively, set an environment variable 'ESPEAK_NG_PATH' pointing to this directory.")
        print("  Restart your terminal/IDE after making PATH or environment variable changes.")


class TextProcessor:
    _backend_init_attempted = False 

    def __init__(self, language: str = 'en_US', phonemizer_lang: str = 'en-us'):
        global PHONEMIZER_BACKEND_INITIALIZED 

        try:
            self.syllabifier = pyphen.Pyphen(lang=language)
        except Exception as e:
            self.syllabifier = None
            print(f"Warning: Failed to initialize pyphen for language '{language}'. Syllable counting will be naive. Error: {e}")

        self.phonemizer_lang = phonemizer_lang
        
        if PHONEMIZER_AVAILABLE and not TextProcessor._backend_init_attempted:
            TextProcessor._backend_init_attempted = True 
            
            original_path_env = os.environ.get("PATH", "")
            path_modified_for_init = False
            dll_set_successfully = False

            if ESPEAK_NG_PATH_HINT: 
                potential_dll_path = os.path.join(ESPEAK_NG_PATH_HINT, "libespeak-ng.dll")
                if os.path.exists(potential_dll_path):
                    try:
                        print(f"  Attempting to set espeak-ng library directly using: {potential_dll_path}")
                        EspeakWrapper.set_library(potential_dll_path)
                        print(f"  Successfully called EspeakWrapper.set_library() with {potential_dll_path}.")
                        dll_set_successfully = True
                    except Exception as e_set_lib:
                        print(f"  Warning: Failed to set espeak-ng library path via EspeakWrapper with '{potential_dll_path}': {e_set_lib}")
                else:
                    print(f"  Note: libespeak-ng.dll not found directly in {ESPEAK_NG_PATH_HINT}. Relying on PATH or other detection.")
            else: 
                common_dll_paths_to_try = [
                    r'C:\Program Files\eSpeak NG\libespeak-ng.dll',
                    r'C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll'
                ]
                for dll_path_try in common_dll_paths_to_try:
                    if os.path.exists(dll_path_try):
                        try:
                            print(f"  Attempting to set espeak-ng library directly using fallback path: {dll_path_try}")
                            EspeakWrapper.set_library(dll_path_try)
                            print(f"  Successfully called EspeakWrapper.set_library() with {dll_path_try}.")
                            dll_set_successfully = True
                            break 
                        except Exception as e_set_lib_fallback:
                             print(f"  Warning: Failed to set espeak-ng library path via EspeakWrapper with '{dll_path_try}': {e_set_lib_fallback}")
                if not dll_set_successfully:
                    print("  Could not set espeak-ng DLL using common hardcoded paths.")

            if sys.platform == "win32" and ESPEAK_NG_PATH_HINT:
                if ESPEAK_NG_PATH_HINT not in original_path_env.split(os.pathsep):
                    print(f"  Attempting to temporarily add {ESPEAK_NG_PATH_HINT} to PATH for EspeakBackend initialization.")
                    os.environ["PATH"] = ESPEAK_NG_PATH_HINT + os.pathsep + original_path_env
                    path_modified_for_init = True
            
            try:
                print(f"  Attempting to initialize EspeakBackend for language '{self.phonemizer_lang}' (using default language_switch)...")
                backend_instance = EspeakBackend(self.phonemizer_lang) 
                print("  EspeakBackend object created. Now attempting to phonemize 'test'...")
                _ = backend_instance.phonemize(["test"]) 
                print(f"Phonemizer EspeakBackend successfully initialized and tested for language: {self.phonemizer_lang}.")
                PHONEMIZER_BACKEND_INITIALIZED = True 
            except Exception as e:
                PHONEMIZER_BACKEND_INITIALIZED = False 
                print(f"ERROR: Phonemizer EspeakBackend for '{self.phonemizer_lang}' could NOT be initialized or tested.")
                print(f"  Error details: {e}")
                print("  Troubleshooting tips for espeak-ng issues:")
                print("  - Ensure 'espeak-ng' (recommended) is properly installed and functional from your command line.")
                print("  - Its main installation directory (containing espeak-ng.exe AND libespeak-ng.dll) SHOULD be in your system PATH.")
                print("  - Alternatively, set an environment variable 'ESPEAK_NG_PATH' pointing to this installation directory.")
                print("  - On Windows, typical paths are 'C:\\Program Files\\eSpeak NG' or 'C:\\Program Files (x86)\\eSpeak NG'.")
                print("  - Consider testing with a very simple Python script that only initializes EspeakBackend and phonemizes a word, outside of this larger project, to isolate the issue.")
                print("  - Check the versions of 'phonemizer' and 'espeak-ng'. There might be incompatibilities.")
                if not dll_set_successfully:
                    print("  Also note: The script's attempt to directly set the espeak-ng DLL path was not successful.")
            finally:
                if path_modified_for_init: 
                    os.environ["PATH"] = original_path_env
                    print(f"  Restored original PATH after EspeakBackend initialization attempt.")
        
        elif PHONEMIZER_BACKEND_INITIALIZED:
            pass 
        elif not PHONEMIZER_AVAILABLE:
             pass 


    def count_syllables_in_word(self, word: str) -> int:
        if not word:
            return 0
        cleaned_word = re.sub(r"[^a-zA-Z0-9']", "", word.lower())
        if not cleaned_word:
            return 0
        if self.syllabifier:
            hyphenated_word = self.syllabifier.inserted(cleaned_word)
            return hyphenated_word.count('-') + 1
        else: 
            vowels = "aeiouy"
            count = 0
            if cleaned_word and cleaned_word[0] in vowels: count +=1
            for index in range(1,len(cleaned_word)):
                if cleaned_word[index] in vowels and cleaned_word[index-1] not in vowels:
                    count +=1
            if cleaned_word.endswith("e") and not (cleaned_word.endswith("le") and len(cleaned_word) > 2 and cleaned_word[-3] not in vowels) : count -=1
            if cleaned_word.endswith("le") and len(cleaned_word) > 2 and cleaned_word[-3] not in vowels: count+=1
            if count <= 0 and len(cleaned_word) > 0: count = 1 
            return count

    def get_syllables_from_word(self, word: str) -> List[str]:
        """
        Splits a word into its constituent syllables using pyphen.
        Returns a list of syllable strings.
        """
        if not word:
            return []
        cleaned_word = re.sub(r"[^a-zA-Z0-9']", "", word.lower())
        if not cleaned_word:
            return []
        
        if self.syllabifier:
            # pyphen.inserted() returns 'syl-la-ble'. We split by '-'
            return self.syllabifier.inserted(cleaned_word).split('-')
        else:
            # Naive fallback if pyphen is not available: return the word as a single syllable
            # A more complex rule-based syllabifier could be implemented here if pyphen often fails.
            return [cleaned_word]


    def count_syllables_in_line(self, line: str) -> int:
        words = re.findall(r"[\w']+", line)
        total_syllables = 0
        for word in words:
            if word: 
                total_syllables += self.count_syllables_in_word(word)
        return total_syllables

    def get_phonemes(self, text: str, strip_stress: bool = True) -> List[str]:
        if not PHONEMIZER_AVAILABLE or not PHONEMIZER_BACKEND_INITIALIZED: 
            if not hasattr(self, '_phonemizer_warning_printed_get_phonemes'): 
                self._phonemizer_warning_printed_get_phonemes = True 
            return []
        
        if not text.strip():
            return []
        try:
            cleaned_text = text.lower() 
            phonemes_str_list = phonemize(
                cleaned_text,
                language=self.phonemizer_lang,
                backend='espeak', 
                separator=None, 
                strip=strip_stress,      
                preserve_punctuation=False,
                njobs=1 
            )
            
            processed_phonemes = []
            if isinstance(phonemes_str_list, list): 
                for item in phonemes_str_list:
                    if isinstance(item, str) and item.strip():
                        processed_phonemes.extend(item.split()) 
                return processed_phonemes
            elif isinstance(phonemes_str_list, str): 
                return [p for p in phonemes_str_list.split() if p.strip()]
            else:
                print(f"Warning: Unexpected phoneme output type from phonemizer: {type(phonemes_str_list)}. Text: '{text}'. Returning empty list.")
                return []

        except Exception as e:
            print(f"Error during phonemization for text '{text}': {e}")
            if "espeak" in str(e).lower() and ("not found" in str(e).lower() or "cannot open shared object" in str(e).lower() or "ailed to load" in str(e).lower()) :
                 print("  This error strongly suggests that the espeak/espeak-ng engine or its dynamic libraries are still not accessible to phonemizer.")
                 print("  Verify PATH, ESPEAK_NG_PATH, and DLL locations again.")
            return []

if __name__ == "__main__":
    print("--- TextProcessor Test ---")
    print(f"Initial status: PHONEMIZER_AVAILABLE={PHONEMIZER_AVAILABLE}, PHONEMIZER_BACKEND_INITIALIZED={PHONEMIZER_BACKEND_INITIALIZED}")
    
    tp = TextProcessor() 
    
    print(f"Status after first TextProcessor init: PHONEMIZER_BACKEND_INITIALIZED={PHONEMIZER_BACKEND_INITIALIZED}")
    
    line = "This is a test sentence for syllables, an example."
    words = ["hello", "world", "beautiful", "rhythm", "example", "fire", "apple", "don't", "strength", "syllabification"]
    print("\nSyllable Counts:")
    for word in words:
        print(f"  Syllables in '{word}': {tp.count_syllables_in_word(word)}")
        print(f"  Syllable breakdown for '{word}': {tp.get_syllables_from_word(word)}")

    print(f"\n  Total syllables in '{line}': {tp.count_syllables_in_line(line)}")

    print("\nPhoneme Generation Test:")
    test_sentence = "Hello world, this is a test."
    phonemes = tp.get_phonemes(test_sentence)
    
    if PHONEMIZER_BACKEND_INITIALIZED:
        if phonemes:
            print(f"  Phonemes for '{test_sentence}': {phonemes}")
        else:
            print(f"  Phoneme generation returned empty for '{test_sentence}', though backend seemed initialized. Check text or phonemizer behavior.")
    else:
        print(f"  Could not generate phonemes for '{test_sentence}' (Phonemizer backend not initialized or failed).")

    phonemes_contraction = tp.get_phonemes("don't")
    if PHONEMIZER_BACKEND_INITIALIZED:
        print(f"  Phonemes for 'don't': {phonemes_contraction}")
    
    print("\n--- Creating second TextProcessor instance ---")
    tp2 = TextProcessor() 
    phonemes_tp2 = tp2.get_phonemes("Another test.")
    if PHONEMIZER_BACKEND_INITIALIZED: 
        print(f"  Phonemes from tp2 for 'Another test.': {phonemes_tp2}")
    else:
        print(f"  Phoneme generation from tp2 failed (backend not initialized or previously failed).")

    print("--- TextProcessor Test Complete ---")