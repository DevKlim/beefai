from typing import List, Dict, Any, Tuple, Optional
import pyphen 
import re 
import os 
import sys 
import ctypes.util 

# Import SyllableDetail for type hinting
from beefai.utils.data_types import SyllableDetail


PHONEMIZER_AVAILABLE = False 
PHONEMIZER_BACKEND_INITIALIZED = False 
ESPEAK_NG_PATH_HINT = None 

try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend 
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer.separator import Separator
    PHONEMIZER_AVAILABLE = True 
except ImportError:
    print("Warning: 'phonemizer' library not found. Phoneme and stress generation will not be functional.")
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
        # FIX: Ensure separators are distinct if not empty.
        # phone: separator for phonemes within a syllable (can be empty if not needed)
        # syllable: separator for syllables within a word
        # word: separator for words
        try:
            self.phonemizer_separator = Separator(phone='_', syllable='|', word=' ') 
        except ValueError as e_sep:
            print(f"CRITICAL ERROR initializing phonemizer.Separator: {e_sep}")
            print("This indicates a problem with the separator characters. Using default fallback.")
            # Fallback to a known good default or simpler Separator if above fails (shouldn't with '_')
            self.phonemizer_separator = Separator(syllable='|')


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
                # Use the separator defined for the class instance if phonemize call needs it
                _ = backend_instance.phonemize(["test"], separator=self.phonemizer_separator) 
                print(f"Phonemizer EspeakBackend successfully initialized and tested for language: {self.phonemizer_lang}.")
                PHONEMIZER_BACKEND_INITIALIZED = True 
            except Exception as e:
                PHONEMIZER_BACKEND_INITIALIZED = False 
                print(f"ERROR: Phonemizer EspeakBackend for '{self.phonemizer_lang}' could NOT be initialized or tested.")
                print(f"  Error details: {e}")
                print("  Troubleshooting tips for espeak-ng issues (Ensure espeak-ng is installed and in PATH, or ESPEAK_NG_PATH is set).")

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
        if PHONEMIZER_AVAILABLE and PHONEMIZER_BACKEND_INITIALIZED:
            try:
                phonemized_word_syllables = phonemize(
                    cleaned_word,
                    language=self.phonemizer_lang,
                    backend='espeak',
                    separator=self.phonemizer_separator, 
                    strip=True, 
                    preserve_punctuation=False,
                    njobs=1
                )
                if phonemized_word_syllables and isinstance(phonemized_word_syllables, str):
                    return phonemized_word_syllables.count(self.phonemizer_separator.syllable) + 1
            except Exception:
                pass 

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

    def get_syllables_with_stress(self, word: str) -> List[Tuple[str, int]]:
        if not word: return []
        cleaned_word = re.sub(r"[^a-zA-Z0-9']", "", word.lower())
        if not cleaned_word: return []

        syllables_with_stress: List[Tuple[str, int]] = []

        if PHONEMIZER_AVAILABLE and PHONEMIZER_BACKEND_INITIALIZED:
            try:
                phonemized_output = phonemize(
                    cleaned_word,
                    language=self.phonemizer_lang, 
                    backend='espeak',
                    separator=self.phonemizer_separator, 
                    strip=False, 
                    preserve_punctuation=False,
                    njobs=1
                )

                if isinstance(phonemized_output, str) and phonemized_output.strip():
                    syllable_phoneme_groups = phonemized_output.split(self.phonemizer_separator.syllable)
                    
                    pyphen_syllables = self.syllabifier.inserted(cleaned_word).split('-') if self.syllabifier else [cleaned_word]
                    num_phon_syllables = len(syllable_phoneme_groups)
                    num_pyphen_syllables = len(pyphen_syllables)

                    for i in range(min(num_phon_syllables, num_pyphen_syllables)):
                        phon_syl = syllable_phoneme_groups[i]
                        pyphen_syl_text = pyphen_syllables[i]
                        stress = 0 
                        if 'ˈ' in phon_syl: 
                            stress = 1
                        elif 'ˌ' in phon_syl: 
                            stress = 2
                        syllables_with_stress.append((pyphen_syl_text, stress))
                    
                    if num_pyphen_syllables > num_phon_syllables:
                        for i in range(num_phon_syllables, num_pyphen_syllables):
                            syllables_with_stress.append((pyphen_syllables[i], 0))
                    
                    if syllables_with_stress: return syllables_with_stress
            except Exception as e:
                pass 

        if self.syllabifier:
            pyphen_syllables = self.syllabifier.inserted(cleaned_word).split('-')
            for i, syl_text in enumerate(pyphen_syllables):
                stress = 0 
                syllables_with_stress.append((syl_text, stress))
            return syllables_with_stress
        else: 
            return [(cleaned_word, 0)]


    def get_syllables_from_word(self, word: str) -> List[str]:
        if not word: return []
        cleaned_word = re.sub(r"[^a-zA-Z0-9']", "", word.lower())
        if not cleaned_word: return []
        if self.syllabifier:
            return self.syllabifier.inserted(cleaned_word).split('-')
        else:
            return [cleaned_word]


    def count_syllables_in_line(self, line: str) -> int:
        words = re.findall(r"[\w']+", line)
        total_syllables = 0
        for word in words:
            if word: 
                total_syllables += self.count_syllables_in_word(word)
        return total_syllables

    def get_phonemes(self, text: str, strip_stress_for_phoneme_list: bool = True) -> List[str]:
        if not PHONEMIZER_AVAILABLE or not PHONEMIZER_BACKEND_INITIALIZED: 
            if not hasattr(self, '_phonemizer_warning_printed_get_phonemes'): 
                self._phonemizer_warning_printed_get_phonemes = True 
            return []
        
        if not text.strip():
            return []
        try:
            cleaned_text = text.lower() 
            # This separator is for get_phonemes specifically, not self.phonemizer_separator
            phonemizer_get_phonemes_separator = Separator(phone=' ', word=';') 

            phonemes_str_list_or_str = phonemize(
                cleaned_text,
                language=self.phonemizer_lang,
                backend='espeak', 
                separator=phonemizer_get_phonemes_separator, 
                strip=strip_stress_for_phoneme_list, 
                preserve_punctuation=False,
                njobs=1 
            )
            
            processed_phonemes = []
            if isinstance(phonemes_str_list_or_str, str):
                words_phonemes = phonemes_str_list_or_str.split(';')
                for word_ph in words_phonemes:
                    processed_phonemes.extend(p for p in word_ph.split(' ') if p.strip())
                return processed_phonemes
            elif isinstance(phonemes_str_list_or_str, list): 
                 for item_str in phonemes_str_list_or_str:
                    if isinstance(item_str, str):
                        words_phonemes = item_str.split(';')
                        for word_ph in words_phonemes:
                            processed_phonemes.extend(p for p in word_ph.split(' ') if p.strip())
                 return processed_phonemes
            else:
                print(f"Warning: Unexpected phoneme output type from phonemizer: {type(phonemes_str_list_or_str)}. Text: '{text}'. Returning empty list.")
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
    
    test_words_for_stress = ["example", "syllabification", "apple", "computer", "today", "another", "rhythm", "extraordinary", "hello"]
    print("\nSyllable and Stress Test:")
    for word in test_words_for_stress:
        syllables_stress_info = tp.get_syllables_with_stress(word)
        print(f"  Word: '{word}'")
        for syl_text, stress_val in syllables_stress_info:
            stress_label = "Unstressed"
            if stress_val == 1: stress_label = "Primary"
            elif stress_val == 2: stress_label = "Secondary"
            print(f"    - Syllable: '{syl_text}', Stress: {stress_label} ({stress_val})")
        print(f"    Syllable count (from count_syllables_in_word): {tp.count_syllables_in_word(word)}")


    line = "This is a test sentence for syllables, an example."
    words = ["hello", "world", "beautiful", "rhythm", "example", "fire", "apple", "don't", "strength", "syllabification"]
    print("\nOriginal Syllable Counts & Breakdown:")
    for word in words:
        print(f"  Syllables in '{word}': {tp.count_syllables_in_word(word)}")
        print(f"  Syllable breakdown for '{word}': {tp.get_syllables_from_word(word)}")
    print(f"\n  Total syllables in '{line}': {tp.count_syllables_in_line(line)}")

    print("\nPhoneme Generation Test:")
    test_sentence = "Hello world, this is a test."
    phonemes_stripped = tp.get_phonemes(test_sentence, strip_stress_for_phoneme_list=True)
    phonemes_with_stress = tp.get_phonemes(test_sentence, strip_stress_for_phoneme_list=False)
    
    if PHONEMIZER_BACKEND_INITIALIZED:
        if phonemes_stripped:
            print(f"  Phonemes for '{test_sentence}' (stress stripped): {phonemes_stripped}")
        if phonemes_with_stress:
            print(f"  Phonemes for '{test_sentence}' (stress kept): {phonemes_with_stress}")
        if not phonemes_stripped and not phonemes_with_stress:
            print(f"  Phoneme generation returned empty for '{test_sentence}', though backend seemed initialized.")
    else:
        print(f"  Could not generate phonemes for '{test_sentence}' (Phonemizer backend not initialized or failed).")

    print("--- TextProcessor Test Complete ---")