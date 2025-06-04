# beefai/data_processing/text_processor.py
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
                break 
    
    if not ESPEAK_NG_PATH_HINT and not PHONEMIZER_BACKEND_INITIALIZED: # Only print if not already working
        pass


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
        try:
            self.phonemizer_separator = Separator(phone='_', syllable='|', word=' ') 
        except ValueError as e_sep: # pragma: no cover
            print(f"CRITICAL ERROR initializing phonemizer.Separator: {e_sep}")
            # Fallback to a simpler separator if the complex one fails (e.g., due to phonemizer version)
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
                        EspeakWrapper.set_library(potential_dll_path)
                        dll_set_successfully = True
                    except Exception as e_set_lib: # pragma: no cover
                        pass
            else: # pragma: no cover
                common_dll_paths_to_try = [
                    r'C:\Program Files\eSpeak NG\libespeak-ng.dll',
                    r'C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll'
                ]
                for dll_path_try in common_dll_paths_to_try:
                    if os.path.exists(dll_path_try):
                        try:
                            EspeakWrapper.set_library(dll_path_try)
                            dll_set_successfully = True
                            break 
                        except Exception as e_set_lib_fallback:
                            pass

            if sys.platform == "win32" and ESPEAK_NG_PATH_HINT: # pragma: no cover
                if ESPEAK_NG_PATH_HINT not in original_path_env.split(os.pathsep):
                    os.environ["PATH"] = ESPEAK_NG_PATH_HINT + os.pathsep + original_path_env
                    path_modified_for_init = True
            
            try:
                backend_instance = EspeakBackend(self.phonemizer_lang) 
                _ = backend_instance.phonemize(["test"], separator=self.phonemizer_separator) 
                PHONEMIZER_BACKEND_INITIALIZED = True 
            except Exception as e: # pragma: no cover
                PHONEMIZER_BACKEND_INITIALIZED = False 

            finally: # pragma: no cover
                if path_modified_for_init: 
                    os.environ["PATH"] = original_path_env
        
        elif PHONEMIZER_BACKEND_INITIALIZED: # pragma: no cover
            pass 
        elif not PHONEMIZER_AVAILABLE: # pragma: no cover
             pass 

    def clean_text_for_words(self, text: str) -> str:
        """Removes punctuation and converts to lowercase, preserving intra-word apostrophes."""
        if not text:
            return ""
        # First, protect intra-word apostrophes by replacing them with a placeholder
        text_with_placeholder = re.sub(r"(\w)'(\w)", r"\1APOSPLACEHOLDER\2", text)
        # Remove all other punctuation and non-alphanumeric characters (except placeholder)
        cleaned_text = re.sub(r"[^\w\sAPOSPLACEHOLDER]", "", text_with_placeholder)
        # Restore the intra-word apostrophes
        cleaned_text = cleaned_text.replace("APOSPLACEHOLDER", "'")
        return cleaned_text.lower()

    def split_text_into_words(self, text: str) -> List[str]:
        """Cleans text and splits it into words."""
        cleaned_line = self.clean_text_for_words(text)
        # Split by whitespace and filter out empty strings that might result
        words = [word for word in cleaned_line.split() if word]
        return words

    def count_syllables_in_word(self, word: str) -> int:
        if not word:
            return 0
        # Cleaning for syllable counting: remove all non-alphanumeric except apostrophes
        cleaned_word = re.sub(r"[^\w']", "", word.lower()) 
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
                    preserve_punctuation=False, # Should be false as we pre-cleaned
                    njobs=1
                )
                if phonemized_word_syllables and isinstance(phonemized_word_syllables, str):
                    # Count based on the syllable separator defined in self.phonemizer_separator
                    return phonemized_word_syllables.count(self.phonemizer_separator.syllable) + 1
            except Exception: # pragma: no cover
                # Fall through to pyphen/naive if phonemizer fails
                pass 

        if self.syllabifier:
            hyphenated_word = self.syllabifier.inserted(cleaned_word)
            return hyphenated_word.count('-') + 1
        else: # Naive fallback
            vowels = "aeiouy"
            count = 0
            if cleaned_word and cleaned_word[0] in vowels: count +=1
            for index in range(1,len(cleaned_word)):
                if cleaned_word[index] in vowels and cleaned_word[index-1] not in vowels:
                    count +=1
            # Refined naive logic for 'e' at end
            if cleaned_word.endswith("e"):
                if len(cleaned_word) > 1 and cleaned_word[-2] not in vowels and not (len(cleaned_word) > 2 and cleaned_word[-2] == 'l' and cleaned_word[-3] in vowels): # like "able" vs "ale"
                     count -=1
            if cleaned_word.endswith("le") and len(cleaned_word) > 2 and cleaned_word[-3] not in vowels: count+=1 # for "able"
            if count <= 0 and len(cleaned_word) > 0: count = 1 
            return count

    def get_syllables_with_stress(self, word: str) -> List[Tuple[str, int]]:
        if not word: return []
        cleaned_word = re.sub(r"[^\w']", "", word.lower()) # Clean for phonemizer/pyphen
        if not cleaned_word: return []

        syllables_with_stress: List[Tuple[str, int]] = []

        if PHONEMIZER_AVAILABLE and PHONEMIZER_BACKEND_INITIALIZED:
            try:
                phonemized_output = phonemize(
                    cleaned_word,
                    language=self.phonemizer_lang, 
                    backend='espeak',
                    separator=self.phonemizer_separator, 
                    strip=False, # Keep stress marks
                    preserve_punctuation=False,
                    njobs=1
                )

                if isinstance(phonemized_output, str) and phonemized_output.strip():
                    syllable_phoneme_groups = phonemized_output.split(self.phonemizer_separator.syllable)
                    
                    # Use pyphen to get orthographic syllables
                    pyphen_syllables = self.syllabifier.inserted(cleaned_word).split('-') if self.syllabifier else [cleaned_word]
                    num_phon_syllables = len(syllable_phoneme_groups)
                    num_pyphen_syllables = len(pyphen_syllables)

                    # Align phonemic syllables (with stress) to orthographic syllables
                    for i in range(min(num_phon_syllables, num_pyphen_syllables)):
                        phon_syl = syllable_phoneme_groups[i]
                        pyphen_syl_text = pyphen_syllables[i]
                        stress = 0 
                        if 'ˈ' in phon_syl: # Primary stress marker from espeak
                            stress = 1
                        elif '��' in phon_syl: # Secondary stress marker
                            stress = 2
                        syllables_with_stress.append((pyphen_syl_text, stress))
                    
                    # If pyphen found more syllables, append them as unstressed
                    if num_pyphen_syllables > num_phon_syllables: # pragma: no cover
                        for i in range(num_phon_syllables, num_pyphen_syllables):
                            syllables_with_stress.append((pyphen_syllables[i], 0)) # Default to unstressed
                    
                    if syllables_with_stress: return syllables_with_stress
            except Exception as e: # pragma: no cover
                pass # Fall through to pyphen-only or naive

        # Fallback if phonemizer failed or not available
        if self.syllabifier:
            pyphen_syllables = self.syllabifier.inserted(cleaned_word).split('-')
            for i, syl_text in enumerate(pyphen_syllables):
                # Basic heuristic: first syllable often stressed if no other info and multiple syllables
                stress = 1 if i == 0 and len(pyphen_syllables) > 1 else 0 
                syllables_with_stress.append((syl_text, stress))
            return syllables_with_stress
        else: # Naive fallback: whole word as one unstressed syllable
            return [(cleaned_word, 0)]


    def get_syllables_from_word(self, word: str) -> List[str]:
        """
        Splits a word into orthographic syllables using pyphen.
        This is crucial for the new phonetic guide generation strategy.
        """
        if not word: return []
        cleaned_word = re.sub(r"[^\w']", "", word.lower())
        if not cleaned_word: return []
        if self.syllabifier:
            # pyphen's `inserted` method adds hyphens, then we split by them.
            hyphenated_word = self.syllabifier.inserted(cleaned_word)
            return hyphenated_word.split('-')
        else:
            # Naive fallback: return the word itself as a single syllable if pyphen fails
            return [cleaned_word]


    def count_syllables_in_line(self, line: str) -> int:
        words = self.split_text_into_words(line) # Use the new method
        total_syllables = 0
        for word in words:
            if word: # Ensure word is not empty after cleaning/splitting
                total_syllables += self.count_syllables_in_word(word)
        return total_syllables

    def get_phonemes(self, text: str, strip_stress_for_phoneme_list: bool = True) -> List[str]:
        if not PHONEMIZER_AVAILABLE or not PHONEMIZER_BACKEND_INITIALIZED:  # pragma: no cover
            if not hasattr(self, '_phonemizer_warning_printed_get_phonemes'): 
                self._phonemizer_warning_printed_get_phonemes = True 
            return []
        
        cleaned_text_for_phonemes = self.clean_text_for_words(text) # Use general cleaning
        if not cleaned_text_for_phonemes.strip():
            return []
        try:
            phonemizer_get_phonemes_separator = Separator(phone=' ', word=';') 

            phonemes_str_list_or_str = phonemize(
                cleaned_text_for_phonemes, # Use cleaned text
                language=self.phonemizer_lang,
                backend='espeak', 
                separator=phonemizer_get_phonemes_separator, 
                strip=strip_stress_for_phoneme_list, 
                preserve_punctuation=False, # Already handled by clean_text_for_words
                njobs=1 
            )
            
            processed_phonemes = []
            if isinstance(phonemes_str_list_or_str, str):
                words_phonemes = phonemes_str_list_or_str.split(';')
                for word_ph in words_phonemes:
                    processed_phonemes.extend(p for p in word_ph.split(' ') if p.strip())
                return processed_phonemes
            elif isinstance(phonemes_str_list_or_str, list):  # pragma: no cover
                 for item_str in phonemes_str_list_or_str:
                    if isinstance(item_str, str):
                        words_phonemes = item_str.split(';')
                        for word_ph in words_phonemes:
                            processed_phonemes.extend(p for p in word_ph.split(' ') if p.strip())
                 return processed_phonemes
            else: # pragma: no cover
                return []

        except Exception as e: # pragma: no cover
            if "espeak" in str(e).lower() and ("not found" in str(e).lower() or "cannot open shared object" in str(e).lower() or "ailed to load" in str(e).lower()) :
                 pass
            return []

if __name__ == "__main__": # pragma: no cover
    print("--- TextProcessor Test ---")
    
    tp = TextProcessor() 
    
    test_words_for_stress = ["example", "syllabification", "apple", "computer", "today", "another", "rhythm", "extraordinary", "hello", "don't", "it's"]
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

    line_with_punctuation = "This is a test sentence, for syllables an example; don't forget it's important!"
    print(f"\nTesting line processing for: \"{line_with_punctuation}\"")
    words_from_line = tp.split_text_into_words(line_with_punctuation)
    print(f"  Cleaned words: {words_from_line}")
    print(f"  Total syllables in line: {tp.count_syllables_in_line(line_with_punctuation)}")


    words = ["hello", "world", "beautiful", "rhythm", "example", "fire", "apple", "don't", "strength", "syllabification"]
    print("\nOriginal Syllable Counts & Breakdown:")
    for word in words:
        print(f"  Syllables in '{word}': {tp.count_syllables_in_word(word)}")
        print(f"  Syllable breakdown for '{word}' (using get_syllables_from_word): {tp.get_syllables_from_word(word)}")

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