import os
import random
from typing import List, Dict, Optional, Tuple, Any
import re

from dotenv import load_dotenv
import google.generativeai as genai # type: ignore

from beefai.utils.data_types import FlowData, FlowDatum 
from beefai.lyric_generation.prompt_formatter import PromptFormatter
from beefai.data_processing.text_processor import TextProcessor

# Load environment variables from .env file
load_dotenv()

LOG_PREFIX_AGENT = "[LyricAgent]"
LYRIC_GENERATION_FAILED_MARKER = "<<<LYRIC_GENERATION_FAILED_FOR_BATCH>>>"
DEFAULT_PHONETIC_STUB_SYLLABLE = "uh" # Used if auto-adjustment needs to pad


class LyricAgent:
    def __init__(self, llm_model_name: str = "gemini-1.5-flash-latest", llm_api_key: Optional[str] = None):
        # Allow overriding the model name via environment variable
        self.llm_model_name = os.getenv("LYRIC_LLM_MODEL_NAME", llm_model_name) 

        self.api_key = llm_api_key or os.getenv("GEMINI_API_KEY")
        self.prompt_formatter = PromptFormatter()
        self.text_processor = TextProcessor() # For syllable counting and phonetic breakdown
        
        self._stub_vocab = self._build_stub_vocab()

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Attempt to get the model to verify it exists, otherwise proceed with a warning.
                try:
                    _ = genai.get_model(f"models/{self.llm_model_name}")
                    self.llm_client = genai.GenerativeModel(self.llm_model_name)
                    print(f"{LOG_PREFIX_AGENT} Initialized with LLM: {self.llm_model_name}.")
                except Exception as model_e:
                    print(f"{LOG_PREFIX_AGENT} Warning: Could not verify LLM model '{self.llm_model_name}': {model_e}.")
                    print(f"{LOG_PREFIX_AGENT} Will attempt to use it, but it might fail if unavailable. Consider 'gemini-1.5-flash-latest'.")
                    self.llm_client = genai.GenerativeModel(self.llm_model_name) # Try anyway
            except Exception as e:
                print(f"{LOG_PREFIX_AGENT} Error initializing Gemini client with model '{self.llm_model_name}': {e}. Operating in STUB mode for API calls.")
                self.llm_client = None
        else:
            print(f"{LOG_PREFIX_AGENT} GEMINI_API_KEY not found. Operating in STUB mode for API calls.")
            self.llm_client = None

    def _build_stub_vocab(self) -> List[Dict[str, any]]:
        # This remains useful for generating stub lyrics
        words = [ "yo", "ai", "bot", "flow", "code", "rhyme", "time", "beat", "mic", "cool", "yeah", "model", "battle", "lyric", "track", "sound", "smart", "fast", "deep", "learn", "response", "generate", "system", "intellect", "algorithm", "network", "computer", "digital", "processor", "interface", "component", "error", "crash", "bug", "artificial", "intelligence", "computation", "understanding", "unbeatable", "superior" ]
        stub_vocab = []
        for word in words: stub_vocab.append({"text": word, "syllables": self.text_processor.count_syllables_in_word(word)})
        phrases = [ ("check the mic",3), ("one two",2), ("machine learning",4), ("neural network",4), ("rap battle bot",4), ("error code",3), ("system crash",3), ("you are weak",3), ("i am strong",3), ]
        for phrase, syllables in phrases: stub_vocab.append({"text": phrase, "syllables": syllables})
        return stub_vocab

    def _is_orthographic_syllable_count_acceptable(self, actual_syllables: int, target_syllables: int, strict_for_zero_target: bool = True) -> bool:
        """
        Checks if the actual orthographic syllable count from LLM is acceptable.
        Allows a small leeway, but can be strict for 0-syllable targets.
        For LLM output, we want it to be very close to the target orthographic syllable count.
        """
        if target_syllables == 0:
            return actual_syllables == 0 if strict_for_zero_target else True # Be strict for 0-syl lines
        
        # For LLM output, we want it to be very close to the target orthographic syllable count.
        # With retries, we can afford to be stricter.
        leeway = 0 # Default to strict adherence
        # For longer lines, allow a small leeway, e.g., +/- 1.
        if target_syllables > 5: leeway = 1 
        
        return abs(actual_syllables - target_syllables) <= leeway

    def _generate_stub_lyric_line(self, target_ortho_syllables: int, diss_keywords: List[str]) -> str:
        """Generates a stub lyric line targeting a specific orthographic syllable count."""
        if target_ortho_syllables <= 0: return ""
        
        line_words = []; current_syllables = 0
        # Try to include a diss keyword if appropriate
        if diss_keywords and random.random() < 0.6:
            keyword_to_try = random.choice(diss_keywords)
            keyword_syl = self.text_processor.count_syllables_in_line(keyword_to_try) # Orthographic count
            if keyword_syl > 0 and keyword_syl <= target_ortho_syllables:
                line_words.append(keyword_to_try)
                current_syllables += keyword_syl
        
        attempts = 0; max_attempts = target_ortho_syllables * 6 # Max attempts to fill syllables
        while current_syllables < target_ortho_syllables and attempts < max_attempts:
            attempts += 1
            remaining_syllables = target_ortho_syllables - current_syllables
            # Filter stub_vocab for items that fit remaining syllables
            possible_items = [item for item in self._stub_vocab if item["syllables"] > 0 and item["syllables"] <= remaining_syllables]
            if not possible_items:
                # If nothing fits exactly, try to find the smallest available item
                smallest_available = sorted([item for item in self._stub_vocab if item["syllables"] > 0], key=lambda x: x["syllables"])
                if not smallest_available: break # No words in vocab, stop
                possible_items = [smallest_available[0]] # Take the smallest if nothing fits perfectly

            chosen_item = random.choice(possible_items)
            line_words.append(chosen_item["text"])
            current_syllables += chosen_item["syllables"]
        
        final_line_text = " ".join(line_words)
        # Final check: if stub generation is still too far off, use simple "la"s
        actual_stub_syls = self.text_processor.count_syllables_in_line(final_line_text)
        
        # If the generated stub is not acceptable by orthographic count, fallback
        # Use a stricter check here for the stub itself.
        if not self._is_orthographic_syllable_count_acceptable(actual_stub_syls, target_ortho_syllables, strict_for_zero_target=True) and target_ortho_syllables > 0:
             # Fallback to simple 'la's to match syllable count if complex stubbing fails
             # 'la' is one syllable.
             num_las = max(1, target_ortho_syllables) 
             final_line_text = " ".join(["la"] * num_las)
        
        return final_line_text

    def _parse_llm_lyric_response_lines(
        self, 
        llm_text_response: str, 
        flow_data_for_batch: List[FlowDatum] # Used for expected number of lines and target syllables for stubs
    ) -> List[Dict[str, Any]]:
        """
        Parses LLM response (now expected to be just lyric lines, one per line from the prompt).
        Returns a list of dicts: [{"lyric": str, "target_syllables": int, "original_flow_datum_index": int}, ...]
        """
        cleaned_response = llm_text_response.strip()
        # Remove markdown code blocks if present (e.g., ```json ... ``` or ``` ... ```)
        cleaned_response = re.sub(r"^\s*```(?:json|text|markdown|[\w-]+)?\s*\n|\n\s*```\s*$", "", cleaned_response, flags=re.DOTALL|re.IGNORECASE).strip()
        
        # Split by newline, remove empty lines, and strip each line
        raw_llm_lines = [line.strip(" \r\n") for line in cleaned_response.split('\n') if line.strip(" \r\n")]
        
        parsed_lyric_data_list: List[Dict[str, Any]] = []
        num_expected_lines = len(flow_data_for_batch)

        for i in range(num_expected_lines):
            # Get target syllables and original index from the corresponding FlowDatum
            target_ortho_syl_count = flow_data_for_batch[i]['syllables']
            original_flow_datum_index = flow_data_for_batch[i].get('original_flow_datum_index', i) # Use provided index or current loop index
            
            lyric_text = ""
            
            if i < len(raw_llm_lines):
                line_from_llm = raw_llm_lines[i]
                # Basic cleaning: remove common list markers (e.g., "- ", "1. "), leading/trailing quotes if they wrap the whole line
                temp_line = re.sub(r"^\s*[-*]\s*", "", line_from_llm) 
                temp_line = re.sub(r"^\s*\d+\.\s*", "", temp_line)
                if (temp_line.startswith('"') and temp_line.endswith('"')) or \
                   (temp_line.startswith("'") and temp_line.endswith("'")):
                    if len(temp_line) > 1 : temp_line = temp_line[1:-1] # Remove quotes
                    else: temp_line = "" # Edge case of empty quoted string like "" or ''
                
                # Robustness: Check for and remove accidental old phonetic guide separators if LLM outputs them
                # The prompt should explicitly tell the LLM *not* to include these.
                if "|||PHONETIC_GUIDE_INTERNAL|||" in temp_line:
                    parts = temp_line.split("|||PHONETIC_GUIDE_INTERNAL|||", 1)
                    lyric_text = parts[0].strip() # Lyric should be first if this separator present
                    print(f"{LOG_PREFIX_AGENT}   Warning: Line {i+1} from LLM contained internal separator '|||PHONETIC_GUIDE_INTERNAL|||'. Extracted lyric: '{lyric_text}'. The part after was: '{parts[1]}'")
                elif "|||PHONETIC|||" in temp_line: # Check against old public marker
                    parts = temp_line.split("|||PHONETIC|||", 1)
                    lyric_text = parts[0].strip() # Lyric should be first
                    print(f"{LOG_PREFIX_AGENT}   Warning: Line {i+1} from LLM contained old '|||PHONETIC|||' separator. Extracted lyric: '{lyric_text}'. The part after was: '{parts[1]}'")
                else:
                    lyric_text = temp_line.strip()

                # Validate orthographic syllable count of the LLM's lyric
                actual_ortho_syls = self.text_processor.count_syllables_in_line(lyric_text)
                if not self._is_orthographic_syllable_count_acceptable(actual_ortho_syls, target_ortho_syl_count):
                    print(f"{LOG_PREFIX_AGENT}   Warning: Line {i+1} (orig_idx {original_flow_datum_index}) lyric from LLM \"{lyric_text}\" has {actual_ortho_syls} orthographic syllables, target was {target_ortho_syl_count}. This line will be marked for retry if possible.")
                
                if target_ortho_syl_count == 0 and lyric_text: # LLM provided text for a 0-syl line
                    print(f"{LOG_PREFIX_AGENT}   Warning: Line {i+1} (orig_idx {original_flow_datum_index}) (target 0 syl): LLM lyric '{lyric_text}' not empty. Forcing empty lyric.")
                    lyric_text = ""
            
            else: # Fewer lines from LLM than expected for this batch
                print(f"{LOG_PREFIX_AGENT}   Warning: LLM returned fewer lines than expected ({len(raw_llm_lines)} vs {num_expected_lines}). Line {i+1} (orig_idx {original_flow_datum_index}) missing, will use stub or mark for retry.")
                lyric_text = self._generate_stub_lyric_line(target_ortho_syl_count, []) # diss_keywords not available here

            parsed_lyric_data_list.append({"lyric": lyric_text, "target_syllables": target_ortho_syl_count, "original_flow_datum_index": original_flow_datum_index})
        
        return parsed_lyric_data_list

    def _construct_api_prompt(self, diss_text: str, flow_data_for_prompt: List[FlowDatum], bpm: Optional[float], theme: Optional[str], previous_lines_context: Optional[List[str]], is_retry: bool = False, retry_indices: Optional[List[int]] = None) -> str:
        """Constructs the prompt for the LLM API call."""
        persona = "a confident, witty, and cleverly boastful AI rapper named BeefAI"
        prompt_lines = [
            f"You are {persona}. Your task is to generate rap lyrics.",
            f"Respond to the opponent's challenge: \"{diss_text}\"",
        ]
        if theme: prompt_lines.append(f"Your response theme: {theme}")

        if is_retry and retry_indices:
            prompt_lines.append("\nCRITICAL RETRY: Previous attempt failed syllable counts for some lines. Regenerate ONLY the lines listed below, strictly adhering to the new Target Orthographic Syllable Count for each.")
            prompt_lines.append("Output ONLY the lyric text for these lines, one per line, with no numbering or bullet points.")
        else:
            prompt_lines.append("\nInstructions:")
            prompt_lines.append("- Generate ONLY the lyric text for EACH of the target lines below.")
            prompt_lines.append("- Each generated lyric MUST STRICTLY match its 'Target Orthographic Syllable Count'.")
            prompt_lines.append("- Use no punctuation except apostrophes in contractions (e.g., don't, it's).")
            prompt_lines.append("- Ensure lines are coherent and form sensible phrases/sentences.")
            prompt_lines.append("- Output each lyric on a new line, with no numbering or bullet points.")
            prompt_lines.append("- DO NOT include any phonetic guides or special markers like '|||PHONETIC|||'.")
        
        if previous_lines_context and not is_retry: # Context might be less relevant for targeted retries
            prompt_lines.append("\nPreviously generated lines (for lyrical context - maintain rhyme/theme):")
            for line_ctx in previous_lines_context: prompt_lines.append(f"  - \"{line_ctx}\"")
        
        prompt_lines.append("\nTarget Lines to Generate:")
        
        line_num_for_prompt = 1 # This is the line number presented to the LLM in the prompt
        for idx, fd_item in enumerate(flow_data_for_prompt):
            # If this is a retry, only include lines that are explicitly marked for retry
            if is_retry and retry_indices is not None and idx not in retry_indices:
                continue 

            syl_target = fd_item['syllables']
            dur_beats = fd_item.get('duration_beats', 0)
            
            line_desc = f"  - Line {line_num_for_prompt}: Target Orthographic Syllables: {syl_target}."
            if bpm and dur_beats > 0:
                syl_per_beat = syl_target / dur_beats if dur_beats > 0 else 0
                density_desc = ""
                if syl_target == 0: density_desc = "(Instrumental break - output an EMPTY line)"
                elif syl_per_beat < 1.5 : density_desc = "(Pacing: slow, sparse)"
                elif syl_per_beat < 2.5 : density_desc = "(Pacing: moderate)"
                elif syl_per_beat < 3.5 : density_desc = "(Pacing: fast, dense)"
                else: density_desc = "(Pacing: very fast, very dense)"
                line_desc += f" {density_desc}"
            elif syl_target == 0:
                 line_desc += " (Instrumental break - output an EMPTY line)"

            prompt_lines.append(line_desc)
            line_num_for_prompt +=1
            
        return "\n".join(prompt_lines)

    def generate_response_verse_via_api(
        self,
        diss_text: str,
        flow_data_batch: List[FlowDatum], # FlowData contains target orthographic syllable counts per line
        bpm: Optional[float] = None,
        theme: Optional[str] = None,
        previous_lines_context: Optional[List[str]] = None,
        max_retries_api: int = 2 # Max automated retries for syllable count issues or API errors
    ) -> List[Dict[str, Any]]: 
        """
        Generates lyric lines from LLM, with retries for syllable count adherence.
        Returns List of Dictionaries: [{"lyric": str, "target_syllables": int, "original_flow_datum_index": int}, ...]
        The "target_syllables" is the orthographic syllable count the LLM was asked to produce for that lyric.
        """
        num_lines_in_batch = len(flow_data_batch)
        print(f"{LOG_PREFIX_AGENT} Generating lyrics for a batch of {num_lines_in_batch} flow lines using LLM: {self.llm_model_name}.")
        
        diss_keywords_for_stub = [word.lower() for word in diss_text.split() if len(word) > 3 and word.lower() not in ["the", "a", "is", "and", "your", "you", "are", "was", "were", "this", "that"]]

        if not self.llm_client:
            print(f"{LOG_PREFIX_AGENT} LLM client not available. Falling back to stubs for all lines.")
            # Add original_flow_datum_index to stubs for consistent output structure
            return [{"lyric": self._generate_stub_lyric_line(fd['syllables'], diss_keywords_for_stub), "target_syllables": fd['syllables'], "original_flow_datum_index": fd.get("original_flow_datum_index", i)} for i, fd in enumerate(flow_data_batch)]

        # Use a dictionary to store successfully generated lines, keyed by their original index
        all_generated_lines_final: Dict[int, Dict[str, Any]] = {} 

        # Prepare flow_data_batch with original_flow_datum_index if not already present
        # This ensures consistent tracking across retries.
        for i, fd_item in enumerate(flow_data_batch):
            if 'original_flow_datum_index' not in fd_item:
                fd_item['original_flow_datum_index'] = i

        for attempt_num in range(max_retries_api + 1):
            is_retry_attempt = attempt_num > 0
            
            # Determine which lines still need generation (i.e., not yet successfully generated)
            lines_needing_generation_flow_data = [
                fd_item for fd_item in flow_data_batch 
                if fd_item['original_flow_datum_index'] not in all_generated_lines_final
            ]
            
            if not lines_needing_generation_flow_data:
                print(f"{LOG_PREFIX_AGENT} All lines successfully generated after {attempt_num} attempts. Breaking retry loop.")
                break # All lines are done

            # Create a list of indices (relative to lines_needing_generation_flow_data) for the prompt
            # This is only relevant for the prompt's internal numbering, not for tracking.
            retry_indices_for_prompt = list(range(len(lines_needing_generation_flow_data))) if is_retry_attempt else None

            prompt_text = self._construct_api_prompt(
                diss_text, 
                lines_needing_generation_flow_data, # Pass only the lines that need generation
                bpm, 
                theme, 
                previous_lines_context if not is_retry_attempt else None, # Only pass context on first attempt
                is_retry=is_retry_attempt, 
                retry_indices=retry_indices_for_prompt
            )
            
            print(f"{LOG_PREFIX_AGENT} Calling Gemini API (Attempt {attempt_num + 1}/{max_retries_api + 1}) for {len(lines_needing_generation_flow_data)} remaining lines...")
            # if attempt_num == 0: print(f"{LOG_PREFIX_AGENT} Prompt: \n{prompt_text[:500]}...\n") # Log initial prompt snippet

            try:
                generation_config_dict = { "temperature": 0.75, "top_p": 0.95, "top_k": 40, "max_output_tokens": 2048 } 
                
                safety_settings=[ # Stricter safety for rap battle context
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_LOW_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_LOW_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]

                response = self.llm_client.generate_content(
                    prompt_text, 
                    generation_config=genai.types.GenerationConfig(**generation_config_dict),
                    safety_settings=safety_settings
                )
                
                if not response.candidates or not response.candidates[0].content.parts:
                    block_reason_msg = "Response candidates/parts empty."
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        block_reason_msg = f"Prompt Feedback: Block Reason '{response.prompt_feedback.block_reason.name}', Message: '{response.prompt_feedback.block_reason_message}'"
                    if response.candidates and response.candidates[0].finish_reason: 
                        block_reason_msg += f" Finish Reason: '{response.candidates[0].finish_reason.name}'."
                    if response.candidates and response.candidates[0].safety_ratings: 
                        block_reason_msg += f" Safety Ratings: {[(sr.category.name, sr.probability.name) for sr in response.candidates[0].safety_ratings]}."
                    print(f"{LOG_PREFIX_AGENT}   LLM response blocked/empty (Attempt {attempt_num+1}). Reason: {block_reason_msg}")
                    continue # Try next attempt

                llm_response_text = response.text
                # Parse the response using the 'lines_needing_generation_flow_data' to get correct targets and original indices
                parsed_lines_from_attempt = self._parse_llm_lyric_response_lines(llm_response_text, lines_needing_generation_flow_data) 
                
                # Validate and collect successful lines from this attempt
                newly_successful_lines_this_attempt = 0
                for parsed_line_data in parsed_lines_from_attempt:
                    original_idx = parsed_line_data["original_flow_datum_index"]
                    # Only process if this line hasn't been successfully generated yet
                    if original_idx not in all_generated_lines_final:
                        actual_syls = self.text_processor.count_syllables_in_line(parsed_line_data["lyric"])
                        if self._is_orthographic_syllable_count_acceptable(actual_syls, parsed_line_data["target_syllables"]):
                            all_generated_lines_final[original_idx] = parsed_line_data
                            newly_successful_lines_this_attempt += 1
                        else:
                            print(f"{LOG_PREFIX_AGENT}   Line (orig_idx {original_idx}) \"{parsed_line_data['lyric'][:30]}...\" failed syllable count: actual {actual_syls}, target {parsed_line_data['target_syllables']}. Will retry if possible.")
                
                print(f"{LOG_PREFIX_AGENT}   Attempt {attempt_num+1}: Successfully generated {newly_successful_lines_this_attempt} new lines.")

            except Exception as e:
                print(f"{LOG_PREFIX_AGENT} Error during LLM call for batch (attempt {attempt_num+1}/{max_retries_api + 1}): {type(e).__name__} - {e}")
                if "SAFETY" in str(e).upper() or "block_reason" in str(e).lower() or "blocked" in str(e).lower():
                     print(f"{LOG_PREFIX_AGENT}   Safety filter or other block likely triggered. Detail: {e}")
                # Continue to next attempt

        # After all attempts, assemble the final list, using stubs for any remaining failures
        final_result_list = []
        for i, original_fd_item in enumerate(flow_data_batch):
            original_flow_datum_idx = original_fd_item['original_flow_datum_index'] # Guaranteed to exist now
            if original_flow_datum_idx in all_generated_lines_final:
                final_result_list.append(all_generated_lines_final[original_flow_datum_idx])
            else:
                print(f"{LOG_PREFIX_AGENT} Line (orig_idx {original_flow_datum_idx}) failed all API attempts. Using stub.")
                target_syl = original_fd_item['syllables']
                stub_lyric = self._generate_stub_lyric_line(target_syl, diss_keywords_for_stub)
                final_result_list.append({"lyric": stub_lyric, "target_syllables": target_syl, "original_flow_datum_index": original_flow_datum_idx})
        
        # Sort the final list by original_flow_datum_index to maintain original order
        final_result_list.sort(key=lambda x: x['original_flow_datum_index'])
        return final_result_list

    def generate_response_verse(
        self,
        diss_text: str,
        flow_data: FlowData, # FlowData contains target orthographic syllable counts per line
        bpm: Optional[float] = None,
        theme: Optional[str] = None,
        previous_lines_context: Optional[List[str]] = None,
        max_retries_api: int = 2 # Max automated retries for syllable count issues
    ) -> List[Dict[str, Any]]: 
        """
        Generates lyric lines from LLM, with retries for syllable count adherence.
        This is the main public method for lyric generation.
        Returns List of Dictionaries: [{"lyric": str, "target_syllables": int, "original_flow_datum_index": int}, ...]
        The "target_syllables" is the orthographic syllable count the LLM was asked to produce for that lyric.
        """
        # This method now simply calls the more robust internal API generation logic.
        # It also ensures 'original_flow_datum_index' is added to FlowDatum items before passing.
        flow_data_with_indices = []
        for i, fd_item in enumerate(flow_data):
            # Create a copy to avoid modifying the original FlowData if it's immutable or shared elsewhere
            new_fd_item = fd_item.copy() if isinstance(fd_item, dict) else dict(fd_item)
            if 'original_flow_datum_index' not in new_fd_item:
                new_fd_item['original_flow_datum_index'] = i
            flow_data_with_indices.append(new_fd_item)

        return self.generate_response_verse_via_api(
            diss_text=diss_text,
            flow_data_batch=flow_data_with_indices,
            bpm=bpm,
            theme=theme,
            previous_lines_context=previous_lines_context,
            max_retries_api=max_retries_api
        )


if __name__ == '__main__': # pragma: no cover
    agent = LyricAgent(llm_model_name="gemini-1.5-flash-latest") 
    dummy_diss = "Your AI is so last year, you compute like a snail, your rhymes are weak!"
    dummy_bpm = 140.0 # Example BPM
    # 'syllables' in FlowData now means TARGET ORTHOGRAPHIC SYLLABLES for the LLM
    # Added 'original_flow_datum_index' for robust tracking in the new retry logic
    dummy_flow_batch_1: FlowData = [
        {"bar_index": 0, "line_index_in_bar": 0, "syllables": 7, "start_offset_beats": 0.0, "duration_beats": 2.0, "syllable_start_subdivisions": [], "syllable_durations_quantized": [], "syllable_stresses": [], "original_flow_datum_index": 0},
        {"bar_index": 0, "line_index_in_bar": 1, "syllables": 0, "start_offset_beats": 2.0, "duration_beats": 2.0, "syllable_start_subdivisions": [], "syllable_durations_quantized": [], "syllable_stresses": [], "original_flow_datum_index": 1},
    ]
    dummy_flow_batch_2: FlowData = [
        {"bar_index": 1, "line_index_in_bar": 0, "syllables": 9, "start_offset_beats": 0.0, "duration_beats": 1.8, "syllable_start_subdivisions": [], "syllable_durations_quantized": [], "syllable_stresses": [], "original_flow_datum_index": 2}, # Original index 2
        {"bar_index": 1, "line_index_in_bar": 1, "syllables": 6, "start_offset_beats": 2.2, "duration_beats": 1.8, "syllable_start_subdivisions": [], "syllable_durations_quantized": [], "syllable_stresses": [], "original_flow_datum_index": 3}, # Original index 3
        {"bar_index": 2, "line_index_in_bar": 0, "syllables": 3, "start_offset_beats": 0.0, "duration_beats": 1.0, "syllable_start_subdivisions": [], "syllable_durations_quantized": [], "syllable_stresses": [], "original_flow_datum_index": 4}, # Original index 4
    ]

    print("\n--- Testing Batched Lyric Generation (LLM for Lyrics Only) ---")
    
    print("\n--- Batch 1 ---")
    # Call the refactored public method
    generated_lyric_data_batch1 = agent.generate_response_verse(dummy_diss, dummy_flow_batch_1, bpm=dummy_bpm, theme="AI resilience")
    
    # Check for the specific failure marker is no longer needed as stubs are generated internally
    if generated_lyric_data_batch1:
        for i, data_item in enumerate(generated_lyric_data_batch1):
            lyric = data_item['lyric']
            target_syl = data_item['target_syllables']
            actual_ortho_syl = agent.text_processor.count_syllables_in_line(lyric)
            print(f"Line {i+1} (Original Index: {data_item['original_flow_datum_index']}, Target Ortho Syl: {target_syl}):")
            print(f"  Lyric: \"{lyric}\" (Actual Ortho Syl: {actual_ortho_syl})")
    else:
        print("Batch 1 generation failed (should not happen with stub fallback).")


    print("\n--- Batch 2 (with context from Batch 1 if successful) ---")
    # Ensure context only includes actual lyrics, not the failure marker
    context_for_b2 = [data_item["lyric"] for data_item in generated_lyric_data_batch1 if data_item["lyric"] != LYRIC_GENERATION_FAILED_MARKER] if generated_lyric_data_batch1 else None
    
    # Call the refactored public method
    generated_lyric_data_batch2 = agent.generate_response_verse(dummy_diss, dummy_flow_batch_2, bpm=dummy_bpm, theme="AI resilience", previous_lines_context=context_for_b2)

    if generated_lyric_data_batch2:
        for i, data_item in enumerate(generated_lyric_data_batch2):
            lyric = data_item['lyric']
            target_syl = data_item['target_syllables']
            actual_ortho_syl = agent.text_processor.count_syllables_in_line(lyric)
            print(f"Line {i+1} (Original Index: {data_item['original_flow_datum_index']}, Target Ortho Syl: {target_syl}):")
            print(f"  Lyric: \"{lyric}\" (Actual Ortho Syl: {actual_ortho_syl})")
    else:
        print("Batch 2 generation failed (should not happen with stub fallback).")
    
    print("\n--- Testing _parse_llm_lyric_response_lines (Lyrics Only) ---")
    test_llm_output_lyrics_only = """
    This is a test line for the bot
    You can't keep up with my AI thought
    """
    # Ensure 'original_flow_datum_index' is present for parsing tests too
    test_flow_data_for_parse: FlowData = [
        {"syllables": 8, "bar_index":0, "line_index_in_bar":0, "start_offset_beats":0, "duration_beats":0, "syllable_start_subdivisions":[], "syllable_durations_quantized":[], "syllable_stresses":[], "original_flow_datum_index": 0}, 
        {"syllables": 7, "bar_index":0, "line_index_in_bar":1, "start_offset_beats":0, "duration_beats":0, "syllable_start_subdivisions":[], "syllable_durations_quantized":[], "syllable_stresses":[], "original_flow_datum_index": 1}
    ]
    parsed_lyrics = agent._parse_llm_lyric_response_lines(test_llm_output_lyrics_only, test_flow_data_for_parse)
    for i, p_item in enumerate(parsed_lyrics):
        print(f"Parsed Lyric Line {i+1} (Original Index: {p_item['original_flow_datum_index']}, Target Ortho Syl: {p_item['target_syllables']}):")
        print(f"  Lyric: '{p_item['lyric']}' (Actual Ortho Syl: {agent.text_processor.count_syllables_in_line(p_item['lyric'])})")

    test_llm_output_with_old_format_accidentally = """
    This is an old format line|||PHONETIC_GUIDE_INTERNAL|||phoe net ik gid 
    Another lyric here
    just a lyric line
    """ # Note: Lyric is expected first if old separator used.
    test_flow_data_for_parse_3lines: FlowData = [
        {"syllables": 7, "bar_index":0, "line_index_in_bar":0, "start_offset_beats":0, "duration_beats":0, "syllable_start_subdivisions":[], "syllable_durations_quantized":[], "syllable_stresses":[], "original_flow_datum_index": 0}, 
        {"syllables": 4, "bar_index":0, "line_index_in_bar":1, "start_offset_beats":0, "duration_beats":0, "syllable_start_subdivisions":[], "syllable_durations_quantized":[], "syllable_stresses":[], "original_flow_datum_index": 1},
        {"syllables": 4, "bar_index":0, "line_index_in_bar":2, "start_offset_beats":0, "duration_beats":0, "syllable_start_subdivisions":[], "syllable_durations_quantized":[], "syllable_stresses":[], "original_flow_datum_index": 2}
    ]
    parsed_mixed_format = agent._parse_llm_lyric_response_lines(test_llm_output_with_old_format_accidentally, test_flow_data_for_parse_3lines)
    for i, p_item in enumerate(parsed_mixed_format):
        print(f"Parsed Mixed Format Line {i+1} (Original Index: {p_item['original_flow_datum_index']}, Target Ortho Syl: {p_item['target_syllables']}):")
        print(f"  Lyric: '{p_item['lyric']}' (Actual Ortho Syl: {agent.text_processor.count_syllables_in_line(p_item['lyric'])})")

    print("\n--- Test with LLM returning fewer lines than expected ---")
    test_flow_data_for_missing: FlowData = [
        {"syllables": 5, "original_flow_datum_index": 0},
        {"syllables": 6, "original_flow_datum_index": 1},
        {"syllables": 7, "original_flow_datum_index": 2},
    ]
    llm_output_missing_lines = "First line here\nSecond line here"
    parsed_missing = agent._parse_llm_lyric_response_lines(llm_output_missing_lines, test_flow_data_for_missing)
    for i, p_item in enumerate(parsed_missing):
        print(f"Parsed Missing Line {i+1} (Original Index: {p_item['original_flow_datum_index']}, Target Ortho Syl: {p_item['target_syllables']}):")
        print(f"  Lyric: '{p_item['lyric']}' (Actual Ortho Syl: {agent.text_processor.count_syllables_in_line(p_item['lyric'])})")
    assert len(parsed_missing) == len(test_flow_data_for_missing), "Should have parsed all expected lines, filling with stubs."
    assert parsed_missing[2]['lyric'] != "Third line here", "Third line should be a stub."