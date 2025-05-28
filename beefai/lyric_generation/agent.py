import random
from typing import List, Optional, Dict, Any
from beefai.utils.data_types import FlowData, FlowDatum
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer # For actual LLM

class LyricAgent:
    def __init__(self, model_name_or_path: Optional[str] = None):
        """
        Placeholder for the Lyric Generation Agent.
        """
        self.model_name = model_name_or_path or "distilgpt2" # Default placeholder
        self.llm_pipeline = None
        # try:
        #     # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #     # self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        #     # self.llm_pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        #     print(f"LyricAgent initialized with placeholder for model: {self.model_name}")
        #     print("To use a real LLM, uncomment lines and install `transformers` and `torch` or `tensorflow`.")
        # except Exception as e:
        #     print(f"Could not load LLM model {self.model_name}. LyricAgent will use very basic placeholders. Error: {e}")
        #     self.llm_pipeline = None
        print(f"LyricAgent initialized. (LLM integration placeholder for: {self.model_name})")
        
        # Placeholder for syllable counting (ideally, use TextProcessor instance passed in or similar)
        from beefai.data_processing.text_processor import TextProcessor as LocalTextProcessor
        self.text_processor = LocalTextProcessor()


    def _construct_prompt(self, opponent_text: Optional[str], flow_datum: FlowDatum, prev_ai_lines: List[str]) -> str:
        """
        Constructs a prompt for the LLM based on battle context and flow constraints.
        """
        syllables_target = flow_datum.get("syllables", 8)
        duration_sec = flow_datum.get("duration_sec", 2.0)
        bar_idx = flow_datum.get("bar_index", "N/A")
        line_idx = flow_datum.get("line_index_in_bar", "N/A")

        prompt_parts = []
        prompt_parts.append(f"You are a witty and skilled rap battle AI. Your current line is line {line_idx} of bar {bar_idx}.")
        
        if opponent_text:
            prompt_parts.append(f"Your opponent just said: \"{opponent_text[-150:]}\".") # Limit context length
        
        if prev_ai_lines:
            context_lines = "\n".join([f"Your previous line: \"{line}\"" for line in prev_ai_lines[-2:]]) # Last 2 AI lines
            prompt_parts.append(context_lines)

        prompt_parts.append(f"Craft a rap line that has approximately {syllables_target} syllables and would fit naturally in about {duration_sec:.1f} seconds.")
        prompt_parts.append("Make it rhyme if appropriate, be clever, and stay on topic if responding.")
        if opponent_text:
             prompt_parts.append("Your response should be a comeback or a counter-argument.")
        else:
            prompt_parts.append("Start with a strong opening line.")
            
        # TODO: Add more sophisticated prompting based on pitch_contour_id, rhyme scheme, etc.
        # E.g., "The line should feel like it's rising in energy."
        # E.g., "Try to make the last word rhyme with 'track'." (if rhyme scheme is managed)
        
        return "\n".join(prompt_parts)

    def generate_lyrics_for_segment(self, opponent_text: Optional[str], flow_datum: FlowDatum, prev_ai_lines: List[str]) -> str:
        """
        Generates lyrics for a single flow segment.
        """
        syllables_target = flow_datum.get("syllables", 8)
        
        if self.llm_pipeline:
            # prompt = self._construct_prompt(opponent_text, flow_datum, prev_ai_lines)
            # # A real LLM would need careful generation parameter tuning (max_new_tokens, temperature, etc.)
            # # and post-processing to fit syllable/rhyme constraints. This is non-trivial.
            # # generated_sequences = self.llm_pipeline(prompt, max_new_tokens=syllables_target * 3, num_return_sequences=1) # Rough estimate
            # # generated_text = generated_sequences[0]['generated_text'].replace(prompt, "").strip() # Remove prompt
            
            # # Placeholder for complex refinement logic to meet constraints:
            # # 1. Generate multiple candidates.
            # # 2. Score candidates based on syllable count (using self.text_processor), rhyme (if applicable), relevance.
            # # 3. Select best candidate, or truncate/extend if necessary (can sound unnatural).
            # words = generated_text.split()
            # current_syllables = self.text_processor.count_syllables_in_line(" ".join(words))
            # # This is a very naive adjustment. Real systems use iterative generation or constrained decoding.
            # while current_syllables > syllables_target and len(words) > 1:
            #     words.pop()
            #     current_syllables = self.text_processor.count_syllables_in_line(" ".join(words))
            # return " ".join(words) if words else "LLM line placeholder"
            pass # Actual LLM logic would go here

        # Fallback: very simple rule-based generation for placeholder
        common_rap_words = ["yo", "check", "mic", "flow", "beat", "rhyme", "time", "drop", "hot", "dope", "word", "lyric", "street", "fire", "game", "true", "blue", "crew"]
        line_words = []
        current_syllables_count = 0
        
        # Attempt to make words fit the syllable target, using the TextProcessor
        for _ in range(syllables_target * 2): # Try more words to have options
            if not common_rap_words: break 
            word_choice = random.choice(common_rap_words)
            word_syllables = self.text_processor.count_syllables_in_word(word_choice)

            if current_syllables_count + word_syllables <= syllables_target + 1: # Allow slight overrun
                line_words.append(word_choice)
                current_syllables_count += word_syllables
                if current_syllables_count >= syllables_target -1: # Close enough
                    break
            if len(line_words) > syllables_target: # Avoid overly long lines if syllable counting is off
                break
        
        # If line is too short, pad it
        while current_syllables_count < syllables_target - 2 and len(line_words) < syllables_target:
            word_choice = random.choice(common_rap_words)
            word_syllables = self.text_processor.count_syllables_in_word(word_choice)
            if current_syllables_count + word_syllables <= syllables_target + 2:
                line_words.append(word_choice)
                current_syllables_count += word_syllables
            else:
                break # Cannot add more without significant overrun

        return " ".join(line_words) if line_words else f"placeholder line ({syllables_target} syl)"


    def generate_verse(self, opponent_text: Optional[str], flow_data: FlowData) -> List[str]:
        """
        Generates a full rap verse based on the opponent's text and the provided flow data.
        Each string in the output list corresponds to a segment in flow_data.
        """
        print("LyricAgent: Generating placeholder lyrics...")
        verse_lines: List[str] = []
        if not flow_data:
            print("Warning: No flow data provided to LyricAgent. Returning empty verse.")
            return ["(silence due to no flow data)"]

        # For maintaining coherence within AI's verse
        ai_generated_lines_so_far: List[str] = []

        for i, flow_datum in enumerate(flow_data):
            contextual_opponent_text = opponent_text
            # if opponent_text and len(opponent_text) > 150: # Limit opponent context
            #     contextual_opponent_text = opponent_text[-150:]
            
            line_lyrics = self.generate_lyrics_for_segment(contextual_opponent_text, flow_datum, ai_generated_lines_so_far)
            verse_lines.append(line_lyrics)
            ai_generated_lines_so_far.append(line_lyrics) # Add to AI's own context for next line
            
            actual_syllables = self.text_processor.count_syllables_in_line(line_lyrics)
            print(f"  Generated line for flow segment {i+1} (target {flow_datum.get('syllables')} syl, actual {actual_syllables} syl): \"{line_lyrics}\"")
        
        return verse_lines

# Example Usage
if __name__ == "__main__":
    agent = LyricAgent() 
    
    mock_flow_data: FlowData = [
        {"syllables": 8, "pitch_contour_id": "mid", "start_time_sec": 0.0, "duration_sec": 2.0, "bar_index":1, "line_index_in_bar":1},
        {"syllables": 10, "pitch_contour_id": "rising", "start_time_sec": 2.0, "duration_sec": 2.0, "bar_index":1, "line_index_in_bar":2},
        {"syllables": 7, "pitch_contour_id": "mid-high", "start_time_sec": 4.0, "duration_sec": 1.5, "bar_index":2, "line_index_in_bar":1},
        {"syllables": 9, "pitch_contour_id": "falling", "start_time_sec": 5.5, "duration_sec": 2.0, "bar_index":2, "line_index_in_bar":2}
    ]
    
    opponent_rap = "You think you're hot stuff, but your rhymes are pretty bad, AI can't really rap, it's just a fad."
    
    ai_verse = agent.generate_verse(opponent_rap, mock_flow_data)
    
    print("\nAI Generated Verse:")
    for i, line in enumerate(ai_verse):
        target_syl = mock_flow_data[i]['syllables']
        actual_syl = agent.text_processor.count_syllables_in_line(line)
        print(f"  Line {i+1} (Flow Syllables: {target_syl}, Actual: {actual_syl}): {line}")

    print("\nGenerating verse with no opponent text (e.g. starting a rap):")
    ai_opening_verse = agent.generate_verse(None, mock_flow_data)
    print("\nAI Generated Opening Verse:")
    for i, line in enumerate(ai_opening_verse):
        target_syl = mock_flow_data[i]['syllables']
        actual_syl = agent.text_processor.count_syllables_in_line(line)
        print(f"  Line {i+1} (Flow Syllables: {target_syl}, Actual: {actual_syl}): {line}")