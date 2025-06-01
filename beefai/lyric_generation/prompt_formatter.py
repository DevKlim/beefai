from typing import List, Dict, Optional
from beefai.utils.data_types import FlowData, FlowDatum, BeatInfo
from beefai.flow_model.tokenizer import FlowTokenizer # To dequantize duration bins

class LLMPromptFormatter:
    def __init__(self, tokenizer: FlowTokenizer):
        """
        Initializes the formatter with a FlowTokenizer instance
        to access dequantization logic.
        """
        self.tokenizer = tokenizer

    def format_flow_for_llm(self, 
                            flow_data: FlowData, 
                            beat_info: BeatInfo,
                            additional_prompt_context: Optional[str] = None
                           ) -> str:
        """
        Formats FlowData and BeatInfo into a textual prompt suitable for an LLM
        to generate lyrics.

        Args:
            flow_data: The generated flow data from the model.
            beat_info: Beat information for the instrumental (BPM, time signature).
            additional_prompt_context: Optional text providing thematic context for the LLM.

        Returns:
            A string prompt for the LLM.
        """
        if not flow_data:
            return "Error: No flow data provided."

        bpm = beat_info.get("bpm", 120.0)
        time_sig_num = beat_info.get("beats_per_bar", 4)
        time_sig_den = 4 # Assuming common time denominator for now
        
        prompt_lines = []

        if additional_prompt_context:
            prompt_lines.append(f"Context: {additional_prompt_context}\n")

        prompt_lines.append(f"Music Beat Context: {bpm:.1f} BPM, {time_sig_num}/{time_sig_den} time signature.")
        prompt_lines.append("Generate rap lyrics that match the following rhythmic flow structure, line by line:")
        prompt_lines.append("-" * 20)

        for i, fd in enumerate(flow_data):
            line_header = f"Line {i+1} (Bar {fd['bar_index']}, Segment {fd['line_index_in_bar']}):"
            prompt_lines.append(f"\n{line_header}")
            prompt_lines.append(f"  - Total Syllables: {fd['syllables']}")
            prompt_lines.append(f"  - Timing: Starts approximately {fd['start_offset_beats']:.2f} beats into its bar, "
                                f"and the lyrical phrase lasts for about {fd['duration_beats']:.2f} beats.")
            
            if fd['syllables'] > 0 and \
               len(fd.get('syllable_start_subdivisions', [])) == fd['syllables'] and \
               len(fd.get('syllable_durations_quantized', [])) == fd['syllables'] and \
               len(fd.get('syllable_stresses', [])) == fd['syllables']:
                
                prompt_lines.append("  - Syllable Pattern:")
                for syl_idx in range(fd['syllables']):
                    start_subdiv = fd['syllable_start_subdivisions'][syl_idx]
                    dur_bin = fd['syllable_durations_quantized'][syl_idx]
                    stress_val = fd['syllable_stresses'][syl_idx]

                    # Dequantize duration for a more human-readable description
                    approx_dur_beats = self.tokenizer.dequantize_syllable_duration_bin(dur_bin)
                    # Convert subdivision to something more intuitive if possible (e.g. "on the 1", "on the 'and' of 2")
                    # For 16 subdivisions: 0=1, 4=2, 8=3, 12=4 (main beats)
                    # 2=1e&a, 6=2e&a etc.
                    subdiv_desc = f"16th note subdivision {start_subdiv}" # Basic description

                    stress_desc = "unstressed"
                    if stress_val == 1: stress_desc = "primary stress"
                    elif stress_val == 2: stress_desc = "secondary stress"
                    
                    prompt_lines.append(f"    - Syllable {syl_idx+1}: onset at {subdiv_desc}, "
                                        f"duration approx. {approx_dur_beats:.2f} beats, {stress_desc}.")
            else:
                prompt_lines.append("  - Syllable Pattern: (Detailed per-syllable timing/stress information is incomplete or unavailable for this line).")
            
            prompt_lines.append(f"  Your rap lyrics for this line ({fd['syllables']} syllables):")

        prompt_lines.append("-" * 20)
        prompt_lines.append("Ensure lyrics are creative, make sense with the context (if any), and follow the syllable counts and rhythmic feel described for each line.")

        return "\n".join(prompt_lines)

if __name__ == "__main__":
    # Dummy data for testing the formatter
    # Need a tokenizer instance for dequantizing durations
    # This assumes flow_tokenizer_config_v2.json exists where FlowTokenizer can find it
    try:
        tokenizer_for_test = FlowTokenizer() 
    except FileNotFoundError:
        print("ERROR: flow_tokenizer_config_v2.json not found. Cannot run LLMPromptFormatter test.")
        print("Ensure the tokenizer config exists, e.g., by running FlowTokenizer's __main__ block once.")
        exit()
        
    formatter = LLMPromptFormatter(tokenizer=tokenizer_for_test)

    dummy_flow: FlowData = [
        {
            "bar_index": 0, "line_index_in_bar": 0, "syllables": 3,
            "start_offset_beats": 0.0, "duration_beats": 1.0,
            "syllable_start_subdivisions": [0, 2, 4],    # e.g., 1, 1e, 1a
            "syllable_durations_quantized": [1, 0, 1], # Bin indices (e.g., short, very short, short)
            "syllable_stresses": [1, 0, 0]             # Primary, Unstressed, Unstressed
        },
        {
            "bar_index": 0, "line_index_in_bar": 1, "syllables": 5,
            "start_offset_beats": 2.0, "duration_beats": 1.75,
            "syllable_start_subdivisions": [8, 9, 10, 12, 14],
            "syllable_durations_quantized": [0, 0, 1, 2, 1],
            "syllable_stresses": [0, 0, 1, 0, 2]
        },
         {
            "bar_index": 1, "line_index_in_bar": 0, "syllables": 0, # Test line with 0 syllables
            "start_offset_beats": 0.0, "duration_beats": 0.0,
            "syllable_start_subdivisions": [], 
            "syllable_durations_quantized": [], 
            "syllable_stresses": []
        }
    ]
    dummy_beat_info: BeatInfo = {
        "bpm": 100.0,
        "beat_times": [], # Not directly used by formatter, but part of BeatInfo
        "downbeat_times": [],
        "estimated_bar_duration": 2.4, # (60/100bpm * 4 beats)
        "beats_per_bar": 4
    }
    
    additional_context = "The theme is AI taking over the rap game."

    llm_prompt = formatter.format_flow_for_llm(dummy_flow, dummy_beat_info, additional_context)
    print("\n--- Generated LLM Prompt ---")
    print(llm_prompt)

    print("\n--- Test with minimal flow data ---")
    minimal_flow: FlowData = [
        {
            "bar_index": 0, "line_index_in_bar": 0, "syllables": 1,
            "start_offset_beats": 0.0, "duration_beats": 0.25,
            "syllable_start_subdivisions": [0],   
            "syllable_durations_quantized": [0], 
            "syllable_stresses": [1]            
        }]
    llm_prompt_minimal = formatter.format_flow_for_llm(minimal_flow, dummy_beat_info)
    print(llm_prompt_minimal)