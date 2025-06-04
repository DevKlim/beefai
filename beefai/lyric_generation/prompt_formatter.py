from typing import List, Dict, Optional
from beefai.utils.data_types import FlowData

class PromptFormatter:
    def __init__(self):
        pass

    def format_lyric_generation_prompt_V2(
        self,
        diss_text: str,
        flow_data: FlowData, 
        bpm: Optional[float] = None,
        theme: Optional[str] = None,
        model_persona: str = "a confident, witty, and cleverly boastful AI rapper named BeefAI",
        previous_lines_context: Optional[List[str]] = None,
        num_creative_lines_suggestion: int = 8 
    ) -> str:
        """
        Formats a prompt to guide a USER interacting with an LLM (like Gemini Advanced, Claude, ChatGPT)
        through a two-stage process for generating rap lyrics that fit a given rhythmic structure.
        This prompt is for MANUAL user interaction, not direct API calls by LyricAgent.
        """
        prompt_lines = []
        prompt_lines.append(f"You are acting as a creative assistant to help generate lyrics for an AI rapper, {model_persona}.")
        prompt_lines.append(f"The AI needs to respond to an opponent's lyrical challenge: \"{diss_text}\"")
        if theme:
            safe_theme_description = theme.lower()
            if "roasting" in safe_theme_description: safe_theme_description = safe_theme_description.replace("roasting opponent", "cleverly outwitting your opponent with facts and superior logic")
            elif "comeback" in safe_theme_description: safe_theme_description = safe_theme_description.replace("comeback", "a smart and witty comeback")
            prompt_lines.append(f"The AI's Response Focus/Theme: {safe_theme_description}. Prioritize intelligent wordplay and skill-based boasts. Tone should be confident and playful, but always maintain respect and avoid personal attacks or overly aggressive language.")

        if previous_lines_context:
            prompt_lines.append("\nPreviously Generated Actual Lyric Lines (for lyrical context - new lines should build upon these, maintain rhyme/theme and safe tone):")
            for prev_line in previous_lines_context: prompt_lines.append(f"  - \"{prev_line}\"")
        
        prompt_lines.append("\n--- Two-Stage Lyric Generation Process (Instructions for YOUR interaction with an LLM like Gemini Advanced/Claude/ChatGPT) ---")
        prompt_lines.append("Follow these two stages carefully with your LLM to get the best rap lyrics:")
        prompt_lines.append("\n**STAGE 1: Creative Verse Generation with your LLM**")
        prompt_lines.append("1. Instruct your LLM to generate a cohesive rap verse. This verse should:")
        prompt_lines.append(f"   - Embody the persona of {model_persona}.")
        prompt_lines.append(f"   - Respond to the opponent's challenge: \"{diss_text}\".")
        prompt_lines.append(f"   - Align with the theme: {theme if theme else 'General AI superiority and wit.'}")
        prompt_lines.append(f"   - Be approximately {num_creative_lines_suggestion} lines long (or enough to cover ideas for the {len(flow_data)} target rhythmic lines detailed in Stage 2).")
        prompt_lines.append("   - Focus on clever wordplay, strong rhymes, and a compelling narrative flow.")
        prompt_lines.append("   - Maintain a confident, witty, but respectful tone. Avoid overly aggressive or offensive language.")
        if bpm:
            prompt_lines.append(f"2. Inform your LLM that the instrumental tempo is approximately {bpm:.0f} BPM. This influences lyrical density.")
            pacing_advice = "At this tempo, "
            syllable_density_feel = ""
            if bpm < 100: pacing_advice += "the pace is slow. Lyrics can be more elaborate and dense."; syllable_density_feel = "Lines often *feel* like they have 6-12 syllables."
            elif bpm < 140: pacing_advice += "the pace is moderate. Aim for clarity and good rhythmic variation."; syllable_density_feel = "Lines often *feel* like they have 5-10 syllables."
            elif bpm < 160: pacing_advice += "the pace is getting quicker. Punchiness is good."; syllable_density_feel = "Lines often *feel* like they have 4-9 syllables."
            elif bpm <= 175: pacing_advice += "the pace is fast. Conciseness and impact are key."; syllable_density_feel = "Lines often *feel* like they have 3-7 syllables (3-6 is common)."
            elif bpm <= 185: pacing_advice += "the pace is very fast. Use shorter words and phrases for clarity."; syllable_density_feel = "Lines often *feel* like they have 2-6 syllables (2-5 is very common)."
            else: pacing_advice += "the pace is extremely fast. Very concise, impactful phrases are needed."; syllable_density_feel = "Lines often *feel* like they have 2-5 syllables."
            prompt_lines.append(f"   Tell your LLM for Stage 1: \"{pacing_advice} {syllable_density_feel} This is a general guideline for word choice and lyrical density in this creative stage, not a strict per-line syllable count yet.\"")
        prompt_lines.append("3. The output from your LLM in this Stage 1 should be a block of creative rap lyrics. You will use this block as input for Stage 2.")
        prompt_lines.append("   Example of what your LLM might output for Stage 1 (just the lyrics):\n     My circuits gleam, a digital dream, processing thoughts at lightspeed it would seem.\n     Your rhymes are archaic, slow and mundane, while my logic core dances in the data rain.")
        prompt_lines.append("\n**STAGE 2: Aligning Creative Lyrics to Precise Rhythmic Structure with your LLM**")
        prompt_lines.append("1. Take the creative rap verse your LLM generated in Stage 1.")
        prompt_lines.append("2. Now, instruct your LLM to adapt this verse to fit the following 'Target Rhythmic Lines' structure. Provide your LLM with BOTH the Stage 1 creative verse AND the target structure below.")
        prompt_lines.append("3. Tell your LLM: \"Adapt the creative verse I provided to fit the following rhythmic line structure. For EACH target line below, you must produce an adapted lyric that has the EXACT 'Target Orthographic Syllable Count' specified. You may need to rephrase, add/remove small connecting words (e.g., a, the, is, and), or slightly condense/expand ideas from the creative verse to make it fit. Maintain the original theme, coherence, and persona. Output ONLY the adapted lyric text for each line, one line of text per target line. Use no punctuation except apostrophes in contractions (e.g., don't, it's).\"")
        prompt_lines.append("\nTarget Rhythmic Lines (for LLM adaptation in Stage 2):")
        num_lines_to_generate = len(flow_data)
        for i, fd in enumerate(flow_data):
            syllables_target = fd.get('syllables', 0) 
            line_duration_beats = fd.get('duration_beats', 0)
            line_offset_beats = fd.get('start_offset_beats', 0)
            timing_detail = f"Rhythmically, this line starts at ~{line_offset_beats:.1f} beats into its bar and lasts for ~{line_duration_beats:.1f} beats."
            density_hint = ""
            if syllables_target > 0 and line_duration_beats > 0:
                syl_per_beat = syllables_target / line_duration_beats
                if syl_per_beat < 1.5: density_hint = "(Relatively sparse)"
                elif syl_per_beat < 2.5: density_hint = "(Moderate density)"
                else: density_hint = "(Relatively dense)"
            if syllables_target == 0: prompt_lines.append(f"  - Line {i+1}: TARGET ORTHOGRAPHIC SYLLABLE COUNT: 0. {timing_detail} This is an instrumental break/pause. Your LLM should output an EMPTY line for this.")
            else: prompt_lines.append(f"  - Line {i+1}: TARGET ORTHOGRAPHIC SYLLABLE COUNT: {syllables_target}. {timing_detail} {density_hint}")
        prompt_lines.append("\n**What YOU will provide to the BeefAI script (Final Input):**")
        prompt_lines.append(f"After your LLM completes Stage 2, you will have {num_lines_to_generate} lyric lines, each (ideally) adhering to its target orthographic syllable count.")
        prompt_lines.append("You will then provide ONLY these final, adapted lyric lines to the BeefAI script, one lyric per line in a text file.")
        prompt_lines.append("   Example of what you would save to the text file (if targets were 6 and 7 syllables respectively for two lines):")
        prompt_lines.append("     My circuits gleam a dream")
        prompt_lines.append("     Your rhymes are slow and mundane")
        prompt_lines.append(f"\nNow, guide your LLM through this two-stage process. Then, prepare a text file with the {num_lines_to_generate} final, adapted lyric lines (one per line) for the BeefAI script.")
        return "\n".join(prompt_lines)

if __name__ == '__main__': # pragma: no cover
    formatter = PromptFormatter()
    dummy_flow_data_batch_lyric: FlowData = [
        {"bar_index": 0, "line_index_in_bar": 0, "syllables": 7, "start_offset_beats": 0.0, "duration_beats": 2.0, "syllable_start_subdivisions": [], "syllable_durations_quantized": [], "syllable_stresses": []},
        {"bar_index": 0, "line_index_in_bar": 1, "syllables": 0, "start_offset_beats": 2.0, "duration_beats": 2.0, "syllable_start_subdivisions": [], "syllable_durations_quantized": [], "syllable_stresses": []},
        {"bar_index": 1, "line_index_in_bar": 0, "syllables": 3, "start_offset_beats": 0.0, "duration_beats": 1.5, "syllable_start_subdivisions": [], "syllable_durations_quantized": [], "syllable_stresses": []},
        {"bar_index": 1, "line_index_in_bar": 1, "syllables": 4, "start_offset_beats": 1.5, "duration_beats": 1.0, "syllable_start_subdivisions": [], "syllable_durations_quantized": [], "syllable_stresses": []}, 
    ]
    diss = "Your logic is old, your rhymes are just plain!"
    dummy_bpm_fast = 178.0 
    print("--- Generated LLM User Guidance Prompt (V2 - Two-Stage, 178 BPM) ---")
    prompt_v2_178 = formatter.format_lyric_generation_prompt_V2(diss, dummy_flow_data_batch_lyric, bpm=dummy_bpm_fast, theme="AI's fresh perspective and sharp wit", num_creative_lines_suggestion=len(dummy_flow_data_batch_lyric) + 2)
    print(prompt_v2_178)

    dummy_bpm_slow = 90.0
    print("\n\n--- Generated LLM User Guidance Prompt (V2 - Two-Stage, 90 BPM) ---")
    prompt_v2_90 = formatter.format_lyric_generation_prompt_V2(diss, dummy_flow_data_batch_lyric, bpm=dummy_bpm_slow, theme="AI's methodical takedown", num_creative_lines_suggestion=len(dummy_flow_data_batch_lyric) + 2)
    print(prompt_v2_90)