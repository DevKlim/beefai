import numpy as np
from typing import List, Optional, Dict, Tuple, Set, Any
import os
import soundfile as sf
import io
import pydub # type: ignore
import random
import concurrent.futures
import time
import librosa
import re
import subprocess # For espeak-ng
import tempfile # For temporary WAV files from espeak-ng

from beefai.utils.data_types import FlowData, AudioData, BeatInfo, FlowDatum
from beefai.data_processing.text_processor import TextProcessor 
from beefai.flow_model.tokenizer import FlowTokenizer

LOG_PREFIX_SYNTH = "[RapSynthesizer]"

def ensure_dir(directory_path: str):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

class RapSynthesizer:
    def __init__(self, 
                 sample_rate: int = 44100, 
                 flow_tokenizer_config_path: Optional[str] = None, 
                 lang: str = 'en', 
                 espeak_voice: str = 'en-us', 
                 max_workers_tts: int = 4,
                 primary_stress_pitch_shift_semitones: float = 0.5,
                 secondary_stress_pitch_shift_semitones: float = 0.25,
                 primary_stress_gain_db: float = 1.5,
                 secondary_stress_gain_db: float = 0.75
                 ):
        self.sample_rate = sample_rate
        self.text_processor = TextProcessor(language='en_US' if 'en' in espeak_voice else espeak_voice.split('-')[0])
        self.lang = lang 
        self.espeak_voice = espeak_voice 
        self.max_workers_tts = max(1, max_workers_tts) 
        
        self.primary_stress_pitch_shift = primary_stress_pitch_shift_semitones
        self.secondary_stress_pitch_shift = secondary_stress_pitch_shift_semitones
        self.primary_stress_gain_factor = 10**(primary_stress_gain_db / 20.0)
        self.secondary_stress_gain_factor = 10**(secondary_stress_gain_db / 20.0)

        default_tokenizer_path = "beefai/flow_model/flow_tokenizer_config_v2.json"
        effective_tokenizer_path = flow_tokenizer_config_path if flow_tokenizer_config_path and os.path.exists(flow_tokenizer_config_path) else default_tokenizer_path
        
        if os.path.exists(effective_tokenizer_path):
            self.flow_tokenizer = FlowTokenizer(config_path=effective_tokenizer_path)
        else: # pragma: no cover
            print(f"{LOG_PREFIX_SYNTH} Warning: FlowTokenizer config not found at '{effective_tokenizer_path}' or default. Using basic FlowTokenizer.")
            self.flow_tokenizer = FlowTokenizer() 
        
        print(f"{LOG_PREFIX_SYNTH} Initialized. Base vocal source: espeak-ng (voice: {self.espeak_voice}). Max TTS workers: {self.max_workers_tts}.")
        print(f"{LOG_PREFIX_SYNTH} Stress modulation: Pitch (Semi) P1={self.primary_stress_pitch_shift},P2={self.secondary_stress_pitch_shift}. Gain (Factor) P1={self.primary_stress_gain_factor:.2f},P2={self.secondary_stress_gain_factor:.2f}")
        print(f"{LOG_PREFIX_SYNTH} Synthesis uses whole lyric lines for TTS, then segments audio. RVC/Vocaloid smoothing is conceptual.")
        self._check_espeak_ng()

    def _check_espeak_ng(self):
        try:
            result = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True, check=True, timeout=5)
            print(f"{LOG_PREFIX_SYNTH} espeak-ng found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e: # pragma: no cover
            print(f"{LOG_PREFIX_SYNTH} CRITICAL WARNING: espeak-ng not found or not working. Synthesis will fail.")
            print(f"{LOG_PREFIX_SYNTH} Error: {e}")
            print(f"{LOG_PREFIX_SYNTH} Please ensure espeak-ng is installed and in your system's PATH.")
            return False

    def _generate_line_audio_espeak(self, lyric_line_text: str) -> Tuple[str, Optional[pydub.AudioSegment]]:
        """Generates audio for an ENTIRE lyric line text using espeak-ng."""
        lyric_line_text_cleaned = lyric_line_text.strip()
        if not lyric_line_text_cleaned:
            return lyric_line_text, None # Return None for empty lines
        
        text_for_espeak = lyric_line_text_cleaned
        tmp_wav_path = "" 
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
                tmp_wav_path = tmp_wav_file.name
            
            cmd = [ "espeak-ng", "-v", self.espeak_voice, "-s", "170", "-w", tmp_wav_path, text_for_espeak ] # -s 170 is a fairly standard speaking rate
            
            process = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore', timeout=15) # Longer timeout for whole lines
            
            if process.returncode != 0: # pragma: no cover
                print(f"{LOG_PREFIX_SYNTH}   espeak-ng error for line '{text_for_espeak}': Code {process.returncode}, STDERR: {process.stderr if process.stderr else 'N/A'}")
                return lyric_line_text, None

            if not os.path.exists(tmp_wav_path) or os.path.getsize(tmp_wav_path) < 44: 
                print(f"{LOG_PREFIX_SYNTH}   espeak-ng produced an empty/missing file for line '{text_for_espeak}'.")
                return lyric_line_text, None

            segment = pydub.AudioSegment.from_wav(tmp_wav_path)
            processed_segment = segment.set_frame_rate(self.sample_rate).set_channels(1)
            return lyric_line_text, processed_segment

        except FileNotFoundError: # pragma: no cover 
            print(f"{LOG_PREFIX_SYNTH} CRITICAL: espeak-ng command not found. Cannot synthesize line '{text_for_espeak}'.")
            return lyric_line_text, None
        except subprocess.TimeoutExpired: # pragma: no cover
            print(f"{LOG_PREFIX_SYNTH}   espeak-ng call timed out for line '{text_for_espeak}'.")
            return lyric_line_text, None
        except Exception as e: # pragma: no cover
            print(f"{LOG_PREFIX_SYNTH}   Exception during espeak-ng call for line '{text_for_espeak}': {type(e).__name__} - {e}")
            return lyric_line_text, None
        finally:
            if tmp_wav_path and os.path.exists(tmp_wav_path):
                try: os.remove(tmp_wav_path)
                except Exception: pass # pragma: no cover

    def _segment_to_numpy(self, segment: pydub.AudioSegment) -> Tuple[np.ndarray, int]:
        samples_np = np.array(segment.get_array_of_samples()).astype(np.float32)
        if segment.channels == 2: # pragma: no cover
            samples_np = samples_np.reshape((-1, 2)).mean(axis=1)
        
        if segment.sample_width == 1: 
            samples_np = (samples_np - 128.0) / 128.0
        elif segment.sample_width == 2: 
            samples_np = samples_np / 32768.0 
        elif segment.sample_width == 4: # pragma: no cover
             samples_np = samples_np / 2147483648.0 
        return samples_np, segment.frame_rate

    def _numpy_to_segment(self, waveform_np: np.ndarray, frame_rate: int) -> pydub.AudioSegment:
        if waveform_np.ndim > 1: waveform_np = waveform_np.mean(axis=1) # pragma: no cover
        
        max_abs = np.max(np.abs(waveform_np)) if waveform_np.size > 0 else 0
        if max_abs > 1.0: waveform_np = waveform_np / max_abs 
        
        if waveform_np.size == 0 or np.all(np.abs(waveform_np) < 1e-9): 
             duration_ms = int(len(waveform_np) / frame_rate * 1000) if waveform_np.size > 0 and frame_rate > 0 else 0
             return pydub.AudioSegment.silent(duration=max(0,duration_ms), frame_rate=frame_rate if frame_rate > 0 else self.sample_rate)

        samples_int16 = (waveform_np * 32767).astype(np.int16)
        
        byte_data = samples_int16.tobytes()
        if not byte_data and waveform_np.size > 0 : # pragma: no cover
            duration_ms = int(len(waveform_np) / frame_rate * 1000) if frame_rate > 0 else 0
            return pydub.AudioSegment.silent(duration=max(0,duration_ms), frame_rate=frame_rate if frame_rate > 0 else self.sample_rate)

        return pydub.AudioSegment(data=byte_data, frame_rate=frame_rate, sample_width=2, channels=1)

    def _adjust_segment_duration_with_speedup(self, 
                                              segment: pydub.AudioSegment, 
                                              target_duration_ms: float,
                                              min_segment_duration_ms: float = 20.0, 
                                              max_speed_factor: float = 3.5, 
                                              min_speed_factor: float = 0.25  
                                              ) -> pydub.AudioSegment:
        if not segment or target_duration_ms <= 10 : 
            return pydub.AudioSegment.silent(duration=max(0, int(target_duration_ms)), frame_rate=self.sample_rate)
        
        original_duration_ms = len(segment)
        if original_duration_ms == 0: return pydub.AudioSegment.silent(duration=int(target_duration_ms), frame_rate=self.sample_rate) # pragma: no cover

        effective_target_duration_ms = max(min_segment_duration_ms, target_duration_ms)

        if abs(original_duration_ms - effective_target_duration_ms) < 10: 
            if original_duration_ms > effective_target_duration_ms: return segment[:int(effective_target_duration_ms)]
            else:
                silence_needed = int(effective_target_duration_ms - original_duration_ms)
                return segment + pydub.AudioSegment.silent(duration=silence_needed, frame_rate=segment.frame_rate) if silence_needed > 0 else segment
        try:
            samples_np, sr = self._segment_to_numpy(segment)
            if samples_np.size == 0: return pydub.AudioSegment.silent(duration=int(effective_target_duration_ms), frame_rate=sr if sr > 0 else self.sample_rate) # pragma: no cover
            
            librosa_stretch_rate = original_duration_ms / effective_target_duration_ms
            clamped_stretch_rate = np.clip(librosa_stretch_rate, min_speed_factor, max_speed_factor)

            stretched_samples_np = librosa.effects.time_stretch(y=samples_np, rate=clamped_stretch_rate) if abs(clamped_stretch_rate - 1.0) > 0.01 else samples_np
            stretched_segment = self._numpy_to_segment(stretched_samples_np, sr)
            
            current_len_ms_after_stretch = len(stretched_segment)
            if current_len_ms_after_stretch > effective_target_duration_ms: stretched_segment = stretched_segment[:int(effective_target_duration_ms)]
            elif current_len_ms_after_stretch < effective_target_duration_ms:
                silence_needed_ms = int(effective_target_duration_ms - current_len_ms_after_stretch)
                if silence_needed_ms > 0: stretched_segment += pydub.AudioSegment.silent(duration=silence_needed_ms, frame_rate=sr)
            return stretched_segment
        except Exception as e_stretch: # pragma: no cover
            print(f"{LOG_PREFIX_SYNTH}   Error during librosa time stretch for segment ({original_duration_ms}ms to {effective_target_duration_ms}ms): {type(e_stretch).__name__} - {e_stretch}. Using pydub fallback.")
            if original_duration_ms > effective_target_duration_ms: 
                try: 
                    speed_factor_pydub = original_duration_ms / effective_target_duration_ms 
                    return segment.speedup(playback_speed=np.clip(speed_factor_pydub, min_speed_factor, max_speed_factor))[:int(effective_target_duration_ms)]
                except: return segment[:int(effective_target_duration_ms)]
            else: 
                silence_needed = int(effective_target_duration_ms - original_duration_ms)
                return segment + pydub.AudioSegment.silent(duration=silence_needed, frame_rate=segment.frame_rate) if silence_needed > 0 else segment

    def _calculate_bar_timings(self, flow_data_for_lyrics: FlowData, beat_info: BeatInfo, beat_duration_sec: float) -> Optional[Dict[int, float]]:
        bar_absolute_start_times_sec: Dict[int, float] = {}
        max_bar_idx_flow = 0
        if flow_data_for_lyrics: 
            valid_indices = [fd.get("bar_index") for fd in flow_data_for_lyrics if isinstance(fd, dict) and fd.get("bar_index") is not None]
            if valid_indices: max_bar_idx_flow = max(valid_indices)
        
        estimated_num_bars_needed = max_bar_idx_flow + 10 

        if beat_info.get("downbeat_times") and len(beat_info["downbeat_times"]) > 0:
            current_downbeats = list(beat_info["downbeat_times"])
            if len(current_downbeats) < estimated_num_bars_needed: # pragma: no cover
                last_dt = current_downbeats[-1] if current_downbeats else 0.0
                est_bar_dur_for_projection = beat_info.get("estimated_bar_duration")
                if not est_bar_dur_for_projection or est_bar_dur_for_projection <= 0.1:
                    beats_per_bar_val = beat_info.get("beats_per_bar", 4); beats_per_bar_val = max(1, beats_per_bar_val)
                    est_bar_dur_for_projection = beats_per_bar_val * beat_duration_sec
                if est_bar_dur_for_projection <= 0.1: est_bar_dur_for_projection = 2.0
                for _ in range(estimated_num_bars_needed - len(current_downbeats)):
                    last_dt += est_bar_dur_for_projection; current_downbeats.append(last_dt)
            for i, dt in enumerate(current_downbeats):
                if i <= max_bar_idx_flow + 5: bar_absolute_start_times_sec[i] = dt
        elif beat_info.get("estimated_bar_duration", 0) > 0.1: # pragma: no cover
            bar_dur = beat_info["estimated_bar_duration"]
            for i in range(estimated_num_bars_needed): bar_absolute_start_times_sec[i] = i * bar_dur
        else: # pragma: no cover
            beats_per_bar_val = beat_info.get("beats_per_bar", 4); beats_per_bar_val = max(1, beats_per_bar_val)
            bar_dur_calc = beats_per_bar_val * beat_duration_sec
            if bar_dur_calc <=0.1: print(f"{LOG_PREFIX_SYNTH} Error: Cannot determine bar duration. BPM ({beat_info.get('bpm')}) or beats_per_bar ({beats_per_bar_val}) invalid."); return None
            for i in range(estimated_num_bars_needed): bar_absolute_start_times_sec[i] = i * bar_dur_calc
        
        if not bar_absolute_start_times_sec: print(f"{LOG_PREFIX_SYNTH} Error: BeatInfo lacks timing details. Cannot determine absolute bar timings."); return None # pragma: no cover
        return bar_absolute_start_times_sec

    def _get_processed_instrumental_np(self, instrumental_audio: Optional[AudioData], output_duration_sec: float) -> np.ndarray:
        required_samples_total = max(1, int(output_duration_sec * self.sample_rate))
        if instrumental_audio and instrumental_audio[0].size > 0 :
            instr_wf_orig, instr_sr_orig = instrumental_audio
            instr_wf_mono = librosa.to_mono(instr_wf_orig) if instr_wf_orig.ndim > 1 else instr_wf_orig
            if instr_sr_orig != self.sample_rate: # pragma: no cover
                instr_wf_mono = librosa.resample(instr_wf_mono, orig_sr=instr_sr_orig, target_sr=self.sample_rate)
            
            if not np.issubdtype(instr_wf_mono.dtype, np.floating): # pragma: no cover
                instr_wf_mono = instr_wf_mono.astype(np.float32) / (np.iinfo(instr_wf_mono.dtype).max if np.issubdtype(instr_wf_mono.dtype, np.integer) else (np.max(np.abs(instr_wf_mono)) if np.max(np.abs(instr_wf_mono)) > 1e-9 else 1.0))
            
            instr_wf_processed_float = instr_wf_mono
            if len(instr_wf_processed_float) < required_samples_total:
                return np.pad(instr_wf_processed_float, (0, required_samples_total - len(instr_wf_processed_float)), 'constant')
            else:
                return instr_wf_processed_float[:required_samples_total]
        else:
            return np.zeros(required_samples_total, dtype=np.float32)

    def synthesize_vocal_track( # Renamed from synthesize_phonetic_track
        self,
        lyric_lines_for_tts: List[str], 
        flow_data_for_lyrics: FlowData,      
        beat_info: BeatInfo,
        syllable_segment_fade_ms: int = 5, # Renamed from phonetic_part_fade_ms
        vocal_level_db: float = 0.0 
    ) -> Optional[AudioData]:
        print(f"{LOG_PREFIX_SYNTH} Synthesizing vocal track from lyric lines using espeak-ng and FlowData timing...")
        s_time = time.time()

        if not lyric_lines_for_tts or not flow_data_for_lyrics or len(lyric_lines_for_tts) != len(flow_data_for_lyrics):
            print(f"{LOG_PREFIX_SYNTH} Error: Mismatch/missing lyric_lines_for_tts ({len(lyric_lines_for_tts)}) or flow_data ({len(flow_data_for_lyrics)}).")
            return None
        
        bpm = beat_info.get("bpm")
        if not bpm or bpm <= 0: print(f"{LOG_PREFIX_SYNTH} Error: Invalid BPM ({bpm})."); return None # pragma: no cover
        beat_duration_sec = 60.0 / bpm
        
        bar_absolute_start_times_sec = self._calculate_bar_timings(flow_data_for_lyrics, beat_info, beat_duration_sec)
        if bar_absolute_start_times_sec is None: return None # pragma: no cover

        max_event_time_sec = 0.0
        # Calculate total duration based on FlowData
        for line_flow_datum in flow_data_for_lyrics:
            bar_idx = line_flow_datum.get("bar_index", 0)
            if bar_idx not in bar_absolute_start_times_sec: # Should be populated by _calculate_bar_timings
                print(f"{LOG_PREFIX_SYNTH}   Warning: Bar index {bar_idx} from FlowData not in bar start times during duration calculation. This is unexpected."); continue # pragma: no cover
            
            bar_start_sec = bar_absolute_start_times_sec[bar_idx]
            flow_syl_starts_subdiv = line_flow_datum.get("syllable_start_subdivisions", [])
            flow_syl_durs_quantized = line_flow_datum.get("syllable_durations_quantized", [])
            
            if flow_syl_starts_subdiv and flow_syl_durs_quantized:
                # This logic needs to use the correct bar-specific timing for subdivision_duration_sec
                bar_bpm_line = bpm # Start with global
                sbf_bar_info = next((bf for bf in beat_info.get("sbf_features_for_timing_ref", []) if bf.get("bar_index") == bar_idx), None) if "sbf_features_for_timing_ref" in beat_info else None
                if sbf_bar_info and sbf_bar_info.get("bpm", 0) > 0: bar_bpm_line = sbf_bar_info["bpm"] # pragma: no cover (covered by tests but flagged)
                
                bar_beat_dur_line = 60.0 / bar_bpm_line
                bar_duration_for_subdivs_line = sbf_bar_info.get("bar_duration_sec") if sbf_bar_info else None # pragma: no cover (covered by tests but flagged)
                if not bar_duration_for_subdivs_line or bar_duration_for_subdivs_line <= 0.1: bar_duration_for_subdivs_line = beat_info.get("estimated_bar_duration", beat_info.get("beats_per_bar", 4) * bar_beat_dur_line) # pragma: no cover (covered by tests but flagged)
                bar_duration_for_subdivs_line = max(0.1, bar_duration_for_subdivs_line)
                subdiv_dur_sec_line = bar_duration_for_subdivs_line / max(1, self.flow_tokenizer.max_subdivisions)

                last_syl_idx = len(flow_syl_starts_subdiv) - 1
                if last_syl_idx >= 0:
                    syl_start_offset_sec = flow_syl_starts_subdiv[last_syl_idx] * subdiv_dur_sec_line
                    syl_dur_beats = self.flow_tokenizer.dequantize_syllable_duration_bin(flow_syl_durs_quantized[last_syl_idx])
                    syl_actual_dur_sec = max(0.020, syl_dur_beats * bar_beat_dur_line)
                    max_event_time_sec = max(max_event_time_sec, bar_start_sec + syl_start_offset_sec + syl_actual_dur_sec)
            else: # Fallback if no per-syllable details (should not happen with valid FlowData)
                line_dur_beats = line_flow_datum.get("duration_beats", 0.1) # pragma: no cover
                max_event_time_sec = max(max_event_time_sec, bar_start_sec + line_flow_datum.get("start_offset_beats",0)*beat_duration_sec + line_dur_beats*beat_duration_sec) # pragma: no cover

        output_duration_sec = max(max_event_time_sec + 1.5, 0.2) # Add 1.5s buffer
        final_sr = self.sample_rate
        vocal_track_pydub = pydub.AudioSegment.silent(duration=int(output_duration_sec * 1000), frame_rate=final_sr)

        # Cache raw audio for unique lyric lines
        unique_lyric_lines: Set[str] = set(line_txt.strip() for line_txt in lyric_lines_for_tts if line_txt.strip())
        raw_line_audio_cache: Dict[str, Optional[pydub.AudioSegment]] = {}

        if unique_lyric_lines:
            print(f"{LOG_PREFIX_SYNTH}   Generating espeak-ng audio for {len(unique_lyric_lines)} unique lyric lines...")
            tts_cache_start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers_tts) as executor:
                futures_map = {executor.submit(self._generate_line_audio_espeak, line_txt): line_txt for line_txt in unique_lyric_lines}
                for future in concurrent.futures.as_completed(futures_map):
                    line_txt, segment = future.result()
                    raw_line_audio_cache[line_txt] = segment
            print(f"{LOG_PREFIX_SYNTH}   espeak-ng audio for lines cached in {time.time() - tts_cache_start_time:.2f}s.")

        placement_start_time = time.time(); total_syllables_placed_on_track = 0
        
        for line_idx, lyric_line_text_for_tts in enumerate(lyric_lines_for_tts):
            line_flow_datum = flow_data_for_lyrics[line_idx]
            target_num_syllables_in_line = line_flow_datum.get("syllables", 0)

            if target_num_syllables_in_line == 0 or not lyric_line_text_for_tts.strip(): continue # Skip silent lines or empty lyrics

            espeak_line_audio_segment = raw_line_audio_cache.get(lyric_line_text_for_tts.strip())
            if not espeak_line_audio_segment or len(espeak_line_audio_segment) == 0:
                print(f"{LOG_PREFIX_SYNTH}     Skipping line {line_idx+1} ('{lyric_line_text_for_tts[:30]}...') as espeak-ng synthesis failed or was empty."); continue

            bar_idx = line_flow_datum.get("bar_index")
            if bar_idx is None or bar_idx not in bar_absolute_start_times_sec: print(f"{LOG_PREFIX_SYNTH}   Warning: Line {line_idx+1} - Bar index {bar_idx} missing from timings. Skipping."); continue # pragma: no cover
            
            bar_start_sec = bar_absolute_start_times_sec[bar_idx]
            flow_syl_starts_subdiv = line_flow_datum.get("syllable_start_subdivisions", [])
            flow_syl_durs_quantized = line_flow_datum.get("syllable_durations_quantized", [])
            flow_syl_stresses = line_flow_datum.get("syllable_stresses", [])

            if not (len(flow_syl_starts_subdiv) == target_num_syllables_in_line and \
                    len(flow_syl_durs_quantized) == target_num_syllables_in_line and \
                    len(flow_syl_stresses) == target_num_syllables_in_line):
                print(f"{LOG_PREFIX_SYNTH}   Warning: Line {line_idx+1} - Incomplete per-syllable timing/stress in FlowData. Skipping."); continue # pragma: no cover

            # Determine bar-specific timing
            bar_bpm_line = bpm; sbf_bar_info_line = next((bf for bf in beat_info.get("sbf_features_for_timing_ref", []) if bf.get("bar_index") == bar_idx), None) if "sbf_features_for_timing_ref" in beat_info else None
            if sbf_bar_info_line and sbf_bar_info_line.get("bpm", 0) > 0: bar_bpm_line = sbf_bar_info_line["bpm"] # pragma: no cover
            bar_beat_dur_line = 60.0 / bar_bpm_line
            bar_duration_for_subdivs_line = sbf_bar_info_line.get("bar_duration_sec") if sbf_bar_info_line else None # pragma: no cover
            if not bar_duration_for_subdivs_line or bar_duration_for_subdivs_line <= 0.1: bar_duration_for_subdivs_line = beat_info.get("estimated_bar_duration", beat_info.get("beats_per_bar", 4) * bar_beat_dur_line) # pragma: no cover
            bar_duration_for_subdivs_line = max(0.1, bar_duration_for_subdivs_line)
            subdivision_duration_sec_line = bar_duration_for_subdivs_line / max(1, self.flow_tokenizer.max_subdivisions)

            # Calculate total duration this line should occupy based on FlowData, relative to first syllable start
            first_syl_start_offset_in_bar_sec = flow_syl_starts_subdiv[0] * subdivision_duration_sec_line
            last_syl_start_offset_in_bar_sec = flow_syl_starts_subdiv[-1] * subdivision_duration_sec_line
            last_syl_dur_beats = self.flow_tokenizer.dequantize_syllable_duration_bin(flow_syl_durs_quantized[-1])
            last_syl_dur_sec = max(0.020, last_syl_dur_beats * bar_beat_dur_line)
            total_flow_duration_for_line_sec = (last_syl_start_offset_in_bar_sec + last_syl_dur_sec) - first_syl_start_offset_in_bar_sec
            total_flow_duration_for_line_sec = max(0.020 * target_num_syllables_in_line, total_flow_duration_for_line_sec) # Ensure minimal duration

            # Stretch/compress the entire synthesized line audio to match this total flow duration
            stretched_line_audio = self._adjust_segment_duration_with_speedup(espeak_line_audio_segment, total_flow_duration_for_line_sec * 1000)
            stretched_line_audio_np, stretched_line_sr = self._segment_to_numpy(stretched_line_audio)

            # Now, segment this stretched_line_audio into individual syllables
            for syl_idx in range(target_num_syllables_in_line):
                # Calculate this syllable's start time and duration *relative to the start of the FlowData line*
                syl_start_in_flow_relative_sec = (flow_syl_starts_subdiv[syl_idx] * subdivision_duration_sec_line) - first_syl_start_offset_in_bar_sec
                syl_dur_beats = self.flow_tokenizer.dequantize_syllable_duration_bin(flow_syl_durs_quantized[syl_idx])
                syl_target_dur_sec = max(0.020, syl_dur_beats * bar_beat_dur_line)

                # Extract the segment from the stretched_line_audio_np
                start_sample = int(syl_start_in_flow_relative_sec * stretched_line_sr)
                end_sample = int((syl_start_in_flow_relative_sec + syl_target_dur_sec) * stretched_line_sr)
                
                # Ensure start_sample and end_sample are valid for stretched_line_audio_np
                start_sample = max(0, min(start_sample, len(stretched_line_audio_np)))
                end_sample = max(start_sample, min(end_sample, len(stretched_line_audio_np)))

                if start_sample >= end_sample : # Segment is zero or negative length
                    # print(f"{LOG_PREFIX_SYNTH}     Syllable {syl_idx+1} in line {line_idx+1} resulted in zero-length segment. Skipping.")
                    continue

                syllable_audio_np = stretched_line_audio_np[start_sample:end_sample].copy()
                
                # Apply stress-based pitch and gain modulation
                syl_stress = flow_syl_stresses[syl_idx]
                pitch_shift_semitones = 0.0
                gain_factor = 1.0
                if syl_stress == 1: # Primary stress
                    pitch_shift_semitones = self.primary_stress_pitch_shift
                    gain_factor = self.primary_stress_gain_factor
                elif syl_stress == 2: # Secondary stress
                    pitch_shift_semitones = self.secondary_stress_pitch_shift
                    gain_factor = self.secondary_stress_gain_factor
                
                if abs(pitch_shift_semitones) > 0.01 and syllable_audio_np.size > 0:
                    try:
                        syllable_audio_np = librosa.effects.pitch_shift(y=syllable_audio_np, sr=stretched_line_sr, n_steps=pitch_shift_semitones, bins_per_octave=12)
                    except Exception as e_ps: # pragma: no cover
                        print(f"{LOG_PREFIX_SYNTH} Warning: Pitch shift failed for syllable {syl_idx+1} in line {line_idx+1}: {e_ps}")

                syllable_audio_np *= gain_factor
                
                syllable_segment_pydub = self._numpy_to_segment(syllable_audio_np, stretched_line_sr)
                if syllable_segment_fade_ms > 0 and len(syllable_segment_pydub) > syllable_segment_fade_ms * 2:
                    syllable_segment_pydub = syllable_segment_pydub.fade_in(syllable_segment_fade_ms).fade_out(syllable_segment_fade_ms)
                elif syllable_segment_fade_ms > 0 and len(syllable_segment_pydub) > syllable_segment_fade_ms: # pragma: no cover
                    syllable_segment_pydub = syllable_segment_pydub.fade_out(min(syllable_segment_fade_ms, len(syllable_segment_pydub)//2))
                
                # Absolute placement time on the main vocal track
                abs_placement_time_sec = bar_start_sec + (flow_syl_starts_subdiv[syl_idx] * subdivision_duration_sec_line)
                vocal_track_pydub = vocal_track_pydub.overlay(
                    syllable_segment_pydub, position=int(abs_placement_time_sec * 1000)
                )
                total_syllables_placed_on_track +=1
        
        print(f"{LOG_PREFIX_SYNTH}   Vocal audio placement completed in {time.time() - placement_start_time:.2f}s.")
        if total_syllables_placed_on_track == 0: print(f"{LOG_PREFIX_SYNTH}   No syllables were successfully synthesized or placed."); return None # pragma: no cover

        vocal_track_pydub = vocal_track_pydub + vocal_level_db
        vocals_final_np, sr_final = self._segment_to_numpy(vocal_track_pydub)
        print(f"{LOG_PREFIX_SYNTH}   Vocal track synthesis finished in {time.time() - s_time:.2f}s total. Placed {total_syllables_placed_on_track} syllables.")
        
        # === CONCEPTUAL INTEGRATION POINT for RVC/Advanced Vocal Model ===
        # At this point, `vocals_final_np` contains the espeak-ng based vocals, timed and segmented.
        # This is where you would pass `vocals_final_np` (and `sr_final`) to an RVC model
        # or a higher-quality TTS/SVS if one was available and integrated.
        # Example:
        # if rvc_model_is_available:
        #    print(f"{LOG_PREFIX_SYNTH} Applying RVC model for vocal enhancement...")
        #    vocals_final_np = rvc_model.process(vocals_final_np, sr_final, target_voice_speaker_embedding)
        # else:
        #    print(f"{LOG_PREFIX_SYNTH} RVC model not available/configured. Using base synthesized vocals.")
        # For now, we pass the espeak-ng (with stress modulation) vocals through.
        # =================================================================

        return (vocals_final_np, sr_final) 

    def synthesize_verse_with_instrumental(
        self,
        phonetic_vocals_data: AudioData, 
        instrumental_audio: Optional[AudioData] = None,
        instrumental_level_db: float = -20.0 
    ) -> Optional[AudioData]:
        phonetic_vocals_np, vocals_sr = phonetic_vocals_data
        if vocals_sr != self.sample_rate: # pragma: no cover
            print(f"{LOG_PREFIX_SYNTH} Warning: Vocal sample rate {vocals_sr} differs from target {self.sample_rate}. Resampling vocals.")
            phonetic_vocals_np = librosa.resample(phonetic_vocals_np, orig_sr=vocals_sr, target_sr=self.sample_rate)

        max_vocal_abs = np.max(np.abs(phonetic_vocals_np)) if phonetic_vocals_np.size > 0 else 0
        norm_vocals_np = (phonetic_vocals_np / max_vocal_abs * 0.95) if max_vocal_abs > 1e-5 else phonetic_vocals_np

        output_duration_sec = len(norm_vocals_np) / self.sample_rate
        if instrumental_audio and instrumental_audio[0].size > 0: # pragma: no cover
             instr_duration_sec = len(instrumental_audio[0]) / instrumental_audio[1] 
             output_duration_sec = max(output_duration_sec, instr_duration_sec)
        output_duration_sec = max(output_duration_sec, 0.1) 

        processed_instrumental_np = self._get_processed_instrumental_np(instrumental_audio, output_duration_sec)

        current_len_vocals = len(norm_vocals_np)
        current_len_instr = len(processed_instrumental_np)
        target_len_samples = max(current_len_vocals, current_len_instr, int(output_duration_sec * self.sample_rate))

        if current_len_vocals < target_len_samples: norm_vocals_np = np.pad(norm_vocals_np, (0, target_len_samples - current_len_vocals), 'constant')
        elif current_len_vocals > target_len_samples: norm_vocals_np = norm_vocals_np[:target_len_samples] # pragma: no cover

        if current_len_instr < target_len_samples: processed_instrumental_np = np.pad(processed_instrumental_np, (0, target_len_samples - current_len_instr), 'constant')
        elif current_len_instr > target_len_samples: processed_instrumental_np = processed_instrumental_np[:target_len_samples] # pragma: no cover
        
        max_instr_abs = np.max(np.abs(processed_instrumental_np)) if processed_instrumental_np.size > 0 else 0
        norm_instrumental_np = (processed_instrumental_np / max_instr_abs * 0.95) if max_instr_abs > 1e-5 else processed_instrumental_np

        instr_gain_linear = 10 ** (instrumental_level_db / 20.0)
        adjusted_instrumental_np = norm_instrumental_np * instr_gain_linear
            
        final_mixed_np = norm_vocals_np + adjusted_instrumental_np 
            
        max_mixed_amp = np.max(np.abs(final_mixed_np)) if final_mixed_np.size > 0 else 0
        if max_mixed_amp > 0.98: final_mixed_np = final_mixed_np / max_mixed_amp * 0.977
        elif max_mixed_amp == 0 and np.max(np.abs(norm_vocals_np)) == 0 and np.max(np.abs(adjusted_instrumental_np)) == 0: # pragma: no cover
            print(f"{LOG_PREFIX_SYNTH} Warning: Final mix is completely silent as both sources were silent.")
            return (np.zeros(target_len_samples, dtype=np.float32), self.sample_rate)

        print(f"{LOG_PREFIX_SYNTH} Vocals mixed with instrumental (instrumental at {instrumental_level_db}dB relative to normalized vocals).")
        return (final_mixed_np, self.sample_rate)

    def save_audio(self, audio_data: AudioData, file_path: str, format: str = "wav"):
        waveform, sr = audio_data
        try:
            ensure_dir(os.path.dirname(file_path)) 
            
            waveform_float = waveform.astype(np.float32) if not np.issubdtype(waveform.dtype, np.floating) else waveform # pragma: no cover (usually float already)
            if np.issubdtype(waveform.dtype, np.integer) and not np.issubdtype(waveform_float.dtype, np.floating): # pragma: no cover (double check)
                 max_possible_val = np.iinfo(waveform.dtype).max
                 if max_possible_val > 0: waveform_float = waveform_float / max_possible_val
            
            max_abs_val_final = np.max(np.abs(waveform_float)) if waveform_float.size > 0 else 0
            if max_abs_val_final > 0.988: waveform_float = waveform_float / max_abs_val_final * 0.988 
            elif max_abs_val_final == 0: print(f"{LOG_PREFIX_SYNTH} Saving completely silent audio to {file_path}.") # pragma: no cover
            
            waveform_contiguous = np.ascontiguousarray(waveform_float)

            if format.lower() == "wav":
                sf.write(file_path, waveform_contiguous, sr, subtype='PCM_16') 
            elif format.lower() == "mp3":
                samples_int16 = (waveform_contiguous * 32767).astype(np.int16)
                byte_data = samples_int16.tobytes()
                if not byte_data and len(waveform_contiguous) > 0: audio_segment = pydub.AudioSegment.silent(duration=int(len(waveform_contiguous)/sr * 1000) , frame_rate=sr) # pragma: no cover
                elif not byte_data and len(waveform_contiguous) == 0: audio_segment = pydub.AudioSegment.silent(duration=0, frame_rate=sr) # pragma: no cover
                else:
                    audio_segment = pydub.AudioSegment( data=byte_data, frame_rate=sr, sample_width=2, channels=1 )
                audio_segment.export(file_path, format="mp3", bitrate="192k") 
            else: # pragma: no cover
                print(f"{LOG_PREFIX_SYNTH} Unsupported audio format: {format}. Saving as WAV (16-bit PCM).")
                new_file_path = os.path.splitext(file_path)[0] + ".wav"
                sf.write(new_file_path, waveform_contiguous, sr, subtype='PCM_16')
                file_path = new_file_path 
            
            print(f"{LOG_PREFIX_SYNTH} Audio saved to {file_path} (format: {os.path.splitext(file_path)[1][1:].lower()})")
        except Exception as e: # pragma: no cover
            print(f"{LOG_PREFIX_SYNTH} Error saving audio to {file_path}: {e}")
            import traceback; traceback.print_exc()