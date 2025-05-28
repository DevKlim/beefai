import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, List

# Assuming FlowTokenizer is in the same directory or accessible via beefai.flow_model
# from .tokenizer import FlowTokenizer # For type hinting if passed to generate

class FlowGPTConfig:
    def __init__(self, 
                 vocab_size: int,             # Size of the token vocabulary
                 block_size: int,             # Max sequence length (context window)
                 n_layer: int = 6,            # Number of Transformer layers
                 n_head: int = 6,             # Number of attention heads
                 n_embd: int = 384,           # Embedding dimension (must be div by n_head)
                 # Context ID vocabulary sizes (max distinct values for these IDs)
                 max_segment_types: int = 16, # Max number of segments (e.g., beat_features, flow_line_1, flow_line_2...)
                 max_intra_line_positions: int = 32, # Max positions within a segment component
                                                      # (e.g., for bar_features: BAR_START, BPM, TS, KICK_AT_0, ...)
                                                      # (e.g., for flow_line: LINE_START, SYL, OFF, DUR)
                 dropout: float = 0.1, 
                 bias: bool = True,           # True: bias in Linears and LayerNorms
                 pad_token_id: Optional[int] = None): # Store pad_token_id for loss ignore_index
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_segment_types = max_segment_types
        self.max_intra_line_positions = max_intra_line_positions
        self.dropout = dropout
        self.bias = bias
        self.pad_token_id = pad_token_id


class CausalSelfAttention(nn.Module):
    def __init__(self, config: FlowGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Flash Attention for PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)**0.5)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: FlowGPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: FlowGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class FlowTransformerDecoder(nn.Module):
    def __init__(self, config: FlowGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),              # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),              # Absolute positional embeddings
            # Contextual Embeddings:
            wse = nn.Embedding(config.max_segment_types, config.n_embd),       # Segment embeddings (e.g., beat vs flow line 1 vs flow line 2)
            wipe = nn.Embedding(config.max_intra_line_positions, config.n_embd),# Intra-segment/line positional embeddings
            
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/((2 * config.n_layer)**0.5))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                idx: torch.Tensor,                      # Input token IDs (B, T)
                segment_ids: Optional[torch.Tensor] = None, # Segment type IDs (B, T)
                intra_line_pos_ids: Optional[torch.Tensor] = None, # Intra-segment position IDs (B, T)
                targets: Optional[torch.Tensor] = None    # Target token IDs for loss calculation (B, T)
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward seq of length {t}, block size is {self.config.block_size}"
        
        absolute_pos = torch.arange(0, t, dtype=torch.long, device=device) # (T)
        
        tok_emb = self.transformer.wte(idx) # (B, T, C) token embeddings
        abs_pos_emb = self.transformer.wpe(absolute_pos) # (T, C) absolute positional embeddings
        
        # Contextual embeddings
        if segment_ids is None: # Default if not provided (though should be for this model)
            segment_ids = torch.zeros_like(idx, dtype=torch.long)
        seg_emb = self.transformer.wse(segment_ids) # (B, T, C)
        
        if intra_line_pos_ids is None: # Default
            intra_line_pos_ids = torch.zeros_like(idx, dtype=torch.long)
        intra_pos_emb = self.transformer.wipe(intra_line_pos_ids) # (B, T, C)
        
        # Combine embeddings
        x = tok_emb + abs_pos_emb + seg_emb + intra_pos_emb
        x = self.transformer.drop(x)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None: # Training or evaluation with targets
            logits = self.lm_head(x) # (B, T, VocabSize)
            # Use pad_token_id from config if available for ignore_index
            ignore_idx = self.config.pad_token_id if self.config.pad_token_id is not None else -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_idx)
        else: # Inference, only compute logits for the last token
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, VocabSize)
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(self, 
                 idx_prompt: torch.Tensor,  # (B, T_prompt) initial prompt tokens
                 segment_ids_prompt: torch.Tensor, # (B, T_prompt)
                 intra_line_pos_ids_prompt: torch.Tensor, # (B, T_prompt)
                 max_new_tokens: int, 
                 tokenizer: 'FlowTokenizer', # Used by get_next_context_ids_for_token
                 temperature: float = 1.0, 
                 top_k: Optional[int] = None
                ) -> torch.Tensor:
        self.eval()
        
        if idx_prompt.size(0) != 1:
            # Current get_next_context_ids_for_token is simpler with B=1
            raise NotImplementedError("Generation currently supports batch size 1 for simplicity of context ID management.")

        # Make mutable copies for extension
        current_token_ids = idx_prompt.clone()
        current_segment_ids = segment_ids_prompt.clone()
        current_intra_pos_ids = intra_line_pos_ids_prompt.clone()

        for _ in range(max_new_tokens):
            # Crop inputs to block_size if they grow too long
            idx_cond = current_token_ids if current_token_ids.size(1) <= self.config.block_size else current_token_ids[:, -self.config.block_size:]
            seg_ids_cond = current_segment_ids if current_segment_ids.size(1) <= self.config.block_size else current_segment_ids[:, -self.config.block_size:]
            intra_pos_ids_cond = current_intra_pos_ids if current_intra_pos_ids.size(1) <= self.config.block_size else current_intra_pos_ids[:, -self.config.block_size:]

            logits, _ = self(idx_cond, segment_ids=seg_ids_cond, intra_line_pos_ids=intra_pos_ids_cond)
            logits = logits[:, -1, :] / temperature # Get last token logits, apply temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # Apply top-k filtering
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # --- Determine context IDs for the newly generated token `idx_next` ---
            # This requires the history of tokens *before* idx_next was appended
            # For B=1:
            history_token_list_for_ctx = current_token_ids[0].tolist()
            
            next_seg_id_val, next_intra_pos_id_val = get_next_context_ids_for_token(
                history_token_list_for_ctx, # Full sequence generated so far (before idx_next)
                idx_next.item(),            # The token ID being added
                tokenizer, 
                self.config.max_segment_types, 
                self.config.max_intra_line_positions
            )
            
            # Append new token and its context IDs
            current_token_ids = torch.cat((current_token_ids, idx_next), dim=1)
            current_segment_ids = torch.cat((current_segment_ids, torch.tensor([[next_seg_id_val]], dtype=torch.long, device=idx_prompt.device)), dim=1)
            current_intra_pos_ids = torch.cat((current_intra_pos_ids, torch.tensor([[next_intra_pos_id_val]], dtype=torch.long, device=idx_prompt.device)), dim=1)
            
            if idx_next.item() == tokenizer.eos_token_id:
                break
        
        self.train() # Set back to train mode
        return current_token_ids


# Helper function for generation: determines context IDs for the NEXT token to be generated.
# This logic should mirror how segment/intra-line-pos IDs are created during dataset preparation.
def get_next_context_ids_for_token(
    previous_token_ids_in_sequence: List[int], # Full list of token IDs generated so far
    token_just_added_id: int,                 # The ID of the token that was *just* added to the sequence
                                              # (or the current token if pre-filling context for a prompt)
    tokenizer: 'FlowTokenizer', 
    max_segment_types: int, 
    max_intra_line_positions: int
    ) -> Tuple[int, int]:
    """
    Determines the segment_id and intra_line_pos_id for `token_just_added_id`
    based on `previous_token_ids_in_sequence` (which does NOT YET include `token_just_added_id`).
    This function is called to determine context for a token *as it's being decided or added*.
    """
    
    # The state is determined by the sequence *before* the current token is finalized.
    # Example: if sequence is [BOS, BAR_START], and we are deciding context for BPM_TOKEN,
    # previous_token_ids_in_sequence = [BOS, BAR_START], token_just_added_id = BPM_TOKEN_ID
    
    # If previous_token_ids_in_sequence is empty, this is the first token (likely BOS)
    if not previous_token_ids_in_sequence:
        # BOS token context
        return 0, 0 # Segment 0, Intra-Pos 0

    last_actual_token_id = previous_token_ids_in_sequence[-1]
    
    # Default to current segment and increment intra-position
    # This needs to be smarter based on token types.
    # Let's reconstruct state by looking at the `previous_token_ids_in_sequence`
    
    current_segment_val = 0
    current_intra_pos_val = 0
    
    # Scan backwards to find the start of the current logical "block"
    # (e.g. BAR_START, SEP_INPUT_FLOW, LINE_START) to determine context.

    # This is a simplified heuristic. A more robust way is to simulate the
    # segment/intra-pos ID generation process from the tokenizer's `encode_song_instance`
    # for the `previous_token_ids_in_sequence` to get the *current* state,
    # then decide the *next* state based on `token_just_added_id`.
    
    # Simplified logic for generation:
    # Assume that the prompt (idx_prompt, etc.) already has correct context IDs.
    # When we generate a new token, its context depends on the *last generated token's context*
    # and the *type* of the new token.

    # Find the segment type of the last token in `previous_token_ids_in_sequence`
    # This requires having the full segment_ids and intra_line_pos_ids for the `previous_token_ids_in_sequence`
    # available during generation. The `generate` function passes these.
    # The logic here should determine the *next* segment/intra-pos based on the current token being added.

    # If `token_just_added_id` is BOS: (should be handled by prompt)
    if token_just_added_id == tokenizer.bos_token_id: return 0, 0
    
    # If `token_just_added_id` is BAR_START:
    #   This usually starts a new "beat features" segment or continues one.
    #   If previous was EOS, then this is effectively segment 0 (or a new song segment).
    #   If previous was related to a flow line, this BAR_START signifies a new bar's features.
    #   The segment_id logic in `encode_song_instance` increments `current_segment_idx` after flow lines
    #   and before the next BAR_START. So, this new BAR_START would get that incremented segment_idx.
    #   Intra-pos would be 0.
    # (This logic is complex to perfectly replicate here without full state from encoding.
    #  The `generate` function must maintain and pass the *full history* of context IDs.)

    # For the `generate` function, it calls this with `history_token_list_for_ctx` (tokens before new one)
    # and `idx_next.item()` (the new token).
    # The simplest way for `generate` is to actually re-tokenize its current sequence to get the
    # next expected context IDs. However, that's inefficient.
    
    # Let's assume a stateful progression based on special tokens for this helper:
    # This requires the full `previous_token_ids_in_sequence` to determine the current state.
    
    # Simulate the `encode_song_instance` logic to find the current context
    # This is a bit redundant but shows the principle. In practice, `generate` would track this.
    
    _ , temp_seg_ids, temp_intra_pos_ids = tokenizer.encode_song_instance(
        [], # dummy beat features, not used for this purpose if we only look at tokens
        []  # dummy flow data
    ) # This is not quite right. We need to parse `previous_token_ids_in_sequence`.

    # --- More direct approach for the helper, assuming it's called sequentially ---
    # This helper's role is: given the history, what are the context IDs for the *next* token.
    # The `tokenizer.encode_song_instance` ALREADY calculates these.
    # So, if `generate` calls this, it should pass what it believes are the
    # current running segment_id and intra_line_pos_id.

    # The `get_next_context_ids_for_token` is tricky because its ideal implementation
    # depends on how `generate` manages and passes the history of context IDs.
    # The `FlowTokenizer.encode_song_instance` is the source of truth for these IDs.
    # For generation, we need to predict the *next* context IDs if the model were to output `token_just_added_id`.

    # Let's refine the logic in `generate` to manage this better.
    # The helper should assume it has the *current* context and predicts the *next* one.
    # Or, it re-derives context based on the stream of tokens.

    # Re-derivation approach (can be slow but robust for a helper):
    seg_val = 0
    intra_val = 0
    
    # Find last SEP_INPUT_FLOW
    last_sep_idx = -1
    for i in range(len(previous_token_ids_in_sequence) -1, -1, -1):
        if previous_token_ids_in_sequence[i] == tokenizer.sep_input_flow_token_id:
            last_sep_idx = i
            break
            
    if token_just_added_id == tokenizer.sep_input_flow_token_id:
        # Figure out what segment SEP belongs to. It's after a full bar_feature block.
        # If previous_token_ids_in_sequence was [BOS, BAR_START, BPM, ..., END_BASS_EVENTS]
        # then SEP starts a new segment.
        # Search for BAR_START to count segments approximately.
        num_bar_starts_before = sum(1 for t_id in previous_token_ids_in_sequence if t_id == tokenizer.bar_start_token_id)
        seg_val = num_bar_starts_before * 2 # Roughly: BarFeatSeg, FlowSeg for each bar
        intra_val = 0
    elif token_just_added_id == tokenizer.line_start_token_id:
        # This starts a flow line. It must be in a "flow" segment.
        num_bar_starts_before = sum(1 for t_id in previous_token_ids_in_sequence if t_id == tokenizer.bar_start_token_id)
        seg_val = (num_bar_starts_before -1) * 2 + 1 # Segment for flow of current bar
        # Intra-pos needs to count line_starts *within this segment*
        intra_val = 0 # LINE_START is pos 0 of its components
    elif token_just_added_id == tokenizer.bar_start_token_id:
        num_bar_starts_before = sum(1 for t_id in previous_token_ids_in_sequence if t_id == tokenizer.bar_start_token_id)
        seg_val = num_bar_starts_before * 2 # This new bar_start begins a new feature segment
        intra_val = 0
    else: # Regular token within a bar feature set or a flow line
        # Find the most recent structural token to determine current segment type
        current_block_type = "beat" # Default
        start_of_current_block_idx = 0
        
        for i in range(len(previous_token_ids_in_sequence) - 1, -1, -1):
            tok = previous_token_ids_in_sequence[i]
            if tok == tokenizer.bar_start_token_id:
                current_block_type = "beat"
                start_of_current_block_idx = i
                num_bar_starts_before = sum(1 for t_id in previous_token_ids_in_sequence[:i+1] if t_id == tokenizer.bar_start_token_id)
                seg_val = (num_bar_starts_before -1) * 2
                break
            if tok == tokenizer.sep_input_flow_token_id:
                current_block_type = "sep" # Transitioning to flow
                start_of_current_block_idx = i
                num_bar_starts_before = sum(1 for t_id in previous_token_ids_in_sequence[:i+1] if t_id == tokenizer.bar_start_token_id)
                seg_val = (num_bar_starts_before -1) * 2 +1 # Segment for flow of current bar
                break
            if tok == tokenizer.line_start_token_id:
                current_block_type = "flow_line"
                start_of_current_block_idx = i
                # Segment for this flow line is determined by the SEP before it
                # Find the SEP that governs this line_start
                temp_sep_idx = -1
                for j in range(i -1, -1, -1):
                    if previous_token_ids_in_sequence[j] == tokenizer.sep_input_flow_token_id:
                        temp_sep_idx = j
                        break
                if temp_sep_idx != -1:
                    num_bar_starts_before_sep = sum(1 for t_id in previous_token_ids_in_sequence[:temp_sep_idx+1] if t_id == tokenizer.bar_start_token_id)
                    seg_val = (num_bar_starts_before_sep -1) * 2 + 1
                else: # Should not happen if structure is [BAR_START...SEP...LINE_START]
                    seg_val = 1 
                break
        
        intra_val = (len(previous_token_ids_in_sequence) - 1) - start_of_current_block_idx + 1

    final_segment_id = min(seg_val, max_segment_types - 1)
    final_intra_line_pos_id = min(intra_val, max_intra_line_positions - 1)
    
    return final_segment_id, final_intra_line_pos_id


if __name__ == '__main__':
    from beefai.flow_model.tokenizer import FlowTokenizer # Relative import
    
    # Create a tokenizer instance (ensure config file path is correct or it builds default)
    tokenizer_config_file = "flow_tokenizer_config_v2.json" # From tokenizer test
    tokenizer = FlowTokenizer(config_path=tokenizer_config_file)
    if not hasattr(tokenizer, 'bos_token_id'): # If vocab was not loaded/built
        print("Tokenizer vocab seems empty, attempting to build/save.")
        tokenizer._build_vocab() # Ensure vocab is built if file was missing
        tokenizer.save_vocab(tokenizer_config_file)

    vocab_size = tokenizer.get_vocab_size()
    block_size = 256  # Example
    
    # Ensure pad_token_id is set in config if tokenizer has one
    pad_id_for_loss = tokenizer.pad_token_id if "[PAD]" in tokenizer.token_to_id else -100

    config = FlowGPTConfig(
        vocab_size=vocab_size, 
        block_size=block_size,
        n_layer=2, n_head=2, n_embd=128, # Small for quick test
        max_segment_types=8, # Max segments expected by tokenizer's encode_song_instance logic
        max_intra_line_positions=20, # Max positions within a bar_feature list or flow_line parts
        pad_token_id=pad_id_for_loss
    )
    model = FlowTransformerDecoder(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    # --- Test generation logic with context IDs ---
    # 1. Create a sample prompt (beat features for one bar)
    sample_bar_feat: BarBeatFeatures = {
        "bar_index": 0, "bpm": 120.0, "time_signature": (4, 4),
        "kick_events": [0, 8], "snare_events": [4, 12], "hihat_events": [], "bass_events": [0]
    }
    
    # Encode this single bar feature set to get initial token and context IDs
    # The `encode_song_instance` is for a whole song. We need a prompt for generation.
    # Prompt typically is: BOS + BarFeatures + SEP_INPUT_FLOW
    
    prompt_token_ids_list = [tokenizer.bos_token_id]
    prompt_seg_ids_list = [0] # BOS is seg 0
    prompt_intra_pos_ids_list = [0] # BOS is intra-pos 0

    bar_feat_tokens = tokenizer.encode_bar_features(sample_bar_feat)
    prompt_token_ids_list.extend(bar_feat_tokens)
    prompt_seg_ids_list.extend([0] * len(bar_feat_tokens)) # Bar features are seg 0
    prompt_intra_pos_ids_list.extend(list(range(len(bar_feat_tokens))))

    prompt_token_ids_list.append(tokenizer.sep_input_flow_token_id)
    prompt_seg_ids_list.append(1) # SEP starts segment 1 (flow lines for this bar)
    prompt_intra_pos_ids_list.append(0) # SEP is intra-pos 0 of its segment

    idx_prompt = torch.tensor([prompt_token_ids_list], dtype=torch.long)
    seg_ids_prompt = torch.tensor([prompt_seg_ids_list], dtype=torch.long)
    intra_pos_ids_prompt = torch.tensor([prompt_intra_pos_ids_list], dtype=torch.long)

    print(f"\nGenerating from prompt (length {idx_prompt.size(1)}):")
    # Print prompt for verification
    for i in range(idx_prompt.size(1)):
        tok_str = tokenizer.id_to_token.get(idx_prompt[0,i].item(), "[UNK]")
        seg_str = f"S:{seg_ids_prompt[0,i].item()}"
        pos_str = f"P:{intra_pos_ids_prompt[0,i].item()}"
        print(f"  {tok_str:<20} {seg_str:<5} {pos_str:<5}")


    generated_ids_full = model.generate(
        idx_prompt, 
        seg_ids_prompt,
        intra_pos_ids_prompt,
        max_new_tokens=10, # Generate a few flow tokens (e.g., 2 lines worth = ~6-8 tokens)
        tokenizer=tokenizer,
        temperature=0.8,
        top_k=10
    )
    print("\nFull generated sequence (tokens):")
    # We need to reconstruct the context IDs for the generated part to display them
    # The `generate` function currently only returns token IDs.
    # For a full display, we'd need to re-run the context ID generation logic for the output.
    
    generated_sequence_list = generated_ids_full[0].tolist()
    print_tokens_with_inferred_context(generated_sequence_list, tokenizer, config)


def print_tokens_with_inferred_context(token_list, tokenizer, config):
    # Helper to display generated sequence with re-inferred context for debugging
    print("  Generated Token | Inferred Segment | Inferred Intra-Pos")
    print("  -------------------------------------------------------")
    
    history_for_ctx = []
    for token_id in token_list:
        seg_id, intra_id = get_next_context_ids_for_token(
            history_for_ctx, token_id, tokenizer, 
            config.max_segment_types, config.max_intra_line_positions
        )
        tok_str = tokenizer.id_to_token.get(token_id, "[UNK]")
        print(f"  {tok_str:<15} | {seg_id:<16} | {intra_id:<18}")
        history_for_ctx.append(token_id)