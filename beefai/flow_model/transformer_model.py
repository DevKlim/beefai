import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .tokenizer import FlowTokenizer # For type hinting if passed to generate
# Import for type hinting in __main__
from beefai.utils.data_types import BarBeatFeatures 


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
        self.dropout = dropout # This is the dropout probability (float)
        self.bias = bias
        self.pad_token_id = pad_token_id


class CausalSelfAttention(nn.Module):
    def __init__(self, config: FlowGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config # Store config

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout) 
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # print("Warning: Using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            # Use self.config.dropout for the dropout probability during training
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                                 dropout_p=self.config.dropout if self.training else 0.0, 
                                                                 is_causal=True)
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
            wse = nn.Embedding(config.max_segment_types, config.n_embd),       # Segment embeddings
            wipe = nn.Embedding(config.max_intra_line_positions, config.n_embd),# Intra-segment/line positional embeddings
            
            drop = nn.Dropout(config.dropout), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.vocab_size > 0 and config.n_embd > 0 : # Ensure valid dimensions for weight tying
            self.transformer.wte.weight = self.lm_head.weight 

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
        
        if segment_ids is None: 
            segment_ids = torch.zeros_like(idx, dtype=torch.long)
        seg_emb = self.transformer.wse(segment_ids) 
        
        if intra_line_pos_ids is None: 
            intra_line_pos_ids = torch.zeros_like(idx, dtype=torch.long)
        intra_pos_emb = self.transformer.wipe(intra_line_pos_ids)
        
        x = tok_emb + abs_pos_emb + seg_emb + intra_pos_emb
        x = self.transformer.drop(x) 
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None: 
            logits = self.lm_head(x) 
            ignore_idx = self.config.pad_token_id if self.config.pad_token_id is not None else -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=ignore_idx)
        else: 
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(self, 
                 idx_prompt: torch.Tensor,  
                 segment_ids_prompt: torch.Tensor, 
                 intra_line_pos_ids_prompt: torch.Tensor, 
                 max_new_tokens: int, 
                 tokenizer: 'FlowTokenizer', 
                 temperature: float = 1.0, 
                 top_k: Optional[int] = None
                ) -> torch.Tensor:
        self.eval() # Ensure model is in eval mode
        
        if idx_prompt.size(0) != 1:
            raise NotImplementedError("Generation currently supports batch size 1 for simplicity of context ID management.")

        current_token_ids = idx_prompt.clone()
        current_segment_ids = segment_ids_prompt.clone()
        current_intra_pos_ids = intra_line_pos_ids_prompt.clone()

        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = current_token_ids if current_token_ids.size(1) <= self.config.block_size else current_token_ids[:, -self.config.block_size:]
            seg_ids_cond = current_segment_ids if current_segment_ids.size(1) <= self.config.block_size else current_segment_ids[:, -self.config.block_size:]
            intra_pos_ids_cond = current_intra_pos_ids if current_intra_pos_ids.size(1) <= self.config.block_size else current_intra_pos_ids[:, -self.config.block_size:]

            logits, _ = self(idx_cond, segment_ids=seg_ids_cond, intra_line_pos_ids=intra_pos_ids_cond)
            logits = logits[:, -1, :] / temperature # Get logits for the last token, apply temperature
            
            if top_k is not None and top_k > 0: # Added top_k > 0 check
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') 
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) 
            
            # Determine context IDs for the newly generated token
            # Pass the *current* sequence of tokens (before adding idx_next) to get context for idx_next
            history_token_list_for_ctx = current_token_ids[0].tolist() 
            
            next_seg_id_val, next_intra_pos_id_val = get_next_context_ids_for_token(
                history_token_list_for_ctx, # Sequence leading up to the new token
                idx_next.item(),            # The new token itself
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
        
        # self.train() # Not strictly necessary if only used for inference after loading, but good practice if it might be trained later
        return current_token_ids


def get_next_context_ids_for_token(
    previous_token_ids_in_sequence: List[int], 
    token_being_added_id: int, # Changed name for clarity                 
    tokenizer: 'FlowTokenizer', 
    max_segment_types: int, 
    max_intra_line_positions: int
    ) -> Tuple[int, int]:
    
    # `previous_token_ids_in_sequence` is the state *before* `token_being_added_id` is appended.
    # We determine the context for `token_being_added_id` based on this history.
    
    # Default: if no history, new token is BOS-like (seg 0, pos 0)
    if not previous_token_ids_in_sequence:
        # This case implies token_being_added_id is the very first token (e.g. BOS)
        return 0, 0 

    last_token_in_history_id = previous_token_ids_in_sequence[-1]
    
    # Determine the segment and intra-position of the *last token in history*
    # to inform the context of the *token_being_added*.
    
    # Iteratively find the start of the block for the *last_token_in_history_id*
    # to determine its segment and intra-position.
    # This logic is tricky because segment IDs are not explicitly stored with history.
    # The robust way is to re-calculate segment context from the start of the current logical block.

    # Let's assume a simpler model for `get_next_context_ids_for_token` for now,
    # focusing on how special tokens change context. This is similar to the tokenizer's
    # `encode_song_instance` logic but in reverse/incrementally.

    # --- Simplified Context Logic for Generation ---
    # This logic determines the segment/pos for `token_being_added_id`
    # based on `last_token_in_history_id` and the type of `token_being_added_id`.
    
    # Find the segment of the *last token in history*
    # This requires iterating back from `last_token_in_history_id`
    # to find the most recent BAR_START, SEP_INPUT_FLOW, or LINE_START
    # to understand what "block" the `last_token_in_history_id` belonged to.

    # Re-scan `previous_token_ids_in_sequence` to determine context for `token_being_added_id`
    # This is similar to how `FlowTokenizer.encode_song_instance` builds contexts.

    _temp_full_seq = previous_token_ids_in_sequence + [token_being_added_id]
    
    current_seg_val = 0
    current_intra_val = 0
    
    # Count bar starts encountered *up to and including* the token that STARTS the current block
    # for `token_being_added_id`.
    
    # Find the start of the current "block" for `token_being_added_id`
    # by looking at `token_being_added_id` itself and what preceded it.
    
    if token_being_added_id == tokenizer.bos_token_id:
        current_seg_val = 0
        current_intra_val = 0
    elif token_being_added_id == tokenizer.bar_start_token_id:
        # This new BAR_START starts a new beat feature block.
        # Its segment ID depends on how many BAR_STARTs were *before it* in `previous_token_ids_in_sequence`.
        num_prior_bar_starts = sum(1 for t_id in previous_token_ids_in_sequence if t_id == tokenizer.bar_start_token_id)
        current_seg_val = num_prior_bar_starts * 2
        current_intra_val = 0 # BAR_START is pos 0 of its block
    elif token_being_added_id == tokenizer.sep_input_flow_token_id:
        # This new SEP starts a new flow block for the *current* bar.
        # Find the most recent BAR_START in `previous_token_ids_in_sequence`
        num_prior_bar_starts = 0
        for t_id in reversed(previous_token_ids_in_sequence):
            if t_id == tokenizer.bar_start_token_id:
                num_prior_bar_starts = sum(1 for tid_hist in previous_token_ids_in_sequence if tid_hist == tokenizer.bar_start_token_id)
                break
        else: # No BAR_START found before this SEP (should not happen if BOS is present and structure is Bar->Sep)
            # This implies it's like SEP for the first bar (bar 0)
            num_prior_bar_starts = 1 # Assuming it's for bar 0 if no explicit BAR_START before.
                                    # This needs to be robust to BOS -> SEP (if that's a valid sequence start)
                                    # If prev_tokens = [BOS], new = SEP. num_prior_bar_starts for BOS is 0.
                                    # If seg is (num_bars_so_far -1)*2 + 1. If BOS, num_bars_so_far=0. Should be seg 1.
            # Let's count bar_starts strictly in `previous_token_ids_in_sequence`
            num_bar_starts_in_history = sum(1 for t_id in previous_token_ids_in_sequence if t_id == tokenizer.bar_start_token_id)
            if num_bar_starts_in_history == 0 and tokenizer.bos_token_id in previous_token_ids_in_sequence:
                 # This SEP is likely for the implicit "first" bar after BOS, before any explicit BAR_START token.
                 # This case is tricky. Let's assume typical structure: BOS BAR_START ... SEP ...
                 # If prompt is just BOS, then SEP added:
                 # context for SEP should be based on "bar 0" context.
                 current_seg_val = 1 # Flow segment for the first bar (bar 0)
            else: # Usual case: BAR_START ... SEP
                 current_seg_val = (num_bar_starts_in_history -1) * 2 + 1 if num_bar_starts_in_history > 0 else 1

        current_intra_val = 0 # SEP is pos 0 of its flow block
    elif token_being_added_id == tokenizer.line_start_token_id:
        # This new LINE_START starts a new line within the *current* flow block.
        # Its segment ID is the same as the SEP that started this flow block.
        # Iterate backwards from `last_token_in_history_id` to find the governing SEP or BAR_START
        num_bar_starts_in_history_for_line = sum(1 for t_id in previous_token_ids_in_sequence if t_id == tokenizer.bar_start_token_id)
        current_seg_val = (num_bar_starts_in_history_for_line -1) * 2 + 1 if num_bar_starts_in_history_for_line > 0 else 1
        current_intra_val = 0 # LINE_START is pos 0 of its line tokens
    else: # Regular token (beat event, syllable, offset, duration, subdiv)
        # It belongs to the segment and position *after* `last_token_in_history_id`.
        # We need to determine the segment and intra-pos of `last_token_in_history_id`
        # and then increment intra-pos. Segment stays the same unless `last_token_in_history_id`
        # was the last token of a max-length intra-pos block (unlikely with typical token types).

        # Find the start of the block for `last_token_in_history_id`
        start_of_block_for_last_token_idx = 0
        seg_type_of_last_token_block = "beat" # default assumption
        
        # Determine segment for `last_token_in_history_id`
        # Count bar starts strictly before or at the block-defining token for `last_token_in_history_id`
        num_bar_starts_for_last_token_seg = 0

        for i in range(len(previous_token_ids_in_sequence) - 1, -1, -1):
            tok = previous_token_ids_in_sequence[i]
            if tok == tokenizer.bar_start_token_id:
                num_bar_starts_for_last_token_seg = sum(1 for t_id_hist in previous_token_ids_in_sequence[:i+1] if t_id_hist == tokenizer.bar_start_token_id)
                current_seg_val = (num_bar_starts_for_last_token_seg - 1) * 2 if num_bar_starts_for_last_token_seg > 0 else 0
                start_of_block_for_last_token_idx = i
                break
            elif tok == tokenizer.sep_input_flow_token_id:
                # Find BAR_START governing this SEP
                temp_bar_starts_for_sep = 0
                for k_sep in range(i -1, -1, -1):
                    if previous_token_ids_in_sequence[k_sep] == tokenizer.bar_start_token_id:
                        temp_bar_starts_for_sep = sum(1 for t_id_hist in previous_token_ids_in_sequence[:k_sep+1] if t_id_hist == tokenizer.bar_start_token_id)
                        break
                num_bar_starts_for_last_token_seg = temp_bar_starts_for_sep if temp_bar_starts_for_sep > 0 else (1 if tokenizer.bos_token_id in previous_token_ids_in_sequence[:i] else 0)

                current_seg_val = (num_bar_starts_for_last_token_seg -1) * 2 + 1 if num_bar_starts_for_last_token_seg > 0 else 1
                start_of_block_for_last_token_idx = i
                break
            elif tok == tokenizer.line_start_token_id:
                # Find SEP or BAR_START governing this LINE_START
                temp_bar_starts_for_line = 0
                # Find its governing SEP first
                sep_governing_line_idx = -1
                for k_line_sep in range(i -1, -1, -1):
                    if previous_token_ids_in_sequence[k_line_sep] == tokenizer.sep_input_flow_token_id:
                        sep_governing_line_idx = k_line_sep
                        break
                if sep_governing_line_idx != -1: # SEP found
                    for k_line_bar in range(sep_governing_line_idx -1, -1, -1):
                         if previous_token_ids_in_sequence[k_line_bar] == tokenizer.bar_start_token_id:
                            temp_bar_starts_for_line = sum(1 for t_id_hist in previous_token_ids_in_sequence[:k_line_bar+1] if t_id_hist == tokenizer.bar_start_token_id)
                            break
                    num_bar_starts_for_last_token_seg = temp_bar_starts_for_line if temp_bar_starts_for_line > 0 else (1 if tokenizer.bos_token_id in previous_token_ids_in_sequence[:sep_governing_line_idx] else 0)

                current_seg_val = (num_bar_starts_for_last_token_seg -1) * 2 + 1 if num_bar_starts_for_last_token_seg > 0 else 1
                start_of_block_for_last_token_idx = i
                break
            elif tok == tokenizer.bos_token_id: # Should be caught by `if not previous_token_ids_in_sequence` earlier for BOS itself
                current_seg_val = 0
                start_of_block_for_last_token_idx = i
                break
        else: # Only BOS was in history
            if previous_token_ids_in_sequence == [tokenizer.bos_token_id]:
                current_seg_val = 0 # token_being_added is after BOS, shares its segment
                start_of_block_for_last_token_idx = 0
            else: # Should not happen if previous_token_ids_in_sequence is not empty
                  # and doesn't contain BOS/BAR_START/SEP/LINE_START (e.g. just [UNK])
                  current_seg_val = 0 # Fallback
                  start_of_block_for_last_token_idx = 0


        # `current_seg_val` is now the segment of `last_token_in_history_id`'s block.
        # `token_being_added_id` inherits this segment.
        # `current_intra_val` is the position of `token_being_added_id` within this block.
        # It's the length of the current block from its start token up to `last_token_in_history_id`, plus 1.
        # len(previous_token_ids_in_sequence) gives total items up to `last_token_in_history_id`.
        # start_of_block_for_last_token_idx is the index of the token that started the block.
        # Number of items in the block so far = (len(previous_token_ids_in_sequence) - 1) - start_of_block_for_last_token_idx + 1
        # So, the intra_pos for `token_being_added_id` is this count.
        current_intra_val = (len(previous_token_ids_in_sequence) - 1) - start_of_block_for_last_token_idx + 1

    final_segment_id = min(current_seg_val, max_segment_types - 1)
    final_intra_line_pos_id = min(current_intra_val, max_intra_line_positions - 1)
    
    return final_segment_id, final_intra_line_pos_id

def print_tokens_with_inferred_context(token_list, tokenizer, gpt_config): 
    print("  Generated Token | Inferred Segment | Inferred Intra-Pos")
    print("  -------------------------------------------------------")
    
    history_for_ctx: List[int] = []
    for token_id in token_list:
        # When getting context for `token_id`, `history_for_ctx` is the sequence *before* it.
        seg_id, intra_id = get_next_context_ids_for_token(
            history_for_ctx, 
            token_id, # The token whose context we want to determine
            tokenizer, 
            gpt_config.max_segment_types, 
            gpt_config.max_intra_line_positions
        )
        tok_str = tokenizer.id_to_token.get(token_id, f"[UNK_ID:{token_id}]")
        print(f"  {tok_str:<15} | {seg_id:<16} | {intra_id:<18}")
        history_for_ctx.append(token_id) # Now add it to history for the *next* token


if __name__ == '__main__':
    from beefai.flow_model.tokenizer import FlowTokenizer # Local import for __main__
    import os 
    
    # Adjust path to be relative to this file's location if needed
    tokenizer_config_file = os.path.join(os.path.dirname(__file__), "flow_tokenizer_config_v2.json")
    if not os.path.exists(tokenizer_config_file):
        # Attempt to find it in a standard project location if running script from root
        proj_root_tokenizer_config = os.path.join("beefai", "flow_model", "flow_tokenizer_config_v2.json")
        if os.path.exists(proj_root_tokenizer_config):
            tokenizer_config_file = proj_root_tokenizer_config
        else:
            print(f"ERROR: Tokenizer config not found at '{tokenizer_config_file}' or '{proj_root_tokenizer_config}'.")
            print("Please ensure the tokenizer config exists. You might need to run FlowTokenizer main once.")
            exit()
            
    tokenizer = FlowTokenizer(config_path=tokenizer_config_file)
    print(f"Tokenizer loaded from {tokenizer_config_file}. Vocab size: {tokenizer.get_vocab_size()}")

    vocab_size = tokenizer.get_vocab_size()
    block_size = 256 
    
    pad_id_for_loss = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100

    # Dummy model config for testing generation context logic
    # In real use, this comes from model_config_*.yaml
    gpt_config_instance = FlowGPTConfig(
        vocab_size=vocab_size, 
        block_size=block_size,
        n_layer=2, n_head=2, n_embd=128, 
        max_segment_types=16, # Example value, should match training
        max_intra_line_positions=32, # Example value, should match training
        dropout=0.1,
        bias=True, 
        pad_token_id=pad_id_for_loss
    )
    model = FlowTransformerDecoder(gpt_config_instance) 
    print(f"Dummy model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    # --- Test prompt construction and context ID generation ---
    # Simulates how visualize_flow_rhythm would build a prompt
    
    # 1. BOS
    prompt_tokens = [tokenizer.bos_token_id]
    bos_seg, bos_intra = get_next_context_ids_for_token([], tokenizer.bos_token_id, tokenizer, gpt_config_instance.max_segment_types, gpt_config_instance.max_intra_line_positions)
    prompt_segments = [bos_seg]
    prompt_intra_pos = [bos_intra]

    # 2. Bar 0 Features
    bar0_feat_tok = tokenizer.bar_start_token_id
    b0f_s, b0f_i = get_next_context_ids_for_token(prompt_tokens, bar0_feat_tok, tokenizer, gpt_config_instance.max_segment_types, gpt_config_instance.max_intra_line_positions)
    prompt_tokens.append(bar0_feat_tok); prompt_segments.append(b0f_s); prompt_intra_pos.append(b0f_i)
    
    # (add a few dummy beat tokens for bar 0)
    for _ in range(3):
        dummy_beat_event_tok = tokenizer.token_to_id['[KICK_AT_0]'] # Example
        dbe_s, dbe_i = get_next_context_ids_for_token(prompt_tokens, dummy_beat_event_tok, tokenizer, gpt_config_instance.max_segment_types, gpt_config_instance.max_intra_line_positions)
        prompt_tokens.append(dummy_beat_event_tok); prompt_segments.append(dbe_s); prompt_intra_pos.append(dbe_i)

    # 3. SEP for Bar 0 Flow
    sep0_tok = tokenizer.sep_input_flow_token_id
    s0_s, s0_i = get_next_context_ids_for_token(prompt_tokens, sep0_tok, tokenizer, gpt_config_instance.max_segment_types, gpt_config_instance.max_intra_line_positions)
    prompt_tokens.append(sep0_tok); prompt_segments.append(s0_s); prompt_intra_pos.append(s0_i)

    # 4. LINE_START for Bar 0, Line 0 (Priming token)
    line0_tok = tokenizer.line_start_token_id
    l0_s, l0_i = get_next_context_ids_for_token(prompt_tokens, line0_tok, tokenizer, gpt_config_instance.max_segment_types, gpt_config_instance.max_intra_line_positions)
    prompt_tokens.append(line0_tok); prompt_segments.append(l0_s); prompt_intra_pos.append(l0_i)


    idx_prompt = torch.tensor([prompt_tokens], dtype=torch.long)
    seg_ids_prompt = torch.tensor([prompt_segments], dtype=torch.long)
    intra_pos_ids_prompt = torch.tensor([prompt_intra_pos], dtype=torch.long)

    print(f"\nConstructed Prompt for Generation (len {idx_prompt.size(1)}):")
    print("  Token            | Seg | IntraPos")
    print("  -----------------|-----|---------")
    for i in range(idx_prompt.size(1)):
        tok_str = tokenizer.id_to_token.get(idx_prompt[0,i].item(), "[UNK]")
        seg_val = seg_ids_prompt[0,i].item()
        pos_val = intra_pos_ids_prompt[0,i].item()
        print(f"  {tok_str:<16} | {seg_val:<3} | {pos_val:<7}")

    print("\nSimulating model.generate()...")
    # Generate a few tokens to see their context IDs
    generated_ids_full = model.generate(
        idx_prompt, 
        seg_ids_prompt,
        intra_pos_ids_prompt,
        max_new_tokens=15, 
        tokenizer=tokenizer,
        temperature=0.8,
        top_k=20 # Use a reasonable top_k for diverse but not too random output
    )
    print("\nFull sequence from model.generate() (Prompt + Generated):")
    
    generated_sequence_list = generated_ids_full[0].tolist()
    print_tokens_with_inferred_context(generated_sequence_list, tokenizer, gpt_config_instance)