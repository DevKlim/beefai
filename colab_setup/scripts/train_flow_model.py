# scripts/train_flow_model.py
import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR # Example scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import random
import numpy as np
from tqdm import tqdm

# --- Robust Path Setup ---
# Add the project root to sys.path to allow importing 'beefai'
SCRIPT_DIR_TRAIN = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_TRAIN = os.path.abspath(os.path.join(SCRIPT_DIR_TRAIN, '..')) # Assumes 'scripts' is one level below project root

if PROJECT_ROOT_TRAIN not in sys.path:
    sys.path.append(PROJECT_ROOT_TRAIN)
    print(f"DEBUG [train_flow_model.py]: Added '{PROJECT_ROOT_TRAIN}' to sys.path")

print(f"DEBUG [train_flow_model.py]: Current sys.path: {sys.path}")
print(f"DEBUG [train_flow_model.py]: SCRIPT_DIR_TRAIN={SCRIPT_DIR_TRAIN}")
print(f"DEBUG [train_flow_model.py]: PROJECT_ROOT_TRAIN={PROJECT_ROOT_TRAIN}")

# --- Configuration Paths for model and data configs ---
# These paths are constructed relative to PROJECT_ROOT_TRAIN
# Assumes 'lite_model_training' directory is at the PROJECT_ROOT_TRAIN level
MODEL_CONFIG_REL_PATH = "lite_model_training/model_config_full.yaml"
DATA_CONFIG_REL_PATH = "lite_model_training/data_config_full.yaml"

MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT_TRAIN, MODEL_CONFIG_REL_PATH)
DATA_CONFIG_PATH = os.path.join(PROJECT_ROOT_TRAIN, DATA_CONFIG_REL_PATH)

print(f"DEBUG [train_flow_model.py]: Trying to load MODEL_CONFIG_PATH='{MODEL_CONFIG_PATH}'")
print(f"DEBUG [train_flow_model.py]: Trying to load DATA_CONFIG_PATH='{DATA_CONFIG_PATH}'")

# --- Project Module Imports ---
try:
    from beefai.flow_model.tokenizer import FlowTokenizer
    from beefai.flow_model.transformer_model import FlowTransformerDecoder, FlowGPTConfig
    from beefai.flow_model.dataset import FlowDataset
    print("DEBUG [train_flow_model.py]: Successfully imported beefai modules.")
except ImportError as e:
    print(f"CRITICAL ERROR [train_flow_model.py]: Failed to import beefai modules. {e}")
    print(f"Ensure '{PROJECT_ROOT_TRAIN}' contains the 'beefai' package and is correctly added to sys.path.")
    sys.exit(1)
except Exception as e_other_import:
    print(f"CRITICAL ERROR [train_flow_model.py]: An unexpected error occurred during beefai module imports: {e_other_import}")
    sys.exit(1)


# --- Helper Functions ---
def load_yaml_config(path: str):
    if not os.path.exists(path):
        print(f"ERROR [load_yaml_config]: Config file not found at {path}")
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Main Training Logic ---
def train_full_model():
    print("--- Training FULL Flow Model ---")

    try:
        model_config_dict = load_yaml_config(MODEL_CONFIG_PATH)
        data_config_dict = load_yaml_config(DATA_CONFIG_PATH)
    except FileNotFoundError:
        print(f"CRITICAL ERROR [train_flow_model.py]: Main model or data config file not found (see error above from load_yaml_config). Exiting.")
        sys.exit(1)
    except Exception as e_yaml:
        print(f"CRITICAL ERROR [train_flow_model.py]: Could not parse YAML configuration files: {e_yaml}")
        sys.exit(1)

    print("DEBUG [train_flow_model.py]: YAML configurations (model_config & data_config) loaded successfully.")

    # --- Seed ---
    seed = model_config_dict.get('seed', 42)
    set_seed(seed)
    print(f"DEBUG [train_flow_model.py]: Random seed set to {seed}")

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEBUG [train_flow_model.py]: Using device: {device.upper()}")
    if device == "cpu":
        print("WARNING: CUDA not available, training on CPU will be very slow.")

    # --- Tokenizer ---
    tokenizer_config_path_from_yaml = data_config_dict.get("tokenizer_path")
    if not tokenizer_config_path_from_yaml:
        print(f"CRITICAL ERROR [train_flow_model.py]: 'tokenizer_path' not found in data config '{DATA_CONFIG_PATH}'")
        sys.exit(1)
    
    # Resolve tokenizer_path relative to PROJECT_ROOT_TRAIN
    tokenizer_config_abs_path = os.path.join(PROJECT_ROOT_TRAIN, tokenizer_config_path_from_yaml)
    
    print(f"DEBUG [train_flow_model.py]: Attempting to load tokenizer config from resolved path: '{tokenizer_config_abs_path}' (Original from data_config: '{tokenizer_config_path_from_yaml}')")
    if not os.path.exists(tokenizer_config_abs_path):
        print(f"CRITICAL ERROR [train_flow_model.py]: Tokenizer config file not found at '{tokenizer_config_abs_path}'")
        sys.exit(1)
        
    try:
        tokenizer = FlowTokenizer(config_path=tokenizer_config_abs_path)
        print(f"DEBUG [train_flow_model.py]: FlowTokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e_tok:
        print(f"CRITICAL ERROR [train_flow_model.py]: Failed to initialize FlowTokenizer with config '{tokenizer_config_abs_path}': {e_tok}")
        sys.exit(1)

    # --- Model Configuration ---
    gpt_config = FlowGPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        block_size=model_config_dict["block_size"],
        n_layer=model_config_dict["n_layer"],
        n_head=model_config_dict["n_head"],
        n_embd=model_config_dict["n_embd"],
        max_segment_types=model_config_dict["max_segment_types"],
        max_intra_line_positions=model_config_dict["max_intra_line_positions"],
        dropout=model_config_dict.get("dropout", 0.1),
        bias=model_config_dict.get("bias", True),
        pad_token_id=tokenizer.pad_token_id
    )
    print("DEBUG [train_flow_model.py]: FlowGPTConfig initialized.")

    # --- Model ---
    model = FlowTransformerDecoder(gpt_config)
    model.to(device)
    print(f"DEBUG [train_flow_model.py]: FlowTransformerDecoder initialized with {sum(p.numel() for p in model.parameters()):,} parameters and moved to {device}.")
    
    # --- DataLoaders ---
    train_data_path_from_yaml = data_config_dict.get("train_data_path")
    val_data_path_from_yaml = data_config_dict.get("val_data_path")

    if not train_data_path_from_yaml or not val_data_path_from_yaml:
        print(f"CRITICAL ERROR [train_flow_model.py]: 'train_data_path' or 'val_data_path' not found in data config '{DATA_CONFIG_PATH}'")
        sys.exit(1)

    train_data_abs_path = os.path.join(PROJECT_ROOT_TRAIN, train_data_path_from_yaml)
    val_data_abs_path = os.path.join(PROJECT_ROOT_TRAIN, val_data_path_from_yaml)
    
    print(f"DEBUG [train_flow_model.py]: Attempting to load training data from: '{train_data_abs_path}' (Original from data_config: '{train_data_path_from_yaml}')")
    if not os.path.exists(train_data_abs_path):
        print(f"CRITICAL ERROR [train_flow_model.py]: Training data file not found: '{train_data_abs_path}'")
        sys.exit(1)
    print(f"DEBUG [train_flow_model.py]: Attempting to load validation data from: '{val_data_abs_path}' (Original from data_config: '{val_data_path_from_yaml}')")
    if not os.path.exists(val_data_abs_path):
        print(f"CRITICAL ERROR [train_flow_model.py]: Validation data file not found: '{val_data_abs_path}'")
        sys.exit(1)

    try:
        train_dataset = FlowDataset(
            data_file_path=train_data_abs_path,
            tokenizer_pad_id=tokenizer.pad_token_id,
            block_size=gpt_config.block_size
        )
        val_dataset = FlowDataset(
            data_file_path=val_data_abs_path,
            tokenizer_pad_id=tokenizer.pad_token_id,
            block_size=gpt_config.block_size
        )

        if not train_dataset or len(train_dataset) == 0:
            print(f"CRITICAL ERROR [train_flow_model.py]: Training dataset is empty after loading from '{train_data_abs_path}'. Check the data file and tokenization process.")
            sys.exit(1)
        if not val_dataset or len(val_dataset) == 0 : # Also check val_dataset if it's expected
             print(f"WARNING [train_flow_model.py]: Validation dataset is empty after loading from '{val_data_abs_path}'. Evaluation steps might be skipped or error.")


        train_loader = DataLoader(
            train_dataset,
            batch_size=model_config_dict.get("batch_size", 16),
            shuffle=True,
            num_workers=model_config_dict.get("dataloader_num_workers", 2), 
            pin_memory=True if device == "cuda" else False,
            drop_last=True # Good for stable batch sizes, especially with accumulation
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=model_config_dict.get("batch_size", 16) * 2, 
            shuffle=False,
            num_workers=model_config_dict.get("dataloader_num_workers", 2),
            pin_memory=True if device == "cuda" else False
        )
        print("DEBUG [train_flow_model.py]: DataLoaders created.")
    except Exception as e_dataset:
        print(f"CRITICAL ERROR [train_flow_model.py]: Failed to create FlowDataset instances or DataLoaders: {e_dataset}")
        sys.exit(1)


    # --- Optimizer and Scheduler ---
    optimizer = AdamW(model.parameters(), lr=model_config_dict.get("learning_rate", 3e-4), weight_decay=model_config_dict.get("weight_decay", 0.01))
    
    epochs = model_config_dict.get("epochs", 10)
    # Ensure train_loader is not empty before calculating total_steps
    if len(train_loader) == 0:
        print("ERROR [train_flow_model.py]: train_loader is empty, cannot calculate total_steps for scheduler. Exiting.")
        sys.exit(1)
    total_steps = len(train_loader) * epochs
    scheduler = OneCycleLR(optimizer, max_lr=model_config_dict.get("learning_rate", 3e-4), total_steps=total_steps)
    print("DEBUG [train_flow_model.py]: Optimizer and Scheduler created.")

    # --- Checkpoint Directory and TensorBoard ---
    checkpoint_dir_from_yaml = data_config_dict.get("checkpoint_dir", "data/checkpoints/flow_model_default/")
    checkpoint_base_dir = os.path.join(PROJECT_ROOT_TRAIN, checkpoint_dir_from_yaml)
    
    os.makedirs(checkpoint_base_dir, exist_ok=True)
    run_name = f"full_run_{time.strftime('%Y%m%d-%H%M%S')}" # Changed from "full_model_run" to avoid potential clash if script is run twice quickly
    run_checkpoint_dir = os.path.join(checkpoint_base_dir, run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=run_checkpoint_dir)
    print(f"DEBUG [train_flow_model.py]: Checkpoints and TensorBoard logs will be saved in: {run_checkpoint_dir}")

    # --- AMP (Automatic Mixed Precision) ---
    scaler = None
    use_amp = model_config_dict.get("use_amp", True) # Add a config option for AMP
    if device == "cuda" and use_amp and torch.cuda.is_available() and hasattr(torch.cuda.amp, "GradScaler"):
        scaler = torch.cuda.amp.GradScaler()
        print("DEBUG [train_flow_model.py]: AMP GradScaler enabled for CUDA training.")
    elif use_amp and device != "cuda":
        print("INFO [train_flow_model.py]: AMP (use_amp=True) requested but device is not CUDA. AMP will not be used.")
    elif not use_amp:
        print("INFO [train_flow_model.py]: AMP use_amp is False in config. AMP will not be used.")


    # --- Training Loop ---
    grad_accumulation_steps = model_config_dict.get("grad_accumulation_steps", 1)
    eval_interval_steps = model_config_dict.get("eval_interval_steps", 200)
    save_interval_steps = model_config_dict.get("save_interval_steps", 500)
    
    global_step = 0
    best_val_loss = float('inf')

    print(f"--- Starting Training for {epochs} epochs ---")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", unit="batch")

        for batch_idx, batch in enumerate(train_iterator):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)
            segment_ids = batch['segment_ids'].to(device, non_blocking=True)
            intra_line_pos_ids = batch['intra_line_pos_ids'].to(device, non_blocking=True)

            if scaler: 
                with torch.cuda.amp.autocast():
                    logits, loss = model(idx=input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
                loss = loss / grad_accumulation_steps 
                scaler.scale(loss).backward()
            else: 
                logits, loss = model(idx=input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
                loss = loss / grad_accumulation_steps
                loss.backward()
            
            epoch_train_loss += loss.item() * grad_accumulation_steps 

            if (batch_idx + 1) % grad_accumulation_steps == 0:
                if scaler:
                    # Unscale gradients before clipping (optional but good practice)
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional grad clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional grad clipping
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True) # More memory efficient
                scheduler.step() 
                global_step += 1

                writer.add_scalar('Loss/train_step', loss.item() * grad_accumulation_steps, global_step)
                writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], global_step)
                
                train_iterator.set_postfix(loss=f"{loss.item() * grad_accumulation_steps:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if global_step > 0 and global_step % eval_interval_steps == 0 and val_loader and len(val_loader) > 0:
                    model.eval()
                    val_loss_accum = 0.0
                    val_steps = 0
                    with torch.no_grad():
                        val_iterator_eval = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation @ GStep {global_step}]", unit="batch", leave=False)
                        for val_batch in val_iterator_eval:
                            v_input_ids = val_batch['input_ids'].to(device, non_blocking=True)
                            v_target_ids = val_batch['target_ids'].to(device, non_blocking=True)
                            v_segment_ids = val_batch['segment_ids'].to(device, non_blocking=True)
                            v_intra_line_pos_ids = val_batch['intra_line_pos_ids'].to(device, non_blocking=True)
                            
                            if scaler: 
                                with torch.cuda.amp.autocast():
                                    _, v_loss = model(idx=v_input_ids, segment_ids=v_segment_ids, intra_line_pos_ids=v_intra_line_pos_ids, targets=v_target_ids)
                            else:
                                _, v_loss = model(idx=v_input_ids, segment_ids=v_segment_ids, intra_line_pos_ids=v_intra_line_pos_ids, targets=v_target_ids)
                            
                            val_loss_accum += v_loss.item()
                            val_steps += 1
                            val_iterator_eval.set_postfix(val_loss_batch=f"{v_loss.item():.4f}")

                    avg_val_loss = val_loss_accum / val_steps if val_steps > 0 else float('inf')
                    writer.add_scalar('Loss/validation', avg_val_loss, global_step)
                    writer.add_scalar('Perplexity/validation', torch.exp(torch.tensor(avg_val_loss)).item() if avg_val_loss != float('inf') else float('inf'), global_step)
                    print(f"\nEpoch {epoch+1}, Global Step {global_step}: Avg Validation Loss: {avg_val_loss:.4f}, Val Perplexity: {torch.exp(torch.tensor(avg_val_loss)).item():.2f}")
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        print(f"  New best validation loss: {best_val_loss:.4f}. Saving model...")
                        checkpoint_path = os.path.join(run_checkpoint_dir, f"best_model_gstep_{global_step}_valloss_{best_val_loss:.4f}.pt")
                        torch.save({
                            'global_step': global_step, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                            'loss': best_val_loss, 'config': gpt_config.__dict__ 
                        }, checkpoint_path)
                    model.train() 

                if global_step > 0 and global_step % save_interval_steps == 0:
                    periodic_checkpoint_path = os.path.join(run_checkpoint_dir, f"ckpt_gstep_{global_step}.pt")
                    print(f"  Saving periodic checkpoint at global step {global_step} to {periodic_checkpoint_path}...")
                    torch.save({
                        'global_step': global_step, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item() * grad_accumulation_steps, 
                        'val_loss_at_save': avg_val_loss if 'avg_val_loss' in locals() and val_steps > 0 else "N/A",
                        'config': gpt_config.__dict__
                    }, periodic_checkpoint_path)

        avg_epoch_train_loss = epoch_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        writer.add_scalar('Loss/train_epoch', avg_epoch_train_loss, epoch + 1)
        print(f"Epoch {epoch+1}/{epochs} completed. Average Training Loss: {avg_epoch_train_loss:.4f}")

    # --- Final Save ---
    final_model_path = os.path.join(run_checkpoint_dir, "final_model.pt")
    print(f"Training finished. Saving final model to {final_model_path}...")
    torch.save({
        'global_step': global_step, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_epoch_train_loss, 
        'best_val_loss': best_val_loss,
        'config': gpt_config.__dict__
    }, final_model_path)
    
    writer.close()
    print("--- Training Complete ---")

if __name__ == "__main__":
    print("DEBUG [train_flow_model.py]: Script executed as __main__.")
    try:
        train_full_model()
    except Exception as e:
        print(f"CRITICAL UNHANDLED EXCEPTION in train_full_model(): {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("DEBUG [train_flow_model.py]: Script execution finished or terminated.")