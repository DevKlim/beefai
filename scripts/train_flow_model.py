import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import yaml 
import time
import sys
import random # For setting seeds
import numpy as np # For setting seeds
import torch.nn as nn # Added for type hint in save_checkpoint

# Adjust import paths if beefai is not directly in PYTHONPATH
sys.path.append(os.getcwd()) 

from beefai.flow_model.tokenizer import FlowTokenizer
from beefai.flow_model.transformer_model import FlowTransformerDecoder, FlowGPTConfig
from beefai.flow_model.dataset import FlowDataset

# --- Configuration Loading ---
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Default paths for config files - these define the "FULL" model training
MODEL_CONFIG_FULL_PATH = "lite_model_training/model_config_full.yaml" 
DATA_CONFIG_FULL_PATH = "lite_model_training/data_config_full.yaml"   

# --- Default Training Hyperparameters (can be overridden by YAML) ---
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 5e-5 # Common starting LR for transformers
DEFAULT_EPOCHS = 20 # Reduced from 100 for practical initial runs
DEFAULT_GRAD_ACCUMULATION_STEPS = 4
DEFAULT_EVAL_INTERVAL_STEPS = 200
DEFAULT_SAVE_INTERVAL_STEPS = 1000 # Save less frequently than eval
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Automatic Mixed Precision (AMP) - enable if using CUDA and PyTorch >= 1.6
USE_AMP = True if DEVICE == "cuda" and hasattr(torch.cuda.amp, 'GradScaler') else False 
# Compile model with torch.compile (PyTorch 2.0+) - enable if using CUDA
COMPILE_MODEL = hasattr(torch, 'compile') and DEVICE == "cuda"


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True # Can impact performance
        torch.backends.cudnn.benchmark = False    # Can impact performance

def train_full_model(): # Renamed function for clarity
    print(f"--- Training FULL Flow Model ---")
    
    # Load configurations
    if not os.path.exists(MODEL_CONFIG_FULL_PATH):
        print(f"ERROR: Model config not found: {MODEL_CONFIG_FULL_PATH}"); return
    model_params_yaml = load_yaml_config(MODEL_CONFIG_FULL_PATH)
    
    if not os.path.exists(DATA_CONFIG_FULL_PATH):
        print(f"ERROR: Data config not found: {DATA_CONFIG_FULL_PATH}"); return
    data_params_yaml = load_yaml_config(DATA_CONFIG_FULL_PATH)

    # Get training parameters, using defaults if not in YAML
    batch_size = model_params_yaml.get("batch_size", DEFAULT_BATCH_SIZE)
    learning_rate = model_params_yaml.get("learning_rate", DEFAULT_LEARNING_RATE)
    epochs = model_params_yaml.get("epochs", DEFAULT_EPOCHS)
    grad_accumulation_steps = model_params_yaml.get("grad_accumulation_steps", DEFAULT_GRAD_ACCUMULATION_STEPS)
    eval_interval_steps = model_params_yaml.get("eval_interval_steps", DEFAULT_EVAL_INTERVAL_STEPS)
    save_interval_steps = model_params_yaml.get("save_interval_steps", DEFAULT_SAVE_INTERVAL_STEPS)
    weight_decay = model_params_yaml.get("weight_decay", DEFAULT_WEIGHT_DECAY)
    seed = model_params_yaml.get("seed", DEFAULT_SEED)
    
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    print(f"Using device: {DEVICE}")
    if USE_AMP: print("Using Automatic Mixed Precision (AMP).")
    if COMPILE_MODEL: print("Attempting to compile model (PyTorch 2.0+).")

    # TensorBoard setup
    # Checkpoint dir from data_params_yaml, as it often contains paths
    checkpoint_dir = data_params_yaml.get("checkpoint_dir", "data/checkpoints/flow_model_full/")
    os.makedirs(checkpoint_dir, exist_ok=True) # Ensure base checkpoint_dir exists
    
    log_runs_dir = os.path.join(checkpoint_dir, 'runs_full') # Specific subdir for full model runs
    os.makedirs(log_runs_dir, exist_ok=True)
    run_name = f'full_model_exp_{time.strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir=os.path.join(log_runs_dir, run_name))
    print(f"TensorBoard logs will be saved to: {os.path.join(log_runs_dir, run_name)}")

    # 1. Initialize Tokenizer
    tokenizer_path = data_params_yaml.get("tokenizer_path")
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer config not found at '{tokenizer_path}' (from data_config). Cannot proceed.")
        writer.close(); return
    tokenizer = FlowTokenizer(config_path=tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.pad_token_id

    # 2. Initialize FULL Model from model_params_yaml
    gpt_config = FlowGPTConfig(
        vocab_size=vocab_size, 
        block_size=model_params_yaml["block_size"], # Required
        n_layer=model_params_yaml["n_layer"],       # Required
        n_head=model_params_yaml["n_head"],         # Required
        n_embd=model_params_yaml["n_embd"],         # Required
        max_segment_types=model_params_yaml["max_segment_types"], # Required
        max_intra_line_positions=model_params_yaml["max_intra_line_positions"], # Required
        dropout=model_params_yaml.get("dropout", 0.1), # Optional in YAML
        bias=model_params_yaml.get("bias", True),      # Optional in YAML
        pad_token_id=pad_token_id
    )
    
    model = FlowTransformerDecoder(gpt_config)
    if COMPILE_MODEL:
        print("Compiling the model...")
        try:
            model = torch.compile(model) 
            print("Model compiled successfully.")
        except Exception as e: # Catch generic exception from compile
            print(f"Model compilation failed: {e}. Proceeding without compilation.")
            
    model.to(DEVICE)
    print(f"FULL Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 3. Load FULL Data (paths from data_params_yaml)
    # train_data_path should be tokenized_data_output_dir + train_data_filename from data_config_full
    tokenized_data_output_dir = data_params_yaml.get("tokenized_data_output_dir")
    train_data_filename = data_params_yaml.get("train_data_filename", "train_full.pt")
    val_data_filename = data_params_yaml.get("val_data_filename", "val_full.pt")

    if not tokenized_data_output_dir:
        print(f"ERROR: 'tokenized_data_output_dir' not specified in {DATA_CONFIG_FULL_PATH}"); writer.close(); return
        
    train_data_path = os.path.join(tokenized_data_output_dir, train_data_filename)
    val_data_path = os.path.join(tokenized_data_output_dir, val_data_filename)


    if not os.path.exists(train_data_path):
        print(f"ERROR: FULL training data not found at {train_data_path}.")
        print(f"Please ensure 'scripts/05b_tokenize_data_full.py' has run successfully and paths in {DATA_CONFIG_FULL_PATH} are correct.")
        writer.close(); return

    train_dataset = FlowDataset(
        data_file_path=train_data_path, 
        tokenizer_pad_id=pad_token_id,
        block_size=gpt_config.block_size # Use block_size from model config
    )
    # Consider num_workers based on CPU cores for DataLoader
    num_dataloader_workers = min(os.cpu_count() // 2, 4) if os.cpu_count() and os.cpu_count() > 1 else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_dataloader_workers, pin_memory=(DEVICE=="cuda"))
    
    val_loader = None
    if os.path.exists(val_data_path):
        val_dataset = FlowDataset(
            data_file_path=val_data_path, 
            tokenizer_pad_id=pad_token_id, 
            block_size=gpt_config.block_size
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_dataloader_workers, pin_memory=(DEVICE=="cuda"))
    else:
        print(f"Warning: FULL validation data path '{val_data_path}' not found. Proceeding without validation during training.")

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95)) # Common betas
    
    effective_steps_per_epoch = max(1, len(train_loader) // grad_accumulation_steps)
    total_training_steps = effective_steps_per_epoch * epochs
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate,
                           total_steps=total_training_steps if total_training_steps > 0 else 100, # ensure total_steps > 0
                           pct_start=0.1, # Percentage of steps for warmup
                           anneal_strategy='cos') # Cosine annealing
    
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    # 5. Training Loop
    model.train()
    global_step = 0
    total_training_time_seconds = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_train_loss = 0.0
        
        # tqdm progress bar for the epoch
        progress_bar = tqdm(train_loader, desc=f"FULL Epoch {epoch+1}/{epochs} Training", leave=False)
        
        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            target_ids = batch["target_ids"].to(DEVICE, non_blocking=True)
            segment_ids = batch["segment_ids"].to(DEVICE, non_blocking=True)
            intra_line_pos_ids = batch["intra_line_pos_ids"].to(DEVICE, non_blocking=True)

            loss_val = 0.0 # Initialize loss_val
            # Forward pass
            if USE_AMP and scaler:
                with torch.cuda.amp.autocast():
                    logits, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
                if loss is not None: 
                    loss_val = loss.item()
                    loss_to_backward = loss / grad_accumulation_steps
                    scaler.scale(loss_to_backward).backward()
            else:
                logits, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
                if loss is not None:
                    loss_val = loss.item()
                    loss_to_backward = loss / grad_accumulation_steps
                    loss_to_backward.backward()
            
            if loss is not None: # Check if loss was computed
                epoch_train_loss += loss_val

            # Gradient accumulation
            if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(train_loader):
                if USE_AMP and scaler:
                    scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                
                if USE_AMP and scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True) 
                if total_training_steps > 0 : scheduler.step() 
                global_step += 1

                if loss is not None: # Log only if loss was computed
                    writer.add_scalar('FULL/Train/Loss_step', loss_val, global_step)
                writer.add_scalar('FULL/Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
                progress_bar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")


                # Evaluation and Checkpointing (based on global_step)
                if global_step > 0 and global_step % eval_interval_steps == 0 and val_loader:
                    evaluate(model, val_loader, DEVICE, pad_token_id, USE_AMP, "FULL", writer, global_step) 
                    model.train() 

                if global_step > 0 and global_step % save_interval_steps == 0:
                    save_checkpoint(model, optimizer, epoch, global_step, scheduler, checkpoint_dir, filename=f"full_ckpt_step_{global_step}.pt", model_type="FULL")
        
        # End of Epoch summary
        epoch_duration = time.time() - epoch_start_time
        total_training_time_seconds += epoch_duration
        avg_epoch_train_loss = epoch_train_loss / max(1, len(train_loader))
        print(f"FULL Epoch {epoch+1} finished. Avg Train Loss: {avg_epoch_train_loss:.4f}. Time: {epoch_duration:.2f}s. LR: {optimizer.param_groups[0]['lr']:.2e}")
        writer.add_scalar('FULL/Train/Loss_epoch', avg_epoch_train_loss, epoch + 1)
        
        if val_loader:
            evaluate(model, val_loader, DEVICE, pad_token_id, USE_AMP, "FULL", writer, global_step) 
            model.train()
        
        save_checkpoint(model, optimizer, epoch + 1, global_step, scheduler, checkpoint_dir, filename=f"full_ckpt_epoch_{epoch+1}.pt", model_type="FULL")

    print(f"\nFULL training complete. Total time: {total_training_time_seconds/3600:.2f} hours.")
    save_checkpoint(model, optimizer, epochs, global_step, scheduler, checkpoint_dir, filename="full_final_model.pt", model_type="FULL")
    writer.close()


def evaluate(model, val_loader, device, pad_token_id, use_amp, model_type_str, writer, current_global_step): 
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    print(f"\nEvaluating {model_type_str} model on validation set (step {current_global_step})...")
    
    progress_bar_val = tqdm(val_loader, desc=f"{model_type_str} Validation", leave=False)
    with torch.no_grad():
        for batch in progress_bar_val:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            segment_ids = batch["segment_ids"].to(device, non_blocking=True)
            intra_line_pos_ids = batch["intra_line_pos_ids"].to(device, non_blocking=True)

            loss_val_item = 0.0
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
            else:
                 _, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
            
            if loss is not None:
                loss_val_item = loss.item()
                total_val_loss += loss_val_item
            num_batches += 1
            progress_bar_val.set_postfix(val_loss_batch=f"{loss_val_item:.4f}")
    
    avg_val_loss = total_val_loss / max(1, num_batches) 
    perplexity_val = torch.exp(torch.tensor(avg_val_loss)) if avg_val_loss > 0 and num_batches > 0 else float('inf')
    print(f"{model_type_str} Validation Summary: Avg Loss: {avg_val_loss:.4f}, Perplexity: {perplexity_val:.2f}")

    if writer: # Check if writer is available
        writer.add_scalar(f'{model_type_str}/Val/Loss', avg_val_loss, current_global_step)
        writer.add_scalar(f'{model_type_str}/Val/Perplexity', perplexity_val, current_global_step)
    return avg_val_loss

def save_checkpoint(model, optimizer, epoch, step, scheduler, checkpoint_dir, filename=None, model_type="Model"): 
    if filename is None:
        filename = f"{model_type.lower()}_ckpt_epoch_{epoch}_step_{step}.pt"
    
    os.makedirs(checkpoint_dir, exist_ok=True) 
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module) else model

    checkpoint_data = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_to_save.config if hasattr(model_to_save, 'config') and model_to_save.config is not None else None
    }
    if scheduler:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    try:
        torch.save(checkpoint_data, checkpoint_path)
        print(f"{model_type} Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        print(f"ERROR saving checkpoint to {checkpoint_path}: {e}")

if __name__ == "__main__":
    train_full_model()