import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR # Example of a more advanced scheduler
from tqdm import tqdm
import os
import yaml # For loading YAML configs
import time

# Adjust import paths if beefai is not directly in PYTHONPATH
# This assumes the script is run from the project root (./)
import sys
sys.path.append(os.getcwd()) 

from beefai.flow_model.tokenizer import FlowTokenizer
from beefai.flow_model.transformer_model import FlowTransformerDecoder, FlowGPTConfig
from beefai.flow_model.dataset import FlowDataset

# --- Configuration Loading ---
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

MODEL_CONFIG_LITE_PATH = "lite_model_training/model_config_lite.yaml"
DATA_CONFIG_LITE_PATH = "lite_model_training/data_config_lite.yaml"

model_params = load_yaml_config(MODEL_CONFIG_LITE_PATH)
data_params = load_yaml_config(DATA_CONFIG_LITE_PATH)

# Training Hyperparameters
BATCH_SIZE = 8 # Smaller batch size might be needed for very small models or if memory is tight
LEARNING_RATE = 5e-4 # Might use a slightly higher LR for smaller models/datasets
EPOCHS = 20 # More epochs on smaller data
GRAD_ACCUMULATION_STEPS = 2 
EVAL_INTERVAL = 50 
SAVE_INTERVAL = 200 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True if DEVICE == "cuda" else False # Enable Mixed Precision for CUDA
COMPILE_MODEL = hasattr(torch, 'compile') and DEVICE == "cuda" # PyTorch 2.0+ feature

def train_lite():
    print(f"--- Training Lite Flow Model ---")
    print(f"Using device: {DEVICE}")
    if USE_AMP: print("Using Automatic Mixed Precision (AMP).")
    if COMPILE_MODEL: print("Attempting to compile model (PyTorch 2.0+).")

    # 1. Initialize Tokenizer
    tokenizer_path = data_params["tokenizer_path"]
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer config not found at {tokenizer_path}.")
        return
    tokenizer = FlowTokenizer(config_path=tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.pad_token_id

    # 2. Initialize Lite Model
    model_config = FlowGPTConfig(
        vocab_size=vocab_size, 
        block_size=model_params["block_size"],
        n_layer=model_params["n_layer"], 
        n_head=model_params["n_head"], 
        n_embd=model_params["n_embd"],
        max_segment_types=model_params["max_segment_types"],
        max_intra_line_positions=model_params["max_intra_line_positions"],
        dropout=model_params["dropout"],
        bias=model_params.get("bias", True) # Get bias or default to True
    )
    model_config.pad_token_id = pad_token_id
    
    model = FlowTransformerDecoder(model_config)
    if COMPILE_MODEL:
        print("Compiling the model...")
        try:
            model = torch.compile(model) # PyTorch 2.0 feature
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}. Proceeding without compilation.")
            
    model.to(DEVICE)
    print(f"Lite Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 3. Load Lite Data
    train_data_path = data_params["train_data_path"]
    val_data_path = data_params.get("val_data_path") # Optional
    checkpoint_dir = data_params["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = FlowDataset(train_data_path, tokenizer, model_params["block_size"])
    # For lite training, a smaller effective batch size is fine.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE=="cuda" else False)
    
    val_loader = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset = FlowDataset(val_data_path, tokenizer, model_params["block_size"])
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # OneCycleLR scheduler for faster convergence potentially
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                           steps_per_epoch=len(train_loader) // GRAD_ACCUMULATION_STEPS, 
                           epochs=EPOCHS)
    
    scaler = None
    if USE_AMP:
        scaler = torch.cuda.amp.GradScaler()

    # 5. Training Loop
    model.train()
    step = 0
    total_time = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Lite Epoch {epoch+1}/{EPOCHS} ---")
        epoch_start_time = time.time()
        epoch_loss = 0
        optimizer.zero_grad() 

        for i, batch in enumerate(tqdm(train_loader, desc=f"Lite Epoch {epoch+1} Training")):
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            segment_ids = batch["segment_ids"].to(DEVICE)
            intra_line_pos_ids = batch["intra_line_pos_ids"].to(DEVICE)

            if USE_AMP:
                with torch.cuda.amp.autocast():
                    logits, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
                if loss is not None:
                    loss = loss / GRAD_ACCUMULATION_STEPS
                    scaler.scale(loss).backward()
            else:
                logits, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
                if loss is not None:
                    loss = loss / GRAD_ACCUMULATION_STEPS
                    loss.backward()
            
            if loss is not None:
                epoch_loss += loss.item() * GRAD_ACCUMULATION_STEPS

            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step() # Step scheduler each optimizer step
                step += 1

                if step > 0 and step % EVAL_INTERVAL == 0 and val_loader:
                    evaluate(model, val_loader, DEVICE, model_config.pad_token_id, USE_AMP) # Pass pad_token_id from config
                    model.train() 

                if step > 0 and step % SAVE_INTERVAL == 0:
                    save_checkpoint(model, optimizer, epoch, step, scheduler, checkpoint_dir, filename=f"lite_ckpt_step_{step}.pt")
        
        epoch_duration = time.time() - epoch_start_time
        total_time += epoch_duration
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Lite Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}. Time: {epoch_duration:.2f}s. LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_loader:
            evaluate(model, val_loader, DEVICE, model_config.pad_token_id, USE_AMP)
            model.train()
        
        save_checkpoint(model, optimizer, epoch, step, scheduler, checkpoint_dir, filename=f"lite_ckpt_epoch_{epoch+1}.pt")

    print(f"Lite training complete. Total time: {total_time/3600:.2f} hours.")
    save_checkpoint(model, optimizer, EPOCHS, step, scheduler, checkpoint_dir, filename="lite_final_model.pt")

def evaluate(model, val_loader, device, pad_token_id, use_amp): # Added use_amp
    model.eval()
    total_val_loss = 0
    print("\nEvaluating lite model on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Lite Validation"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            intra_line_pos_ids = batch["intra_line_pos_ids"].to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
            else:
                 _, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
            
            if loss is not None:
                total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_val_loss)) if avg_val_loss > 0 else float('inf')
    print(f"Lite Validation Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")
    return avg_val_loss

def save_checkpoint(model, optimizer, epoch, step, scheduler, checkpoint_dir, filename=None): # Added scheduler
    if filename is None:
        filename = f"lite_ckpt_step_{step}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # If model is compiled, get the original model for saving state_dict
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model

    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state
        'config': model_to_save.config 
    }, checkpoint_path)
    print(f"Lite Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    # Ensure you have prepared the lite data and config files before running.
    # 1. Create model_config_lite.yaml and data_config_lite.yaml
    # 2. Run a script like `scripts/05a_tokenize_data_lite.py` to create the .pt files.
    train_lite()