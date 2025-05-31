import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter # Added for TensorBoard
from tqdm import tqdm
import os
import yaml
import time
import sys

# Adjust import paths if beefai is not directly in PYTHONPATH
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

# --- Training Hyperparameters ---
# These will be primarily sourced from the YAML config files.
# Defaults are provided here if not found in YAML.
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_EPOCHS = 20
DEFAULT_GRAD_ACCUMULATION_STEPS = 2
DEFAULT_EVAL_INTERVAL_STEPS = 50
DEFAULT_SAVE_INTERVAL_STEPS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True if DEVICE == "cuda" else False 
COMPILE_MODEL = hasattr(torch, 'compile') and DEVICE == "cuda" 

def train_lite():
    print(f"--- Training Lite Flow Model ---")
    model_params = load_yaml_config(MODEL_CONFIG_LITE_PATH)
    data_params = load_yaml_config(DATA_CONFIG_LITE_PATH)

    # Override defaults with config values if present
    batch_size = model_params.get("batch_size", DEFAULT_BATCH_SIZE)
    learning_rate = model_params.get("learning_rate", DEFAULT_LEARNING_RATE)
    epochs = model_params.get("epochs", DEFAULT_EPOCHS)
    grad_accumulation_steps = model_params.get("grad_accumulation_steps", DEFAULT_GRAD_ACCUMULATION_STEPS)
    eval_interval_steps = model_params.get("eval_interval_steps", DEFAULT_EVAL_INTERVAL_STEPS)
    save_interval_steps = model_params.get("save_interval_steps", DEFAULT_SAVE_INTERVAL_STEPS)

    print(f"Using device: {DEVICE}")
    if USE_AMP: print("Using Automatic Mixed Precision (AMP).")
    if COMPILE_MODEL: print("Attempting to compile model (PyTorch 2.0+).")

    # Initialize TensorBoard SummaryWriter
    log_dir_base = data_params.get("checkpoint_dir", "data/checkpoints/flow_model_lite/")
    run_name = f'lite_experiment_{time.strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir=os.path.join(log_dir_base, 'runs', run_name))
    print(f"TensorBoard logs will be saved to: {os.path.join(log_dir_base, 'runs', run_name)}")

    # 1. Initialize Tokenizer
    tokenizer_path = data_params["tokenizer_path"]
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer config not found at {tokenizer_path}.")
        print(f"Please ensure data preparation scripts have run successfully to create it.")
        writer.close()
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
        bias=model_params.get("bias", True) 
    )
    model_config.pad_token_id = pad_token_id
    
    model = FlowTransformerDecoder(model_config)
    if COMPILE_MODEL:
        print("Compiling the model...")
        try:
            model = torch.compile(model) 
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}. Proceeding without compilation.")
            
    model.to(DEVICE)
    print(f"Lite Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 3. Load Lite Data
    train_data_path = data_params["train_data_path"]
    val_data_path = data_params.get("val_data_path") 
    checkpoint_dir = data_params["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(train_data_path):
        print(f"Lite training data not found at {train_data_path}.")
        print(f"Please ensure data preparation scripts have run successfully.")
        writer.close()
        return

    train_dataset = FlowDataset(
        data_file_path=train_data_path, 
        tokenizer_pad_id=pad_token_id,
        block_size=model_params["block_size"]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if DEVICE=="cuda" else False)
    
    val_loader = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset = FlowDataset(
            data_file_path=val_data_path, 
            tokenizer_pad_id=pad_token_id, 
            block_size=model_params["block_size"]
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    elif val_data_path:
        print(f"Warning: Lite validation data path '{val_data_path}' specified but file not found. Proceeding without validation.")

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Calculate total_steps for OneCycleLR
    # If train_loader is empty (e.g., no data), default to 1 to avoid division by zero
    effective_steps_per_epoch = max(1, len(train_loader) // grad_accumulation_steps)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, 
                           steps_per_epoch=effective_steps_per_epoch,
                           epochs=epochs)
    
    scaler = None
    if USE_AMP:
        scaler = torch.cuda.amp.GradScaler()

    # 5. Training Loop
    model.train()
    step = 0
    total_time_seconds = 0

    for epoch in range(epochs):
        print(f"\n--- Lite Epoch {epoch+1}/{epochs} ---")
        epoch_start_time = time.time()
        epoch_loss = 0
        optimizer.zero_grad() 

        if not train_loader: 
            print("Error: train_loader is not initialized. Cannot start epoch.")
            writer.close()
            return

        for i, batch in enumerate(tqdm(train_loader, desc=f"Lite Epoch {epoch+1} Training")):
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            segment_ids = batch["segment_ids"].to(DEVICE)
            intra_line_pos_ids = batch["intra_line_pos_ids"].to(DEVICE)

            current_loss_val = 0.0
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    logits, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
                if loss is not None:
                    current_loss_val = loss.item()
                    loss = loss / grad_accumulation_steps
                    scaler.scale(loss).backward()
            else:
                logits, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
                if loss is not None:
                    current_loss_val = loss.item()
                    loss = loss / grad_accumulation_steps
                    loss.backward()
            
            if loss is not None: # Check if loss was computed
                epoch_loss += current_loss_val # Accumulate un-normalized loss for epoch avg

            if (i + 1) % grad_accumulation_steps == 0 or (i + 1) == len(train_loader):
                if USE_AMP:
                    scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                if scheduler.total_steps > 0 : # Make sure scheduler can step
                     scheduler.step() 
                step += 1

                # Log to TensorBoard
                if loss is not None: # Only log if loss was computed
                    writer.add_scalar('Lite/Train/Loss_step', current_loss_val, step)
                writer.add_scalar('Lite/Train/LearningRate', optimizer.param_groups[0]['lr'], step)

                if step > 0 and step % eval_interval_steps == 0 and val_loader:
                    evaluate(model, val_loader, DEVICE, pad_token_id, USE_AMP, "Lite", writer, step) 
                    model.train() 

                if step > 0 and step % save_interval_steps == 0:
                    save_checkpoint(model, optimizer, epoch, step, scheduler, checkpoint_dir, filename=f"lite_ckpt_step_{step}.pt", model_type="Lite")
        
        epoch_duration = time.time() - epoch_start_time
        total_time_seconds += epoch_duration
        avg_epoch_loss = epoch_loss / max(1, len(train_loader)) 
        print(f"Lite Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}. Time: {epoch_duration:.2f}s. LR: {optimizer.param_groups[0]['lr']:.2e}")
        writer.add_scalar('Lite/Train/Loss_epoch', avg_epoch_loss, epoch + 1)
        
        if val_loader:
            evaluate(model, val_loader, DEVICE, pad_token_id, USE_AMP, "Lite", writer, step) # Use current step for epoch eval too
            model.train()
        
        save_checkpoint(model, optimizer, epoch, step, scheduler, checkpoint_dir, filename=f"lite_ckpt_epoch_{epoch+1}.pt", model_type="Lite")

    print(f"Lite training complete. Total time: {total_time_seconds/3600:.2f} hours.")
    save_checkpoint(model, optimizer, epochs, step, scheduler, checkpoint_dir, filename="lite_final_model.pt", model_type="Lite")
    writer.close()

def evaluate(model, val_loader, device, pad_token_id, use_amp, model_type_str, writer, current_global_step): 
    model.eval()
    total_val_loss = 0
    print(f"\nEvaluating {model_type_str} model on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"{model_type_str} Validation"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            intra_line_pos_ids = batch["intra_line_pos_ids"].to(device)

            current_loss_val = 0.0
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
            else:
                 _, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
            
            if loss is not None:
                current_loss_val = loss.item()
                total_val_loss += current_loss_val
    
    avg_val_loss = total_val_loss / max(1, len(val_loader)) 
    perplexity = torch.exp(torch.tensor(avg_val_loss)) if avg_val_loss > 0 and len(val_loader) > 0 else float('inf')
    print(f"{model_type_str} Validation Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")

    # Log to TensorBoard
    writer.add_scalar(f'{model_type_str}/Val/Loss', avg_val_loss, current_global_step)
    writer.add_scalar(f'{model_type_str}/Val/Perplexity', perplexity, current_global_step)
    return avg_val_loss

def save_checkpoint(model, optimizer, epoch, step, scheduler, checkpoint_dir, filename=None, model_type="Model"): 
    if filename is None:
        filename = f"{model_type.lower()}_ckpt_step_{step}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model # Handle compiled model

    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 
        'config': model_to_save.config if hasattr(model_to_save, 'config') else None
    }, checkpoint_path)
    print(f"{model_type} Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train_lite()