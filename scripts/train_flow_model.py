import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from beefai.flow_model.tokenizer import FlowTokenizer
from beefai.flow_model.transformer_model import FlowTransformerDecoder, FlowGPTConfig
from beefai.flow_model.dataset import FlowDataset # Make sure this is correctly importable

# --- Configuration ---
# Data paths
TOKENIZER_PATH = "data/tokenizer_config.json"
TRAIN_DATA_PATH = "data/tokenized/train_data.pt"
VAL_DATA_PATH = "data/tokenized/val_data.pt" # Optional, but highly recommended
CHECKPOINT_DIR = "data/checkpoints/flow_model/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Model Hyperparameters
BLOCK_SIZE = 256  # Max sequence length (must match data tokenization)
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384 # n_embd must be div by n_head
MAX_SEGMENT_TYPES = 4 # Should match tokenizer/data prep (e.g., for up to 4 lines per bar)
MAX_INTRA_LINE_POSITIONS = 4 # For LINE_START, SYL, OFFSET, DUR
DROPOUT = 0.1

# Training Hyperparameters
BATCH_SIZE = 16 # Adjust based on GPU memory
LEARNING_RATE = 3e-4
EPOCHS = 10
GRAD_ACCUMULATION_STEPS = 4 # Accumulate gradients for effective batch size BATCH_SIZE * GRAD_ACCUMULATION_STEPS
EVAL_INTERVAL = 200 # Steps
SAVE_INTERVAL = 1000 # Steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print(f"Using device: {DEVICE}")

    # 1. Initialize Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer config not found at {TOKENIZER_PATH}. Please run data preparation scripts first.")
        return
    tokenizer = FlowTokenizer(config_path=TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.pad_token_id # Important for loss ignore_index

    # 2. Initialize Model
    model_config = FlowGPTConfig(
        vocab_size=vocab_size, 
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        max_segment_types=MAX_SEGMENT_TYPES,
        max_intra_line_positions=MAX_INTRA_LINE_POSITIONS,
        dropout=DROPOUT
    )
    # Attach pad_token_id to config for the model to use in loss
    model_config.pad_token_id = pad_token_id 
    
    model = FlowTransformerDecoder(model_config).to(DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 3. Load Data
    train_dataset = FlowDataset(TRAIN_DATA_PATH, tokenizer, BLOCK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True if DEVICE=="cuda" else False)
    
    val_loader = None
    if os.path.exists(VAL_DATA_PATH):
        val_dataset = FlowDataset(VAL_DATA_PATH, tokenizer, BLOCK_SIZE)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95) # Example

    # 5. Training Loop
    model.train()
    step = 0
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        epoch_loss = 0
        optimizer.zero_grad() # Reset gradients at the start of each epoch / accumulation cycle

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            segment_ids = batch["segment_ids"].to(DEVICE)
            intra_line_pos_ids = batch["intra_line_pos_ids"].to(DEVICE)

            logits, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
            
            if loss is not None:
                loss = loss / GRAD_ACCUMULATION_STEPS # Normalize loss for accumulation
                loss.backward()
                epoch_loss += loss.item() * GRAD_ACCUMULATION_STEPS # Accumulate actual loss

            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % EVAL_INTERVAL == 0 and val_loader:
                    evaluate(model, val_loader, DEVICE, pad_token_id)
                    model.train() # Switch back to train mode

                if step % SAVE_INTERVAL == 0:
                    save_checkpoint(model, optimizer, epoch, step, CHECKPOINT_DIR)
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Average Training Loss: {avg_epoch_loss:.4f}")
        # if scheduler: scheduler.step()

        if val_loader: # Final validation for the epoch
            evaluate(model, val_loader, DEVICE, pad_token_id)
            model.train()
        
        save_checkpoint(model, optimizer, epoch, step, CHECKPOINT_DIR, filename=f"checkpoint_epoch_{epoch+1}.pt")

    print("Training complete.")
    save_checkpoint(model, optimizer, EPOCHS, step, CHECKPOINT_DIR, filename="final_model.pt")

def evaluate(model, val_loader, device, pad_token_id):
    model.eval()
    total_val_loss = 0
    print("\nEvaluating on validation set...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            segment_ids = batch["segment_ids"].to(device)
            intra_line_pos_ids = batch["intra_line_pos_ids"].to(device)

            _, loss = model(input_ids, segment_ids=segment_ids, intra_line_pos_ids=intra_line_pos_ids, targets=target_ids)
            if loss is not None:
                total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Perplexity: {torch.exp(torch.tensor(avg_val_loss)):.2f}")
    return avg_val_loss

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, filename=None):
    if filename is None:
        filename = f"checkpoint_step_{step}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config # Save model config
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()