import sys
import os
import re
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

import torch
import math
from tqdm import tqdm
from src.model.gpt_model import StoryGPT
from src.tokenizer.tokenizer_utils import StoryTokenizer
from src.dataset.dataloader import get_dataloader
from src.utils.metrics_logger import MetricsLogger

@torch.no_grad()
def validate(model, val_loader, device):
    """Checks the model's performance on unseen data."""
    model.eval()
    total_loss = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()
    return total_loss / len(val_loader)

def find_latest_checkpoint(checkpoint_dir="outputs/checkpoints"):
    """Scans the directory and returns the path of the latest epoch checkpoint and its number."""
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    files = os.listdir(checkpoint_dir)
    epoch_checkpoints = []
    
    for f in files:
        match = re.match(r"gpt_ep(\d+)\.pt", f)
        if match:
            epoch_num = int(match.group(1))
            epoch_checkpoints.append((epoch_num, os.path.join(checkpoint_dir, f)))
            
    if not epoch_checkpoints:
        return None, 0
    
    epoch_checkpoints.sort(key=lambda x: x[0])
    return epoch_checkpoints[-1][1], epoch_checkpoints[-1][0]

def main():
    vocab_size = 16000
    d_model = 512
    n_layers = 6
    n_heads = 8
    max_len = 256
    batch_size = 16
    accumulation_steps = 4
    epochs = 10
    max_lr = 3e-4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 50)
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"HARDWARE ACCELERATION ENABLED")
        print(f"Using GPU Device: {gpu_name}")
        torch.backends.cudnn.benchmark = True 
    else:
        print("CUDA GPU NOT FOUND. Falling back to CPU training.")
    print("=" * 50)

    tokenizer = StoryTokenizer()
    train_loader = get_dataloader("data/cleaned/train.jsonl", tokenizer, batch_size, max_len)
    val_loader = get_dataloader("data/cleaned/validation.jsonl", tokenizer, batch_size, max_len)

    model = StoryGPT(vocab_size, d_model, n_heads, n_layers, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)

    checkpoint_path, last_epoch = find_latest_checkpoint()
    start_epoch = last_epoch + 1

    logger = MetricsLogger(output_dir="outputs/metrics")

    steps_per_epoch = len(train_loader)
    step_count = last_epoch * steps_per_epoch
    total_steps = len(train_loader) * epochs

    if checkpoint_path:
        print(f"Found previous checkpoint: {checkpoint_path}")
        print(f"Resuming training starting from Epoch {start_epoch}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No previous checkpoints found. Starting a fresh training run.")
    print("=" * 50)

    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    print("Starting Master Run with Metric Logging Enabled...")
    
    best_val_loss = float('inf') 
    step_count = 0

    print(f"Starting Master Run with Validation Tracking...")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        epoch_train_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        for i, (x, y) in enumerate(train_pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            

            step_count += 1

            logits, loss = model(x, targets=y)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_loss += loss.item() * accumulation_steps
            train_pbar.set_postfix({"loss": loss.item() * accumulation_steps})

            if i % 50 == 0 and device == "cuda":
                allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
                train_pbar.set_postfix({
                    "loss": f"{loss.item() * accumulation_steps:.4f}",
                    "VRAM": f"{allocated_vram:.0f}MB"
                })
            else:
                train_pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        val_loss = validate(model, val_loader, device)
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")

        logger.log_epoch(epoch, avg_train_loss, val_loss)
        print("Metrics saved and training graphs updated successfully.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/checkpoints/pt_best.pt")
            print(f"New Best Model Saved! (Val Loss: {val_loss:.4f})")
        
        torch.save(model.state_dict(), f"outputs/checkpoints/gpt_ep{epoch}.pt")

if __name__ == "__main__":
    main()