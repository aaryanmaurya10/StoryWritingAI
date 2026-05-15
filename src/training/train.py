import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

import torch
import math
from tqdm import tqdm
from src.model.gpt_model import StoryGPT
from src.tokenizer.tokenizer_utils import StoryTokenizer
from src.dataset.dataloader import get_dataloader

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

def main():
    vocab_size = 16000
    d_model = 512
    n_layers = 6
    n_heads = 8
    max_len = 512
    batch_size = 16
    accumulation_steps = 4
    epochs = 10
    max_lr = 3e-4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = StoryTokenizer()
    train_loader = get_dataloader("data/cleaned/train.jsonl", tokenizer, batch_size, max_len)
    val_loader = get_dataloader("data/cleaned/validation.jsonl", tokenizer, batch_size, max_len)

    model = StoryGPT(vocab_size, d_model, n_heads, n_layers, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    
    best_val_loss = float('inf') 
    step_count = 0

    print(f"Starting Master Run with Validation Tracking...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        epoch_train_loss = 0
        
        for i, (x, y) in enumerate(train_pbar):
            x, y = x.to(device), y.to(device)
            
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


        val_loss = validate(model, val_loader, device)
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/checkpoints/pt_best.pt")
            print(f"New Best Model Saved! (Val Loss: {val_loss:.4f})")
        
        torch.save(model.state_dict(), f"outputs/checkpoints/gpt_ep{epoch}.pt")

if __name__ == "__main__":
    main()