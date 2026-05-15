import torch
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 2. Forward pass
            logits, loss = self.model(x, targets=y)
            
            # 3. Backward pass
            loss.backward()
            
            # 4. Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, loss = self.model(x, targets=y)
            total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved to {path}")