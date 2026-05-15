import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))


import torch
import torch.nn as nn
from src.model.transformer_block import TransformerBlock

class StoryGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, vocab_size)
        
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.shape
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.dropout(x)
        
        mask = torch.tril(torch.ones((t, t), device=device)).view(1, 1, t, t)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = nn.functional.cross_entropy(logits, targets)
            
        return logits, loss

