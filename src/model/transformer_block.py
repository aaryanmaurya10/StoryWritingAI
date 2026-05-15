import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))


import torch
import torch.nn as nn
from src.model.attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    A simple linear network: Expand -> Activate -> Contract.
    This allows the model to process the information gathered by attention.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    The 'modular brick' of our GPT model.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.feed_forward = FeedForward(d_model, d_ff=4 * d_model, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x -> Norm -> Attention -> Add back to original x
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, mask=mask)
        x = x + self.dropout(attn_out)
        
        # x -> Norm -> Feed-Forward -> Add back to original x
        norm_x = self.norm2(x)
        ff_out = self.feed_forward(norm_x)
        x = x + self.dropout(ff_out)
        
        return x
    