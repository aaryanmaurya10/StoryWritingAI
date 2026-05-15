import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers to create Q, K, and V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        
        # 1. Create Q, K, V and split into multiple heads
        q = self.w_q(x).view(batch_size, seq_length, self.num_heads, self.d_k)
        k = self.w_k(x).view(batch_size, seq_length, self.num_heads, self.d_k)
        v = self.w_v(x).view(batch_size, seq_length, self.num_heads, self.d_k)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v) # (batch, heads, seq_len, d_k)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        return self.fc_out(out)
    