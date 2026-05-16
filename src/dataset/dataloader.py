import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

import json
import torch
import os
from torch.utils.data import Dataset, DataLoader

class StoryDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_path = file_path
        
        self.offsets = []
        
        print(f"Indexing {file_path} for memory efficiency...")
        with open(file_path, 'rb') as f:
            offset = 0
            for line in f:
                self.offsets.append(offset)
                offset += len(line)
        
        print(f"Indexed {len(self.offsets)} stories using almost zero RAM!")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            
        data = json.loads(line)
        story = data.get("text", "")
        
        tokens = self.tokenizer.encode(story)

        if len(tokens) > self.max_length + 1:
            tokens = tokens[:self.max_length + 1]
        
        padding_len = (self.max_length + 1) - len(tokens)
        if padding_len > 0:
            tokens = tokens + [self.tokenizer.pad_id] * padding_len

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        return x, y

def get_dataloader(file_path, tokenizer, batch_size=32, max_length=256, shuffle=True, pin_memory=False):
    dataset = StoryDataset(file_path, tokenizer, max_length)
    # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0, 
        pin_memory=pin_memory
    )