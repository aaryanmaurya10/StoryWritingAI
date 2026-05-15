import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_training_corpus(input_file):
    """
    A generator that yields the text from our JSONL file 
    so we don't load the whole file into RAM at once.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

def train_my_tokenizer(input_file="data/cleaned/train.jsonl", vocab_size=8000):
    """
    Trains a BPE tokenizer on our specific story data.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
    )

    print(f"Starting training on {input_file}...")
    tokenizer.train_from_iterator(get_training_corpus(input_file), trainer=trainer)

    output_path = "outputs/tokenizer.json"
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")

if __name__ == "__main__":
    train_my_tokenizer()