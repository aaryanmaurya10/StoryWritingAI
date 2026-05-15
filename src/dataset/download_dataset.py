import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

def download_and_split(split_ratio=0.9):
    """
    Downloads, mixes, and splits multiple datasets into Train and Validation sets.
    """
    all_data = []
    
    # Dataset A: TinyStories (Simple)
    print("Fetching TinyStories...")
    ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, entry in enumerate(ts):
        if i >= 20000: break # Grab 20k simple stories
        all_data.append({"text": entry['text']})

    # Dataset B: WritingPrompts (Creative)
    try:
        print("Fetching WritingPrompts...")
        wp = load_dataset("euclaise/writingprompts", split="train", streaming=True)
        for i, entry in enumerate(wp):
            if i >= 20000: break
            all_data.append({"text": entry['story']})
    except Exception as e:
        print(f"WritingPrompts skip: {e}")

    # Dataset C: Gutenberg (Classical)
    try:
        print("Fetching Gutenberg...")
        gb = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
        for i, entry in enumerate(gb):
            if i >= 20000: break
            all_data.append({"text": entry['TEXT']})
    except Exception as e:
        print(f"Gutenberg skip: {e}")

    
    print("Shuffling the master mix...")
    random.shuffle(all_data)

    split_idx = int(len(all_data) * split_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    output_dir = "data/cleaned"
    os.makedirs(output_dir, exist_ok=True)

    for name, data in [("train", train_data), ("validation", val_data)]:
        path = os.path.join(output_dir, f"{name}.jsonl")
        print(f"Saving {len(data)} stories to {path}...")
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                clean_text = " ".join(item['text'].split())
                if len(clean_text) > 100:
                    f.write(json.dumps({"text": clean_text}) + '\n')

    print("All set! Master dataset is split and ready for the tokenizer.")

if __name__ == "__main__":
    download_and_split()