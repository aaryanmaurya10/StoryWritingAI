import os
import json
from tqdm import tqdm

def clean_text(text):
    """
    Basic cleaning: remove extra whitespace and fix simple formatting.
    """
    if not text:
        return ""
    text = " ".join(text.split())
    return text

def process_files(input_dir="data/raw", output_dir="data/cleaned"):
    """
    Filters out stories that are too short and cleans the text.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in ["train.jsonl", "validation.jsonl"]:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"Skipping {filename}: Raw file not found.")
            continue

        print(f"Cleaning {filename}...")
        cleaned_count = 0
        total_count = 0

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in):
                total_count += 1
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    
                    text = clean_text(text)
                    
                    if len(text) > 100:
                        f_out.write(json.dumps({"text": text}) + '\n')
                        cleaned_count += 1
                except json.JSONDecodeError:
                    continue

        print(f"Finished {filename}: Kept {cleaned_count}/{total_count} stories.")

if __name__ == "__main__":
    process_files()