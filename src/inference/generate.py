import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

import torch
import torch.nn.functional as F
from src.model.gpt_model import StoryGPT
from src.tokenizer.tokenizer_utils import StoryTokenizer

class StoryGenerator:
    def __init__(self, checkpoint_path, tokenizer_path="outputs/tokenizer.json"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available(): self.device = "mps"
        
        self.tokenizer = StoryTokenizer(tokenizer_path)
        
        # --- MUST MATCH MASTER TRAIN CONFIG ---
        self.model = StoryGPT(
            vocab_size=16000, 
            d_model=512, 
            n_heads=8,
            n_layers=6,
            max_len=256
        ).to(self.device)
        
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Master Model loaded from {checkpoint_path}")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=300, temperature=0.8, top_k=50, top_p=0.9):
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        idx = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -256:] # Match max_len
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, next_token), dim=1)
            
            if next_token.item() == self.tokenizer.eos_id:
                break
                
        return self.tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    checkpoint = "outputs/checkpoints/pt_best.pt"
    
    gen = StoryGenerator(checkpoint)
    
    my_prompt = "A hard working man suddenly fell sick"
    
    story = gen.generate(my_prompt, max_new_tokens=250, temperature=0.85)
    
    print("\n" + "="*40)
    print("GENERATED TEXT")
    print("="*40)
    print(story)