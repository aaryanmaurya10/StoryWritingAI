from tokenizers import Tokenizer

class StoryTokenizer:
    def __init__(self, path="outputs/tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(path)
        
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")

    def encode(self, text, add_bos=True, add_eos=False):
        """Turns string into a list of numbers."""
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        """Turns a list of numbers back into a string."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()