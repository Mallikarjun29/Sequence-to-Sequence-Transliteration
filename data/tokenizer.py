import torch

class CharTokenizer:
    """Character-level tokenizer for transliteration tasks."""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.num_tokens = 0
        
    def fit(self, texts):
        """Build vocabulary from list of texts."""
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
        
        # Create mapping
        for idx, char in enumerate(sorted(unique_chars)):
            self.char_to_idx[char] = idx + 1  # Reserve 0 for padding
            self.idx_to_char[idx + 1] = char
            
        # Add special tokens
        self.pad_token_id = 0
        self.idx_to_char[0] = '<PAD>'
        self.num_tokens = len(self.char_to_idx) + 1  # +1 for padding token
        
    def encode(self, text):
        """Convert text to token IDs."""
        return [self.char_to_idx.get(char, 0) for char in text]
        
    def decode(self, ids):
        """Convert token IDs back to text."""
        return ''.join([self.idx_to_char.get(id, '') for id in ids if id != 0])
    
    def __len__(self):
        return self.num_tokens