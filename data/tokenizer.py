import torch

class CharTokenizer:
    """Character-level tokenizer for transliteration tasks.
    
    This tokenizer handles character-level tokenization, including special tokens
    for start of sequence, end of sequence, and padding. It provides methods for
    encoding text to token IDs and decoding token IDs back to text.
    """

    def __init__(self, chars=None, sos_token='\t', eos_token='\n', pad_token='_'):
        """Initialize the CharTokenizer.

        Args:
            chars (list, optional): List of characters to build the vocabulary from.
                                   If None, vocabulary will be built from data using fit().
            sos_token (str): String representation of the Start-of-Sequence token.
            eos_token (str): String representation of the End-of-Sequence token.
            pad_token (str): String representation of the Padding token.
        """
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.num_tokens = 0

        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.pad_token_id = 0

        if chars is not None:
            self.fit(chars)

    def fit(self, texts):
        """Build vocabulary from a list of texts or a list of characters.
        
        Args:
            texts (list): List of strings or list of characters to build vocabulary from.
            
        Raises:
            TypeError: If input is not a list of strings or a list of characters.
        """
        unique_chars = set()
        if isinstance(texts, list) and all(isinstance(t, str) for t in texts):
            for text in texts:
                unique_chars.update(text)
        elif isinstance(texts, list) and all(isinstance(t, str) and len(t) == 1 for t in texts):
             unique_chars.update(texts)
        else:
             raise TypeError("Input to fit must be a list of strings or a list of characters.")

        unique_chars.add(self.sos_token)
        unique_chars.add(self.eos_token)
        unique_chars.add(self.pad_token)

        sorted_chars = sorted(list(unique_chars - {self.pad_token}))

        for idx, char in enumerate(sorted_chars):
            self.char_to_idx[char] = idx + 1

        self.char_to_idx[self.pad_token] = self.pad_token_id

        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        self.num_tokens = len(self.char_to_idx)

        print(f"Tokenizer vocabulary built. Total tokens: {self.num_tokens}")

    def encode(self, text):
        """Convert text to token IDs, including SOS and EOS.
        
        Args:
            text (str): Text to encode.
            
        Returns:
            list: List of token IDs.
        """
        text_with_specials = self.sos_token + text + self.eos_token
        return [self.char_to_idx.get(char, self.pad_token_id) for char in text_with_specials]

    def decode(self, ids):
        """Convert token IDs back to text, excluding PAD, SOS, and EOS.
        
        Args:
            ids (list): List of token IDs.
            
        Returns:
            str: Decoded text.
        """
        decoded_chars = []
        for id in ids:
            char = self.idx_to_char.get(id)
            if char is not None and char not in [self.pad_token, self.sos_token, self.eos_token]:
                decoded_chars.append(char)
        return ''.join(decoded_chars)

    def __len__(self):
        """Return the number of tokens in the vocabulary.
        
        Returns:
            int: Number of tokens.
        """
        return self.num_tokens

    def sos_id(self):
        """Get the ID for the start-of-sequence token.
        
        Returns:
            int: ID of the SOS token or pad token ID if SOS is not in vocabulary.
        """
        return self.char_to_idx.get(self.sos_token, self.pad_token_id)

    def eos_id(self):
        """Get the ID for the end-of-sequence token.
        
        Returns:
            int: ID of the EOS token or pad token ID if EOS is not in vocabulary.
        """
        return self.char_to_idx.get(self.eos_token, self.pad_token_id)

    def pad_id(self):
        """Get the ID for the padding token.
        
        Returns:
            int: ID of the padding token (always 0).
        """
        return self.pad_token_id

