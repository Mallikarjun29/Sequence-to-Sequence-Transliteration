import torch

class CharTokenizer:
    """Character-level tokenizer for transliteration tasks."""

    def __init__(self, chars=None, sos_token='\t', eos_token='\n', pad_token='_'):
        """
        Initializes the CharTokenizer.

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

        # Store special tokens as attributes
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token # Store pad_token string
        self.pad_token_id = 0      # Padding index is always 0

        # If initial characters are provided, build the vocabulary
        if chars is not None:
            self.fit(chars) # Call fit with the initial characters

    def fit(self, texts):
        """
        Build vocabulary from a list of texts or a list of characters.
        Includes special tokens.
        """
        unique_chars = set()
        if isinstance(texts, list) and all(isinstance(t, str) for t in texts):
            # If input is a list of strings, extract unique characters
            for text in texts:
                unique_chars.update(text)
        elif isinstance(texts, list) and all(isinstance(t, str) and len(t) == 1 for t in texts):
             # If input is a list of characters
             unique_chars.update(texts)
        else:
             raise TypeError("Input to fit must be a list of strings or a list of characters.")


        # Add special tokens to the set of unique characters
        unique_chars.add(self.sos_token)
        unique_chars.add(self.eos_token)
        unique_chars.add(self.pad_token) # Add pad token string

        # Sort characters for consistent mapping (excluding PAD token initially)
        sorted_chars = sorted(list(unique_chars - {self.pad_token}))

        # Assign index 1 onwards to sorted characters
        for idx, char in enumerate(sorted_chars):
            self.char_to_idx[char] = idx + 1

        # Assign index 0 to the padding token
        self.char_to_idx[self.pad_token] = self.pad_token_id

        # Create index to character mapping
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        self.num_tokens = len(self.char_to_idx) # Total number of tokens including specials

        print(f"Tokenizer vocabulary built. Total tokens: {self.num_tokens}")
        # print(f"Char to idx: {self.char_to_idx}") # Optional: print vocab for debugging

    def encode(self, text):
        """Convert text to token IDs, including SOS and EOS."""
        # Add SOS and EOS tokens to the text before encoding
        text_with_specials = self.sos_token + text + self.eos_token
        # Use .get() with a default value (e.g., pad_token_id) for unknown characters
        return [self.char_to_idx.get(char, self.pad_token_id) for char in text_with_specials]

    def decode(self, ids):
        """Convert token IDs back to text, excluding PAD, SOS, and EOS."""
        decoded_chars = []
        for id in ids:
            char = self.idx_to_char.get(id)
            # Exclude PAD, SOS, and EOS tokens from the decoded string
            if char is not None and char not in [self.pad_token, self.sos_token, self.eos_token]:
                decoded_chars.append(char)
        return ''.join(decoded_chars)

    def __len__(self):
        return self.num_tokens

    # Add methods to get special token IDs for convenience
    def sos_id(self):
        return self.char_to_idx.get(self.sos_token, self.pad_token_id) # Return pad_token_id if SOS not in vocab

    def eos_id(self):
        return self.char_to_idx.get(self.eos_token, self.pad_token_id) # Return pad_token_id if EOS not in vocab

    def pad_id(self):
        return self.pad_token_id # Pad ID is always 0

