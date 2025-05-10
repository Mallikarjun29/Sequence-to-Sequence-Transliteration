import torch
import torch.nn as nn
import random
from encoder import Encoder
from decoder import Decoder

class Seq2SeqModel(nn.Module):
    """Sequence-to-sequence model for transliteration."""
    
    def __init__(self, source_vocab_size, target_vocab_size, embedding_dim, hidden_size, 
                 num_layers, dropout=0.1, rnn_type="lstm"):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = Encoder(
            source_vocab_size, embedding_dim, hidden_size, 
            num_layers, dropout, rnn_type
        )
        
        self.decoder = Decoder(
            target_vocab_size, embedding_dim, hidden_size,
            num_layers, dropout, rnn_type
        )
        
        self.target_vocab_size = target_vocab_size
        
    def forward(self, source, target, teacher_forcing_ratio=1.0):
        """
        Args:
            source: Source sequence [batch_size, source_len]
            target: Target sequence [batch_size, target_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Decoder outputs [batch_size, target_len, target_vocab_size]
        """
        batch_size = source.shape[0]
        target_len = target.shape[1]
        device = source.device
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, target_len, self.target_vocab_size).to(device)
        
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(source)
        
        # First input to the decoder is the SOS token
        decoder_input = target[:, 0].unsqueeze(1)
        decoder_hidden = encoder_hidden

        # Decode one step at a time
        for t in range(1, target_len):
            decoder_output, decoder_hidden= self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            # Store output and attention weights
            outputs[:, t, :] = decoder_output
            
            # Teacher forcing: use actual target token as next input
            # or use predicted token
            if random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach().unsqueeze(1)
        
        return outputs
    
    def translate(self, source, source_tokenizer, target_tokenizer, max_length=50, device="cpu"):
        """Translate a single source sequence."""
        self.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            # Encode source sequence
            source_tensor = torch.tensor(source_tokenizer.encode(source), device=device).unsqueeze(0)
            encoder_outputs, encoder_hidden = self.encoder(source_tensor)
            
            # Initialize decoder input with SOS token
            decoder_input = torch.tensor([[target_tokenizer.char_to_idx["\t"]]], device=device)
            decoder_hidden = encoder_hidden
            
            decoded_chars = []
            
            # Decode until max length or EOS token
            for _ in range(max_length):
                decoder_output, decoder_hidden= self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                
                # Get most likely token
                topv, topi = decoder_output.topk(1)
                token = topi.item()
                
                if token == target_tokenizer.char_to_idx["\n"]:
                    # End of sequence
                    break
                
                # Add token to output
                decoded_chars.append(target_tokenizer.idx_to_char[token])
                
                
                # Next input is the predicted token
                decoder_input = topi.detach()
        
        return ''.join(decoded_chars)
    

if __name__ == "__main__":
    import sys
    import torch
    sys.path.append('..')
    from data.tokenizer import CharTokenizer
    
    # Create sample tokenizers
    source_chars = ["\t", "a", "b", "c", "d", "e", "f", "\n"]
    target_chars = ["\t", "x", "y", "z", "w", "\n"]
    source_tokenizer = CharTokenizer()
    source_tokenizer.fit(["".join(source_chars)])
    target_tokenizer = CharTokenizer()
    target_tokenizer.fit(["".join(target_chars)])
    
    # Model parameters
    source_vocab_size = len(source_tokenizer)
    target_vocab_size = len(target_tokenizer)
    embedding_dim = 16
    hidden_size = 32
    num_layers = 1
    
    # Create model
    model = Seq2SeqModel(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    # Test translate function
    source_text = "abc"
    translation= model.translate(
        source=source_text,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer
    )
    
    print(f"Source: {source_text}")
    print(f"Translation: {translation}")