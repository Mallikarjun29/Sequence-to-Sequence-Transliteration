import torch
import torch.nn.functional as F
import numpy as np

class BeamSearch:
    """Beam search for sequence-to-sequence model."""
    
    def __init__(self, model, beam_width=3):
        self.model = model
        self.beam_width = beam_width
        
    def translate(self, source, source_tokenizer, target_tokenizer, max_length=50, device="cpu"):
        """Translate a source sequence using beam search."""
        self.model.eval()  # Set model to evaluation mode
        
        # Encode source sequence
        source_tensor = torch.tensor(source_tokenizer.encode(source), device=device).unsqueeze(0)
        encoder_outputs, encoder_hidden = self.model.encoder(source_tensor)
        
        # Initialize beams with SOS token
        SOS_token_id = target_tokenizer.char_to_idx["\t"]
        EOS_token_id = target_tokenizer.char_to_idx["\n"]
        
        # Initial beam: (sequence, score)
        beams = [([SOS_token_id], 0.0)]
        completed_beams = []
        
        # Beam search for max_length steps
        for _ in range(max_length):
            candidates = []
            
            for sequence, score in beams:
                # Check if the sequence ends with EOS
                if sequence[-1] == EOS_token_id:
                    completed_beams.append((sequence, score))
                    continue
                
                # Prepare decoder input
                decoder_input = torch.tensor([[sequence[-1]]], device=device)
                
                # Set decoder hidden state
                if _ == 0:
                    decoder_hidden = encoder_hidden
                else:
                    # Run the decoder to update hidden state
                    with torch.no_grad():
                        for idx in sequence[1:]:  # Skip SOS token
                            decoder_input = torch.tensor([[idx]], device=device)
                            _, decoder_hidden, _ = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                
                # Get predictions
                with torch.no_grad():
                    output, decoder_hidden, _ = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)
                    
                    # Get top k tokens
                    log_probs = F.log_softmax(output, dim=1)
                    topk_probs, topk_ids = log_probs.topk(self.beam_width)
                    
                    # Create new candidate beams
                    for i in range(self.beam_width):
                        token_id = topk_ids[0][i].item()
                        token_score = topk_probs[0][i].item()
                        new_sequence = sequence + [token_id]
                        new_score = score - token_score  # Using negative log probability
                        candidates.append((new_sequence, new_score))
            
            # If all beams are completed, break
            if len(candidates) == 0:
                break
            
            # Select top k candidates
            beams = sorted(candidates, key=lambda x: x[1])[:self.beam_width]
            
            # If beam has enough completed sequences, stop
            if len(completed_beams) >= self.beam_width:
                break
        
        # If no completed beams, use best incomplete beam
        if not completed_beams and beams:
            completed_beams = beams
        
        # Sort completed beams by score and return
        completed_beams = sorted(completed_beams, key=lambda x: x[1])
        
        # Decode sequences
        results = []
        for sequence, score in completed_beams:
            # Skip SOS and EOS tokens in the decoded result
            decoded = ''.join([target_tokenizer.idx_to_char.get(idx, '') 
                              for idx in sequence[1:] if idx != EOS_token_id])
            results.append((decoded, score))
        
        return results
