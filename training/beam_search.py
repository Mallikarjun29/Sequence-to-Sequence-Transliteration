import torch
import torch.nn.functional as F
import numpy as np

class BeamSearch:
    """Beam search for sequence-to-sequence model."""

    def __init__(self, model, beam_width=3):
        self.model = model
        self.beam_width = beam_width
        # Access the transformation layers from the model
        self.encoder_to_decoder_hidden = model.encoder_to_decoder_hidden
        self.encoder_to_decoder_cell = model.encoder_to_decoder_cell if model.rnn_type == "lstm" else None
        self.num_layers = model.num_layers
        self.hidden_size = model.hidden_size # Unidirectional hidden size
        self.rnn_type = model.rnn_type


    def translate(self, source, source_tokenizer, target_tokenizer, max_length=50, device="cpu"):
        """Translate a source sequence using beam search."""
        self.model.eval()  # Set model to evaluation mode

        # Encode source sequence
        # Ensure source_tokenizer.encode returns a list of integers
        source_tensor = torch.tensor(source_tokenizer.encode(source), device=device).unsqueeze(0)
        # encoder_outputs shape: [1, source_len, hidden_size * 2]
        # encoder_hidden shape: (num_layers * 2, 1, hidden_size) for GRU/RNN
        #                      ((num_layers * 2, 1, hidden_size), (num_layers * 2, 1, hidden_size)) for LSTM
        encoder_outputs, encoder_hidden = self.model.encoder(source_tensor)

        # --- Prepare initial decoder hidden state from bidirectional encoder hidden state ---
        # This logic is adapted from Seq2SeqModel.forward and Seq2SeqModel.translate
        if self.rnn_type == "lstm":
            # encoder_hidden is (h_n, c_n)
            # h_n shape: (num_layers * 2, 1, hidden_size)
            # c_n shape: (num_layers * 2, 1, hidden_size)

            # Take the final hidden state of the LAST layer of the encoder (forward and backward)
            # Last layer forward hidden state: encoder_hidden[0][-2, :, :]
            # Last layer backward hidden state: encoder_hidden[0][-1, :, :]
            # Concatenate them: [1, hidden_size * 2]
            last_layer_encoder_h = torch.cat((encoder_hidden[0][-2, :, :], encoder_hidden[0][-1, :, :]), dim=1)
            last_layer_encoder_c = torch.cat((encoder_hidden[1][-2, :, :], encoder_hidden[1][-1, :, :]), dim=1)

            # Project to decoder hidden size using the transformation layers from Seq2SeqModel
            # projected_h shape: [1, hidden_size]
            # projected_c shape: [1, hidden_size]
            projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
            projected_c = torch.tanh(self.encoder_to_decoder_cell(last_layer_encoder_c))

            # Repeat the projected state for all decoder layers
            # initial_decoder_hidden_h shape: (num_layers, 1, hidden_size)
            # initial_decoder_hidden_c shape: (num_layers, 1, hidden_size)
            initial_decoder_hidden_h = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)
            initial_decoder_hidden_c = projected_c.unsqueeze(0).repeat(self.num_layers, 1, 1)

            initial_decoder_hidden = (initial_decoder_hidden_h, initial_decoder_hidden_c)

        else: # GRU or RNN
            # encoder_hidden shape: (num_layers * 2, 1, hidden_size)

            # Take the final hidden state of the LAST layer of the encoder (forward and backward)
            # Last layer forward hidden state: encoder_hidden[-2, :, :]
            # Last layer backward hidden state: encoder_hidden[-1, :, :]
            # Concatenate them: [1, hidden_size * 2]
            last_layer_encoder_h = torch.cat((encoder_hidden[-2, :, :], encoder_hidden[-1, :, :]), dim=1)

            # Project to decoder hidden size using the transformation layer from Seq2SeqModel
            # projected_h shape: [1, hidden_size]
            projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))

            # Repeat the projected state for all decoder layers
            # initial_decoder_hidden shape: (num_layers, 1, hidden_size)
            initial_decoder_hidden = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)
        # --- End Prepare initial decoder hidden state ---


        # Initialize beams with SOS token and the initial decoder hidden state
        SOS_token_id = target_tokenizer.sos_id()
        EOS_token_id = target_tokenizer.eos_id()
        PAD_token_id = target_tokenizer.pad_id()

        # Initial beam: (sequence, score, decoder_hidden)
        # Use the derived initial_decoder_hidden
        beams = [([SOS_token_id], 0.0, initial_decoder_hidden)]
        completed_beams = []

        # Beam search for max_length steps
        for step in range(max_length):
            candidates = []
            # Create a list to hold the next set of beams (not strictly needed here, but good practice)
            # next_beams = []

            # Sort current beams by score to process the most promising ones first
            current_beams = sorted(beams, key=lambda x: x[1])[:self.beam_width]
            beams = [] # Clear beams for the next step, will be repopulated with candidates

            for sequence, score, decoder_hidden in current_beams:
                # Check if the sequence ends with EOS
                if sequence[-1] == EOS_token_id:
                    # Append completed beam with None for the hidden state to maintain tuple size 3
                    completed_beams.append((sequence, score, None)) # <--- Changed here
                    # If we have enough completed beams, we can stop adding more candidates from this step
                    if len(completed_beams) >= self.beam_width:
                         continue # Skip generating candidates from this completed beam

                # If not completed, prepare decoder input (the last token)
                decoder_input = torch.tensor([[sequence[-1]]], device=device)

                # Get predictions for the next token using the last token and the passed hidden state
                with torch.no_grad():
                    # Pass the decoder_hidden state from the previous step
                    # encoder_outputs shape: [1, source_len, hidden_size * 2] (handled by decoder's attention if used)
                    output, new_decoder_hidden, _ = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)

                    # Get top k tokens and their log probabilities
                    log_probs = F.log_softmax(output, dim=-1)
                    topk_log_probs, topk_ids = log_probs.topk(self.beam_width)

                    # Create new candidate beams from the top k predictions
                    for i in range(self.beam_width):
                        token_id = topk_ids[0][i].item()
                        token_log_prob = topk_log_probs[0][i].item()

                        # Avoid adding padding token to the sequence during generation
                        if token_id == PAD_token_id:
                            continue

                        new_sequence = sequence + [token_id]
                        new_score = score - token_log_prob
                        # Store the new_decoder_hidden state along with the new sequence and score
                        new_decoder_hidden_for_beam = new_decoder_hidden # Use the updated hidden state
                        candidates.append((new_sequence, new_score, new_decoder_hidden_for_beam))

            # If no new candidates are generated (e.g., all beams ended with EOS or max_length reached)
            if not candidates:
                 break

            # Select the top beam_width candidates from all generated candidates across all current beams
            # Sort by score (lower is better for negative log probability)
            candidates = sorted(candidates, key=lambda x: x[1])[:self.beam_width]

            # Add the top candidates as the beams for the next step
            beams = candidates

            # If we have enough completed beams AND no beams left to explore, break
            # Check if we have enough completed beams OR if there are no more beams to explore
            if len(completed_beams) >= self.beam_width or not beams: # <--- Adjusted early stopping condition
                 break


        # After the loop, add any remaining beams that haven't reached EOS to completed_beams
        # This handles cases where max_length is reached before EOS
        # Only add if there are still beams left
        if beams:
             completed_beams.extend(beams)

        # Sort all completed beams by score (negative log probability)
        completed_beams = sorted(completed_beams, key=lambda x: x[1])

        results = []
        # Decode sequences and return results
        # Use the tokenizers' decode method, which should handle special tokens
        # Only decode the sequence part of the beam tuple
        for sequence, score, _ in completed_beams: # Ignore the hidden state when returning results
            decoded_text = target_tokenizer.decode(sequence)
            results.append((decoded_text, score))

        return results

