import torch
import torch.nn.functional as F
import numpy as np

class BeamSearch:
    """Beam search for sequence-to-sequence model."""

    def __init__(self, model, beam_width=3):
        """Initialize the beam search decoder.
        
        Args:
            model: The sequence-to-sequence model to use for translation.
            beam_width (int): Width of the beam search.
        """
        self.model = model
        self.beam_width = beam_width
        self.encoder_to_decoder_hidden = model.encoder_to_decoder_hidden
        self.encoder_to_decoder_cell = model.encoder_to_decoder_cell if model.rnn_type == "lstm" else None
        self.num_layers = model.num_layers
        self.hidden_size = model.hidden_size
        self.rnn_type = model.rnn_type


    def translate(self, source, source_tokenizer, target_tokenizer, max_length=50, device="cpu"):
        """Translate a source sequence using beam search.
        
        Args:
            source (str): Source text to translate.
            source_tokenizer: Tokenizer for the source language.
            target_tokenizer: Tokenizer for the target language.
            max_length (int): Maximum length of the generated sequence.
            device (str): Device to run translation on.
            
        Returns:
            list: List of tuples (translation, score) sorted by score.
        """
        self.model.eval()

        source_tensor = torch.tensor(source_tokenizer.encode(source), device=device).unsqueeze(0)
        encoder_outputs, encoder_hidden = self.model.encoder(source_tensor)

        if self.rnn_type == "lstm":
            last_layer_encoder_h = torch.cat((encoder_hidden[0][-2, :, :], encoder_hidden[0][-1, :, :]), dim=1)
            last_layer_encoder_c = torch.cat((encoder_hidden[1][-2, :, :], encoder_hidden[1][-1, :, :]), dim=1)

            projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
            projected_c = torch.tanh(self.encoder_to_decoder_cell(last_layer_encoder_c))

            initial_decoder_hidden_h = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)
            initial_decoder_hidden_c = projected_c.unsqueeze(0).repeat(self.num_layers, 1, 1)

            initial_decoder_hidden = (initial_decoder_hidden_h, initial_decoder_hidden_c)

        else:
            last_layer_encoder_h = torch.cat((encoder_hidden[-2, :, :], encoder_hidden[-1, :, :]), dim=1)
            projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
            initial_decoder_hidden = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)

        SOS_token_id = target_tokenizer.sos_id()
        EOS_token_id = target_tokenizer.eos_id()
        PAD_token_id = target_tokenizer.pad_id()

        beams = [([SOS_token_id], 0.0, initial_decoder_hidden)]
        completed_beams = []

        for step in range(max_length):
            candidates = []
            
            current_beams = sorted(beams, key=lambda x: x[1])[:self.beam_width]
            beams = []

            for sequence, score, decoder_hidden in current_beams:
                if sequence[-1] == EOS_token_id:
                    completed_beams.append((sequence, score, None))
                    if len(completed_beams) >= self.beam_width:
                         continue

                decoder_input = torch.tensor([[sequence[-1]]], device=device)

                with torch.no_grad():
                    output, new_decoder_hidden, _ = self.model.decoder(decoder_input, decoder_hidden, encoder_outputs)

                    log_probs = F.log_softmax(output, dim=-1)
                    topk_log_probs, topk_ids = log_probs.topk(self.beam_width)

                    for i in range(self.beam_width):
                        token_id = topk_ids[0][i].item()
                        token_log_prob = topk_log_probs[0][i].item()

                        if token_id == PAD_token_id:
                            continue

                        new_sequence = sequence + [token_id]
                        new_score = score - token_log_prob
                        new_decoder_hidden_for_beam = new_decoder_hidden
                        candidates.append((new_sequence, new_score, new_decoder_hidden_for_beam))

            if not candidates:
                 break

            candidates = sorted(candidates, key=lambda x: x[1])[:self.beam_width]
            beams = candidates

            if len(completed_beams) >= self.beam_width or not beams:
                 break

        if beams:
             completed_beams.extend(beams)

        completed_beams = sorted(completed_beams, key=lambda x: x[1])

        results = []
        for sequence, score, _ in completed_beams:
            decoded_text = target_tokenizer.decode(sequence)
            results.append((decoded_text, score))

        return results

