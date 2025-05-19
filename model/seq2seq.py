import torch
import torch.nn as nn
import random
from .encoder import Encoder
from .decoder import Decoder
import traceback
import numpy as np

class Seq2SeqModel(nn.Module):
    """Sequence-to-sequence model for transliteration."""

    def __init__(self, source_vocab_size, target_vocab_size, embedding_dim, hidden_size,
                 num_layers, dropout=0.1, rnn_type="lstm", attention=False):
        super(Seq2SeqModel, self).__init__()

        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.hidden_size = hidden_size # Unidirectional hidden size
        self.attention = attention
        self.target_vocab_size = target_vocab_size

        # Encoder is bidirectional
        self.encoder = Encoder(
            source_vocab_size, embedding_dim, hidden_size,
            num_layers, dropout, rnn_type
        )

        # Decoder is typically unidirectional
        # If using attention, decoder needs to know the size of encoder outputs (hidden_size * 2)
        encoder_output_size = hidden_size * 2 # Bidirectional encoder output size
        self.decoder = Decoder(
            target_vocab_size, embedding_dim, hidden_size, # Decoder hidden_size is still the unidirectional size
            num_layers, dropout, rnn_type, attention,
            encoder_output_size=encoder_output_size if attention else None # Pass encoder output size if attention
        )

        # Linear layer to transform bidirectional encoder hidden state to unidirectional decoder hidden state
        # This layer is needed because the encoder's final hidden state (num_layers * 2, batch_size, hidden_size)
        # needs to be transformed into the decoder's initial hidden state shape (num_layers, batch_size, hidden_size).
        # A common approach is to take the concatenated final forward and backward states of the encoder's last layer
        # and project it to the decoder's hidden size, then replicate it for all decoder layers.
        # Let's use a simpler approach: take the final hidden state of the LAST layer of the encoder (forward and backward),
        # concatenate them, and use a linear layer to project it to the decoder's hidden size.
        # This projected state will be used as the initial hidden state for *all* layers of the decoder.
        # This requires a linear layer that takes 2 * hidden_size and outputs hidden_size.
        # We'll apply this transformation for both h_n and c_n if using LSTM.

        # Linear layer for initial hidden state transformation
        # Input size is hidden_size * 2 (concatenated final forward/backward of last encoder layer)
        # Output size is hidden_size (for each decoder layer's initial state)
        self.encoder_to_decoder_hidden = nn.Linear(hidden_size * 2, hidden_size)

        # If using LSTM, we also need to transform the cell state
        if self.rnn_type == "lstm":
             self.encoder_to_decoder_cell = nn.Linear(hidden_size * 2, hidden_size)


    def forward(self, source, target, teacher_forcing_ratio=1.0):
        """
        Args:
            source: Source sequence [batch_size, source_len]
            target: Target sequence [batch_size, target_len]
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: Decoder outputs [batch_size, target_len, target_vocab_size]
            attention_weights: Attention weights if using attention (list of tensors [batch_size, source_len])
        """
        batch_size = source.shape[0]
        target_len = target.shape[1]
        device = source.device

        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, target_len, self.target_vocab_size).to(device)

        # Encode source sequence
        # encoder_outputs shape: [batch_size, source_len, hidden_size * 2]
        # encoder_hidden shape: (num_layers * 2, batch_size, hidden_size) for GRU/RNN
        #                      ((num_layers * 2, batch_size, hidden_size), (num_layers * 2, batch_size, hidden_size)) for LSTM
        encoder_outputs, encoder_hidden = self.encoder(source)

        # Prepare initial decoder hidden state from bidirectional encoder hidden state
        # We need to transform encoder_hidden (num_layers * 2, batch_size, hidden_size)
        # to decoder_hidden (num_layers, batch_size, hidden_size)

        if self.rnn_type == "lstm":
            # encoder_hidden is (h_n, c_n)
            # h_n shape: (num_layers * 2, batch_size, hidden_size)
            # c_n shape: (num_layers * 2, batch_size, hidden_size)

            # Take the final hidden state of the LAST layer of the encoder (forward and backward)
            # Last layer forward hidden state: encoder_hidden[0][-2, :, :]
            # Last layer backward hidden state: encoder_hidden[0][-1, :, :]
            # Concatenate them: [batch_size, hidden_size * 2]
            last_layer_encoder_h = torch.cat((encoder_hidden[0][-2, :, :], encoder_hidden[0][-1, :, :]), dim=1)
            last_layer_encoder_c = torch.cat((encoder_hidden[1][-2, :, :], encoder_hidden[1][-1, :, :]), dim=1)

            # Project to decoder hidden size
            # projected_h shape: [batch_size, hidden_size]
            # projected_c shape: [batch_size, hidden_size]
            projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
            projected_c = torch.tanh(self.encoder_to_decoder_cell(last_layer_encoder_c))

            # Repeat the projected state for all decoder layers
            # decoder_hidden_h shape: (num_layers, batch_size, hidden_size)
            # decoder_hidden_c shape: (num_layers, batch_size, hidden_size)
            decoder_hidden_h = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)
            decoder_hidden_c = projected_c.unsqueeze(0).repeat(self.num_layers, 1, 1)

            decoder_hidden = (decoder_hidden_h, decoder_hidden_c)

        else: # GRU or RNN
            # encoder_hidden shape: (num_layers * 2, batch_size, hidden_size)

            # Take the final hidden state of the LAST layer of the encoder (forward and backward)
            # Last layer forward hidden state: encoder_hidden[-2, :, :]
            # Last layer backward hidden state: encoder_hidden[-1, :, :]
            # Concatenate them: [batch_size, hidden_size * 2]
            last_layer_encoder_h = torch.cat((encoder_hidden[-2, :, :], encoder_hidden[-1, :, :]), dim=1)

            # Project to decoder hidden size
            # projected_h shape: [batch_size, hidden_size]
            projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))

            # Repeat the projected state for all decoder layers
            # decoder_hidden shape: (num_layers, batch_size, hidden_size)
            decoder_hidden = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)


        # First input to the decoder is the SOS token
        # target[:, 0] shape: [batch_size]
        # unsqueeze(1) shape: [batch_size, 1]
        decoder_input = target[:, 0].unsqueeze(1)

        # Attention weights to return (if attention is used)
        attention_weights_list = []

        # Decode one step at a time
        # Start from index 1 of the target sequence (after SOS)
        for t in range(1, target_len):
            # decoder_input shape: [batch_size, 1]
            # decoder_hidden shape: (num_layers, batch_size, hidden_size) or ((num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size))
            # encoder_outputs shape: [batch_size, source_len, hidden_size * 2]
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs # Pass encoder_outputs for attention
            )

            # Store output and attention weights
            # decoder_output shape: [batch_size, target_vocab_size]
            outputs[:, t, :] = decoder_output
            if attn_weights is not None:
                # attn_weights shape from BahdanauAttention: [batch_size, source_len]
                attention_weights_list.append(attn_weights)

            # Teacher forcing: use actual target token as next input
            # or use predicted token
            if random.random() < teacher_forcing_ratio:
                # Use ground truth token from target sequence
                # target[:, t] shape: [batch_size]
                # unsqueeze(1) shape: [batch_size, 1]
                decoder_input = target[:, t].unsqueeze(1)
            else:
                # Use model's prediction
                # decoder_output shape: [batch_size, target_vocab_size]
                # topk(1) shape: (values [batch_size, 1], indices [batch_size, 1])
                topv, topi = decoder_output.topk(1)
                # topi shape: [batch_size, 1]
                # detach() to prevent gradients from flowing back through this prediction
                decoder_input = topi.detach()

        # If attention was used, stack the attention weights
        final_attention_weights = torch.stack(attention_weights_list, dim=1) if attention_weights_list else None # Shape: [batch_size, target_len - 1, source_len]

        return outputs, final_attention_weights

    # The translate method for greedy decoding also needs to handle the bidirectional encoder output
    def translate(self, source, source_tokenizer, target_tokenizer, max_length=50, device="cpu"):
        """Translate a single source sequence using greedy decoding."""
        self.eval()  # Set model to evaluation mode

        with torch.no_grad():
            # Encode source sequence
            # source_tensor shape: [1, source_len]
            source_tensor = torch.tensor(source_tokenizer.encode(source), device=device).unsqueeze(0)
            # encoder_outputs shape: [1, source_len, hidden_size * 2]
            # encoder_hidden shape: (num_layers * 2, 1, hidden_size) for GRU/RNN
            #                      ((num_layers * 2, 1, hidden_size), (num_layers * 2, 1, hidden_size)) for LSTM
            encoder_outputs, encoder_hidden = self.encoder(source_tensor)

            # Prepare initial decoder hidden state from bidirectional encoder hidden state
            # Same logic as in forward, but batch_size is 1
            if self.rnn_type == "lstm":
                last_layer_encoder_h = torch.cat((encoder_hidden[0][-2, :, :], encoder_hidden[0][-1, :, :]), dim=1)
                last_layer_encoder_c = torch.cat((encoder_hidden[1][-2, :, :], encoder_hidden[1][-1, :, :]), dim=1)
                projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
                projected_c = torch.tanh(self.encoder_to_decoder_cell(last_layer_encoder_c))
                decoder_hidden_h = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)
                decoder_hidden_c = projected_c.unsqueeze(0).repeat(self.num_layers, 1, 1)
                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)
            else: # GRU or RNN
                last_layer_encoder_h = torch.cat((encoder_hidden[-2, :, :], encoder_hidden[-1, :, :]), dim=1)
                projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
                decoder_hidden = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)


            # Initialize decoder input with SOS token
            # SOS token ID from the target tokenizer
            SOS_token_id = target_tokenizer.sos_id()
            EOS_token_id = target_tokenizer.eos_id()
            PAD_token_id = target_tokenizer.pad_id()

            decoder_input = torch.tensor([[SOS_token_id]], device=device) # shape [1, 1]

            decoded_chars = []
            attention_weights_list = []

            # Decode until max length or EOS token
            for _ in range(max_length):
                # decoder_input shape: [1, 1]
                # decoder_hidden shape: (num_layers, 1, hidden_size) or ((num_layers, 1, hidden_size), (num_layers, 1, hidden_size))
                # encoder_outputs shape: [1, source_len, hidden_size * 2]
                decoder_output, decoder_hidden, attn_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                # Get most likely token (greedy decoding)
                # decoder_output shape: [1, target_vocab_size]
                topv, topi = decoder_output.topk(1)
                token_id = topi.item() # Get the scalar token ID

                # Stop if EOS token is predicted
                if token_id == EOS_token_id:
                    break

                # Add token to output (skip PAD)
                if token_id != PAD_token_id:
                    decoded_chars.append(target_tokenizer.idx_to_char.get(token_id, '')) # Use .get for safety

                # Save attention weights
                if attn_weights is not None:
                     # attn_weights shape from BahdanauAttention: [1, source_len]
                    attention_weights_list.append(attn_weights.squeeze().cpu().numpy()) # Squeeze to [source_len]

                # Next input is the predicted token
                decoder_input = topi.detach() # Use the predicted token as input for the next step

            # If attention was used, stack the attention weights
            final_attention_weights = np.stack(attention_weights_list, axis=0) if attention_weights_list else None # Shape: [decoded_len, source_len]

            return ''.join(decoded_chars), final_attention_weights

# The __main__ block remains the same for testing purposes
import os
import sys
import torch

# Set up relative paths for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from data.tokenizer import CharTokenizer
import traceback

if __name__ == "__main__":
    # Create sample tokenizers
    source_chars = ["\t", "a", "b", "c", "d", "e", "f", "\n"]
    target_chars = ["\t", "x", "y", "z", "w", "\n"]

    source_tokenizer = CharTokenizer(sos_token="\t", eos_token="\n", pad_token="<PAD>")
    source_tokenizer.fit(["".join(source_chars)])

    target_tokenizer = CharTokenizer(sos_token="\t", eos_token="\n", pad_token="<PAD>")
    target_tokenizer.fit(["".join(target_chars)])

    # Model parameters
    source_vocab_size = len(source_tokenizer)
    target_vocab_size = len(target_tokenizer)
    embedding_dim = 16
    hidden_size = 32
    num_layers = 1
    rnn_type = "lstm"
    attention = True

    print(f"Source Vocab Size: {source_vocab_size}")
    print(f"Target Vocab Size: {target_vocab_size}")

    # Create the model
    model = Seq2SeqModel(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.1,
        rnn_type=rnn_type,
        attention=attention
    )

    print("\nModel created successfully:")
    print(model)

    # Dummy inputs for forward pass test
    batch_size = 2
    source_len = 5
    target_len = 7

    dummy_source = torch.randint(1, source_vocab_size, (batch_size, source_len))  # Avoid PAD=0
    dummy_target = torch.randint(1, target_vocab_size, (batch_size, target_len))  # Avoid PAD=0
    dummy_target[:, 0] = target_tokenizer.sos_id()

    print("\nTesting forward pass...")
    try:
        outputs, attn_weights = model(dummy_source, dummy_target, teacher_forcing_ratio=0.5)
        print(f"Outputs shape: {outputs.shape}")
        if attn_weights is not None:
            print(f"Attention weights shape: {attn_weights.shape}")
        else:
            print("Attention weights: None")
    except Exception as e:
        print(f"Error during forward pass test: {e}")
        traceback.print_exc()

    print("\nTesting translate function...")
    source_text_to_translate = "abc"

    try:
        translation, attention_weights_translate = model.translate(
            source=source_text_to_translate,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            device="cpu"
        )
        print(f"Source: {source_text_to_translate}")
        print(f"Translation: {translation}")
        print(f"Attention weights available from translate: {attention_weights_translate is not None}")
        if attention_weights_translate is not None:
            print(f"Attention weights shape from translate: {attention_weights_translate.shape}")
    except Exception as e:
        print(f"Error during translate test: {e}")
        traceback.print_exc()
