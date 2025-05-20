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
        """Initialize sequence-to-sequence model.
        
        Args:
            source_vocab_size (int): Size of the source vocabulary.
            target_vocab_size (int): Size of the target vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
            hidden_size (int): Size of the hidden state for unidirectional RNNs.
            num_layers (int): Number of RNN layers in encoder and decoder.
            dropout (float): Dropout probability for RNN layers.
            rnn_type (str): Type of RNN cell ("lstm", "gru", or "rnn").
            attention (bool): Whether to use attention mechanism.
        """
        super(Seq2SeqModel, self).__init__()

        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.attention = attention
        self.target_vocab_size = target_vocab_size

        self.encoder = Encoder(
            source_vocab_size, embedding_dim, hidden_size,
            num_layers, dropout, rnn_type
        )

        encoder_output_size = hidden_size * 2
        self.decoder = Decoder(
            target_vocab_size, embedding_dim, hidden_size,
            num_layers, dropout, rnn_type, attention,
            encoder_output_size=encoder_output_size if attention else None
        )

        self.encoder_to_decoder_hidden = nn.Linear(hidden_size * 2, hidden_size)

        if self.rnn_type == "lstm":
             self.encoder_to_decoder_cell = nn.Linear(hidden_size * 2, hidden_size)


    def forward(self, source, target, teacher_forcing_ratio=1.0):
        """Process a batch through the sequence-to-sequence model.
        
        Args:
            source (Tensor): Source sequence [batch_size, source_len].
            target (Tensor): Target sequence [batch_size, target_len].
            teacher_forcing_ratio (float): Probability of using teacher forcing.

        Returns:
            tuple: (outputs, attention_weights)
                outputs: Decoder outputs [batch_size, target_len, target_vocab_size].
                attention_weights: Attention weights if using attention [batch_size, target_len-1, source_len].
        """
        batch_size = source.shape[0]
        target_len = target.shape[1]
        device = source.device

        outputs = torch.zeros(batch_size, target_len, self.target_vocab_size).to(device)

        encoder_outputs, encoder_hidden = self.encoder(source)

        if self.rnn_type == "lstm":
            last_layer_encoder_h = torch.cat((encoder_hidden[0][-2, :, :], encoder_hidden[0][-1, :, :]), dim=1)
            last_layer_encoder_c = torch.cat((encoder_hidden[1][-2, :, :], encoder_hidden[1][-1, :, :]), dim=1)

            projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
            projected_c = torch.tanh(self.encoder_to_decoder_cell(last_layer_encoder_c))

            decoder_hidden_h = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)
            decoder_hidden_c = projected_c.unsqueeze(0).repeat(self.num_layers, 1, 1)

            decoder_hidden = (decoder_hidden_h, decoder_hidden_c)

        else:
            last_layer_encoder_h = torch.cat((encoder_hidden[-2, :, :], encoder_hidden[-1, :, :]), dim=1)
            projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
            decoder_hidden = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)

        decoder_input = target[:, 0].unsqueeze(1)

        attention_weights_list = []

        for t in range(1, target_len):
            decoder_output, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            outputs[:, t, :] = decoder_output
            if attn_weights is not None:
                attention_weights_list.append(attn_weights)

            if random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.detach()

        final_attention_weights = torch.stack(attention_weights_list, dim=1) if attention_weights_list else None

        return outputs, final_attention_weights

    def translate(self, source, source_tokenizer, target_tokenizer, max_length=50, device="cpu"):
        """Translate a single source sequence using greedy decoding.
        
        Args:
            source (str): Source text to translate.
            source_tokenizer (CharTokenizer): Tokenizer for source language.
            target_tokenizer (CharTokenizer): Tokenizer for target language.
            max_length (int): Maximum length of generated sequence.
            device (str): Device to run inference on.
            
        Returns:
            tuple: (translation, attention_weights)
                translation (str): Translated text.
                attention_weights (ndarray): Attention weights if attention was used.
        """
        self.eval()

        with torch.no_grad():
            source_tensor = torch.tensor(source_tokenizer.encode(source), device=device).unsqueeze(0)
            encoder_outputs, encoder_hidden = self.encoder(source_tensor)

            if self.rnn_type == "lstm":
                last_layer_encoder_h = torch.cat((encoder_hidden[0][-2, :, :], encoder_hidden[0][-1, :, :]), dim=1)
                last_layer_encoder_c = torch.cat((encoder_hidden[1][-2, :, :], encoder_hidden[1][-1, :, :]), dim=1)
                projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
                projected_c = torch.tanh(self.encoder_to_decoder_cell(last_layer_encoder_c))
                decoder_hidden_h = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)
                decoder_hidden_c = projected_c.unsqueeze(0).repeat(self.num_layers, 1, 1)
                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)
            else:
                last_layer_encoder_h = torch.cat((encoder_hidden[-2, :, :], encoder_hidden[-1, :, :]), dim=1)
                projected_h = torch.tanh(self.encoder_to_decoder_hidden(last_layer_encoder_h))
                decoder_hidden = projected_h.unsqueeze(0).repeat(self.num_layers, 1, 1)

            SOS_token_id = target_tokenizer.sos_id()
            EOS_token_id = target_tokenizer.eos_id()
            PAD_token_id = target_tokenizer.pad_id()

            decoder_input = torch.tensor([[SOS_token_id]], device=device)

            decoded_chars = []
            attention_weights_list = []

            for _ in range(max_length):
                decoder_output, decoder_hidden, attn_weights = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )

                topv, topi = decoder_output.topk(1)
                token_id = topi.item()

                if token_id == EOS_token_id:
                    break

                if token_id != PAD_token_id:
                    decoded_chars.append(target_tokenizer.idx_to_char.get(token_id, ''))

                if attn_weights is not None:
                    attention_weights_list.append(attn_weights.squeeze().cpu().numpy())

                decoder_input = topi.detach()

            final_attention_weights = np.stack(attention_weights_list, axis=0) if attention_weights_list else None

            return ''.join(decoded_chars), final_attention_weights


import os
import sys
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from data.tokenizer import CharTokenizer
import traceback

if __name__ == "__main__":
    source_chars = ["\t", "a", "b", "c", "d", "e", "f", "\n"]
    target_chars = ["\t", "x", "y", "z", "w", "\n"]

    source_tokenizer = CharTokenizer(sos_token="\t", eos_token="\n", pad_token="<PAD>")
    source_tokenizer.fit(["".join(source_chars)])

    target_tokenizer = CharTokenizer(sos_token="\t", eos_token="\n", pad_token="<PAD>")
    target_tokenizer.fit(["".join(target_chars)])

    source_vocab_size = len(source_tokenizer)
    target_vocab_size = len(target_tokenizer)
    embedding_dim = 16
    hidden_size = 32
    num_layers = 1
    rnn_type = "lstm"
    attention = True

    print(f"Source Vocab Size: {source_vocab_size}")
    print(f"Target Vocab Size: {target_vocab_size}")

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

    batch_size = 2
    source_len = 5
    target_len = 7

    dummy_source = torch.randint(1, source_vocab_size, (batch_size, source_len))
    dummy_target = torch.randint(1, target_vocab_size, (batch_size, target_len))
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
