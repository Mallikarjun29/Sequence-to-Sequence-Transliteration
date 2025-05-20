import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import BahdanauAttention

class Decoder(nn.Module):
    """Decoder for sequence-to-sequence model."""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
                 dropout, rnn_type="lstm", attention=False, encoder_output_size=None):
        """Initialize decoder with specified parameters.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
            hidden_size (int): Size of the hidden state.
            num_layers (int): Number of RNN layers.
            dropout (float): Dropout probability (applied between layers).
            rnn_type (str): Type of RNN cell ("lstm", "gru", or "rnn").
            attention (bool): Whether to use attention mechanism.
            encoder_output_size (int, optional): Size of encoder outputs (required if attention is True).
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.rnn_type = rnn_type.lower()
        self.encoder_output_size = encoder_output_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        attention_hidden_size = hidden_size
        if attention:
            if encoder_output_size is None:
                 raise ValueError("encoder_output_size must be provided if attention is True")
            self.attention_layer = BahdanauAttention(hidden_size, encoder_output_size)
            rnn_input_dim = embedding_dim + encoder_output_size
        else:
            rnn_input_dim = embedding_dim

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                rnn_input_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                rnn_input_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        else:
            self.rnn = nn.RNN(
                rnn_input_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )

        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_step, hidden, encoder_outputs=None):
        """Process one decoder step.
        
        Args:
            input_step (Tensor): Input for current time step [batch_size, 1].
            hidden (Tensor or tuple): Hidden state from previous time step.
                    Shape varies based on RNN type.
            encoder_outputs (Tensor, optional): Encoder outputs for attention [batch_size, seq_length, encoder_output_size].

        Returns:
            tuple: (output, hidden, attention_weights)
                output: Output probabilities [batch_size, vocab_size].
                hidden: Updated hidden state.
                attention_weights: Attention weights if using attention, else None.
        """
        embedded = self.embedding(input_step)

        attention_weights = None
        if self.attention and encoder_outputs is not None:
            if self.rnn_type == "lstm":
                context, attention_weights = self.attention_layer(hidden[0][-1], encoder_outputs)
            else:
                context, attention_weights = self.attention_layer(hidden[-1], encoder_outputs)

            rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        else:
            rnn_input = embedded

        output, new_hidden = self.rnn(rnn_input, hidden)
        output = self.output_layer(output.squeeze(1))

        return output, new_hidden, attention_weights

