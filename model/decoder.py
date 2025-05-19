import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import BahdanauAttention

class Decoder(nn.Module):
    """Decoder for sequence-to-sequence model."""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers,
                 dropout, rnn_type="lstm", attention=False, encoder_output_size=None): # Added encoder_output_size
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.rnn_type = rnn_type.lower()
        self.encoder_output_size = encoder_output_size # Store encoder output size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Attention mechanism
        attention_hidden_size = hidden_size # Attention context vector size will be encoder_output_size
        if attention:
            if encoder_output_size is None:
                 raise ValueError("encoder_output_size must be provided if attention is True")
            # Pass decoder_hidden_size and encoder_output_size to attention layer
            self.attention_layer = BahdanauAttention(hidden_size, encoder_output_size)
            # RNN input dimension is embedding_dim + size of the context vector (which is encoder_output_size)
            rnn_input_dim = embedding_dim + encoder_output_size
        else:
            rnn_input_dim = embedding_dim

        # RNN layer (Decoder is typically unidirectional)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                rnn_input_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False # Decoder is typically unidirectional
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                rnn_input_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False # Decoder is typically unidirectional
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                rnn_input_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False # Decoder is typically unidirectional
            )

        # Output layer
        # The output layer takes the decoder's hidden state (hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_step, hidden, encoder_outputs=None):
        """
        Args:
            input_step: Input for current time step [batch_size, 1]
            hidden: Hidden state from previous time step (h_t-1)
                    Shape: (num_layers, batch_size, hidden_size) for GRU/RNN
                           ((num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size)) for LSTM
            encoder_outputs: Encoder outputs for attention [batch_size, seq_length, encoder_output_size]

        Returns:
            output: Output probabilities for next token [batch_size, vocab_size]
            hidden: Updated hidden state (h_t)
            attention_weights: Attention weights [batch_size, seq_length] if using attention, else None
        """
        # Embed input
        # embedded shape: [batch_size, 1, embedding_dim]
        embedded = self.embedding(input_step)

        # Apply attention if enabled
        attention_weights = None
        if self.attention and encoder_outputs is not None:
            # Get context vector from attention
            # Attention query is the decoder's hidden state.
            # For multi-layer RNNs, typically the hidden state of the last layer is used.
            if self.rnn_type == "lstm":
                # For LSTM, hidden is a tuple (h, c). Use h[-1] for the last layer's hidden state.
                context, attention_weights = self.attention_layer(hidden[0][-1], encoder_outputs)
            else: # GRU or RNN
                # For GRU/RNN, hidden is a single tensor. Use hidden[-1] for the last layer's hidden state.
                context, attention_weights = self.attention_layer(hidden[-1], encoder_outputs)

            # Combine embedding and context vector
            # embedded shape: [batch_size, 1, embedding_dim]
            # context shape: [batch_size, encoder_output_size] -> unsqueeze(1) -> [batch_size, 1, encoder_output_size]
            # rnn_input shape: [batch_size, 1, embedding_dim + encoder_output_size]
            rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        else:
            # rnn_input shape: [batch_size, 1, embedding_dim]
            rnn_input = embedded

        # Pass through RNN
        # rnn_input shape: [batch_size, 1, rnn_input_dim]
        # hidden shape: (num_layers, batch_size, hidden_size) or ((num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size))
        # output shape: [batch_size, 1, hidden_size] (since seq_len is 1)
        # new_hidden shape: (num_layers, batch_size, hidden_size) or ((num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size))
        output, new_hidden = self.rnn(rnn_input, hidden)

        # Get output probabilities
        # output shape [batch_size, 1, hidden_size] -> squeeze(1) -> [batch_size, hidden_size]
        # output_layer output shape: [batch_size, vocab_size]
        output = self.output_layer(output.squeeze(1))

        return output, new_hidden, attention_weights

