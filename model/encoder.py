import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder for sequence-to-sequence model."""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, rnn_type="lstm"):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.num_directions = 2 # Bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN layer - Set bidirectional=True
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True # Make encoder RNN bidirectional
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True # Make encoder RNN bidirectional
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True # Make encoder RNN bidirectional
            )

    def forward(self, input_seq, hidden=None):
        """
        Args:
            input_seq: Input sequence tensor [batch_size, seq_length]
            hidden: Initial hidden state (optional, will be initialized to zeros if None)

        Returns:
            outputs: Output features from the last RNN layer [batch_size, seq_length, hidden_size * num_directions]
            hidden: Final hidden state from the last RNN layer (h_n for GRU/RNN, (h_n, c_n) for LSTM)
                    Shape: (num_layers * num_directions, batch_size, hidden_size)
        """
        # Embed input sequence
        embedded = self.embedding(input_seq)

        # Pass through RNN
        # outputs shape: [batch_size, seq_length, hidden_size * num_directions]
        # hidden shape: (num_layers * num_directions, batch_size, hidden_size) for GRU/RNN
        #               ((num_layers * num_directions, batch_size, hidden_size), (num_layers * num_directions, batch_size, hidden_size)) for LSTM
        outputs, hidden = self.rnn(embedded, hidden)

        return outputs, hidden

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state compatible with bidirectional RNN.
        Shape: (num_layers * num_directions, batch_size, hidden_size)
        """
        if self.rnn_type == "lstm":
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            return (h0, c0)
        else: # GRU or RNN
            return torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

