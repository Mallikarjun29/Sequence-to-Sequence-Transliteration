import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder for sequence-to-sequence model."""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, rnn_type="lstm"):
        """Initialize encoder with specified parameters.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
            hidden_size (int): Size of the hidden state.
            num_layers (int): Number of RNN layers.
            dropout (float): Dropout probability (applied between layers).
            rnn_type (str): Type of RNN cell ("lstm", "gru", or "rnn").
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.num_directions = 2

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
        else:
            self.rnn = nn.RNN(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )

    def forward(self, input_seq, hidden=None):
        """Process input sequence through the encoder.
        
        Args:
            input_seq (Tensor): Input sequence tensor [batch_size, seq_length].
            hidden (Tensor, optional): Initial hidden state.

        Returns:
            tuple: (outputs, hidden)
                outputs: Output features from the RNN [batch_size, seq_length, hidden_size * num_directions].
                hidden: Final hidden state (h_n for GRU/RNN, (h_n, c_n) for LSTM).
                        Shape: (num_layers * num_directions, batch_size, hidden_size).
        """
        embedded = self.embedding(input_seq)
        outputs, hidden = self.rnn(embedded, hidden)
        return outputs, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state for the RNN.
        
        Args:
            batch_size (int): Size of the batch.
            device (torch.device): Device to create the tensor on.
            
        Returns:
            Tensor or tuple: Initial hidden state with appropriate shape for the RNN type.
        """
        if self.rnn_type == "lstm":
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
            return (h0, c0)
        else:
            return torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)

