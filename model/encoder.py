import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Encoder for sequence-to-sequence model."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, rnn_type="lstm"):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                embedding_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
    
    def forward(self, input_seq, hidden=None):
        """
        Args:
            input_seq: Input sequence tensor [batch_size, seq_length]
            hidden: Initial hidden state
        
        Returns:
            outputs: Output features from the last RNN layer
            hidden: Hidden state from the last RNN layer
        """
        # Embed input sequence
        embedded = self.embedding(input_seq)
        
        # Pass through RNN
        outputs, hidden = self.rnn(embedded, hidden)
        
        return outputs, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state."""
        if self.rnn_type == "lstm":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            return (h0, c0)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
