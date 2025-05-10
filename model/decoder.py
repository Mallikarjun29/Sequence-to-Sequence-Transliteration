import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """Decoder for sequence-to-sequence model."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, 
                 dropout, rnn_type="lstm"):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        rnn_input_dim = embedding_dim
            
        # RNN layer
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                rnn_input_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                rnn_input_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:  # default to RNN
            self.rnn = nn.RNN(
                rnn_input_dim, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_step, hidden, encoder_outputs=None):
        """
        Args:
            input_step: Input for current time step [batch_size, 1]
            hidden: Hidden state from previous time step
        
        Returns:
            output: Output probabilities for next token [batch_size, vocab_size]
            hidden: Updated hidden state
        """
        # Embed input
        embedded = self.embedding(input_step)
        
        rnn_input = embedded
        
        # Pass through RNN
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Get output probabilities
        output = self.output_layer(output.squeeze(1))
        
        return output, hidden