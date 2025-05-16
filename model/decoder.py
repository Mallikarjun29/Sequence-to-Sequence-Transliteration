import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import BahdanauAttention

class Decoder(nn.Module):
    """Decoder for sequence-to-sequence model."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, 
                 dropout, rnn_type="lstm", attention=False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.rnn_type = rnn_type.lower()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Attention mechanism
        if attention:
            self.attention_layer = BahdanauAttention(hidden_size)
            rnn_input_dim = embedding_dim + hidden_size
        else:
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
            encoder_outputs: Encoder outputs for attention [batch_size, seq_length, hidden_size]
        
        Returns:
            output: Output probabilities for next token [batch_size, vocab_size]
            hidden: Updated hidden state
            attention_weights: Attention weights if using attention
        """
        # Embed input
        embedded = self.embedding(input_step)
        
        # Apply attention if enabled
        attention_weights = None
        if self.attention and encoder_outputs is not None:
            # Get context vector from attention
            if self.rnn_type == "lstm":
                context, attention_weights = self.attention_layer(hidden[0][-1], encoder_outputs)
            else:
                context, attention_weights = self.attention_layer(hidden[-1], encoder_outputs)
                
            # Combine embedding and context vector
            rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        else:
            rnn_input = embedded
        
        # Pass through RNN
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Get output probabilities
        output = self.output_layer(output.squeeze(1))
        
        return output, hidden, attention_weights
