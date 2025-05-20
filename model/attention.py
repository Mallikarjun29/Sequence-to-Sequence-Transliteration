import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism."""

    def __init__(self, decoder_hidden_size, encoder_output_size):
        """Initialize Bahdanau attention mechanism.
        
        Args:
            decoder_hidden_size (int): Hidden size of the decoder RNN.
            encoder_output_size (int): Feature size of the encoder outputs (hidden_size * num_directions).
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.W2 = nn.Linear(encoder_output_size, decoder_hidden_size)
        self.V = nn.Linear(decoder_hidden_size, 1)

    def forward(self, query, keys):
        """Calculate attention context vector and weights.
        
        Args:
            query (Tensor): Decoder hidden state [batch_size, decoder_hidden_size].
            keys (Tensor): Encoder outputs [batch_size, seq_length, encoder_output_size].

        Returns:
            tuple: (context_vector, attention_weights)
                context_vector (Tensor): Context vector [batch_size, encoder_output_size].
                attention_weights (Tensor): Attention weights [batch_size, seq_length].
        """
        query = query.unsqueeze(1)
        scores = self.V(torch.tanh(self.W1(query) + self.W2(keys)))
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        
        return context_vector, attention_weights

