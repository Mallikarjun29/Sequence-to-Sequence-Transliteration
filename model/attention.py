import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism."""

    def __init__(self, decoder_hidden_size, encoder_output_size):
        """
        Args:
            decoder_hidden_size: Hidden size of the decoder RNN.
            encoder_output_size: Feature size of the encoder outputs (hidden_size * num_directions).
        """
        super(BahdanauAttention, self).__init__()
        # W1 takes the decoder hidden state
        self.W1 = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        # W2 takes the encoder outputs (keys) - needs to match encoder_output_size
        self.W2 = nn.Linear(encoder_output_size, decoder_hidden_size)
        # V projects to a single score
        self.V = nn.Linear(decoder_hidden_size, 1)

    def forward(self, query, keys):
        """
        Args:
            query: Decoder hidden state [batch_size, decoder_hidden_size]
            keys: Encoder outputs [batch_size, seq_length, encoder_output_size]

        Returns:
            context_vector: Context vector [batch_size, encoder_output_size] # Context vector is sum of encoder outputs
            attention_weights: Attention weights [batch_size, seq_length]
        """
        # Expand query to [batch_size, 1, decoder_hidden_size]
        query = query.unsqueeze(1)

        # Calculate scores
        # W1(query) shape: [batch_size, 1, decoder_hidden_size]
        # W2(keys) shape: [batch_size, seq_length, decoder_hidden_size]
        # scores shape: [batch_size, seq_length, 1]
        scores = self.V(torch.tanh(self.W1(query) + self.W2(keys)))

        # Get attention weights with softmax
        # attention_weights shape: [batch_size, seq_length]
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)

        # Get context vector
        # attention_weights.unsqueeze(1) shape: [batch_size, 1, seq_length]
        # keys shape: [batch_size, seq_length, encoder_output_size]
        # context_vector shape: [batch_size, 1, encoder_output_size] -> squeeze(1) -> [batch_size, encoder_output_size]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)

        return context_vector, attention_weights

