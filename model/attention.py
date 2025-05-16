import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism."""
    
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        
    def forward(self, query, keys):
        """
        Args:
            query: Decoder hidden state [batch_size, hidden_size]
            keys: Encoder outputs [batch_size, seq_length, hidden_size]
        
        Returns:
            context_vector: Context vector [batch_size, hidden_size]
            attention_weights: Attention weights [batch_size, seq_length]
        """
        # Expand query to [batch_size, 1, hidden_size]
        query = query.unsqueeze(1)
        
        # Calculate scores
        scores = self.V(torch.tanh(self.W1(query) + self.W2(keys)))
        
        # Get attention weights with softmax
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)
        
        # Get context vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        
        return context_vector, attention_weights
