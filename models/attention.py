import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention Mechanism to weigh the importance of different time steps.
    Reduces (Batch, Seq_Len, Hidden_Dim) -> (Batch, Hidden_Dim).
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Hidden_Dim)
        
        # Calculate attention scores
        # (Batch, Seq_Len, 1)
        scores = self.attn(x)
        
        # Softmax over the sequence dimension (dim=1)
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum: (Batch, Seq_Len, 1) * (Batch, Seq_Len, Hidden_Dim) -> (Batch, Hidden_Dim)
        # Transpose weights to broadcast properly or use matmul
        
        # weights: (Batch, Seq, 1)
        # x: (Batch, Seq, Hidden)
        # We want to sum over Seq.
        
        context = torch.sum(weights * x, dim=1)
        
        return context, weights
