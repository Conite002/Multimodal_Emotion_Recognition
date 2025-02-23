import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np

class BiGRUWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(BiGRUWithAttention, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=2 * hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        gru_output, _ = self.gru(x)
        attn_output, _ = self.attention(gru_output, gru_output, gru_output)
        x = self.layer_norm(attn_output + gru_output)
        x = self.dropout(x)
        return x

