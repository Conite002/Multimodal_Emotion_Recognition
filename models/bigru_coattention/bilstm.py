import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np



class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(BiLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=2 * hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        x = self.layer_norm(attn_output + lstm_output)
        x = self.dropout(x)
        return x
    
