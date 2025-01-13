import torch
import torch.nn as nn
import torch.optim as optim


class TextLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.2):
        """
        LSTM-based classifier for text embeddings.
        Parameters:
        - input_dim: int, the embedding dimension (input size).
        - num_classes: int, the number of output classes.
        - hidden_dim: int, number of LSTM hidden units.
        - num_layers: int, number of LSTM layers.
        - dropout: float, dropout rate.
        """
        super(TextLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """
        Forward pass of the LSTM classifier.
        x: torch.Tensor of shape (batch_size, seq_len, input_dim).
        Returns logits of shape (batch_size, num_classes).
        """
        out, (hn, cn) = self.lstm(x)  # LSTM output
        out = self.fc(hn[-1])  # Use the hidden state from the last LSTM layer
        return out
