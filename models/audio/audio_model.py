import torch
import torch.nn as nn


class AudioCNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AudioCNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        dummy_input = torch.zeros(1, 1, input_dim)
        self.flattened_size = self._get_flattened_size(dummy_input)
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_flattened_size(self, dummy_input):
        x = self.relu(self.bn1(self.conv1(dummy_input)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x.view(1, -1).size(1)