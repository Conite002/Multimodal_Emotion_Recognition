import torch
import torch.nn as nn


# class VideoMLPClassifier(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         """
#         MLP-based classifier for video embeddings.
#         Parameters:
#         - input_dim: int, number of input features (e.g., 768 for ViT embeddings).
#         - num_classes: int, number of output classes.
#         """
#         super(VideoMLPClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.fc2 = nn.Linear(512, 128)
#         self.fc3 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         print(f"Input shape to forward: {x.shape}") 
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         print(f"Output shape from forward: {x.shape}")  # Debugging
#         return x

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1) 

    def forward(self, x):
        attention_scores = self.attention_weights(x)  
        attention_weights = torch.softmax(attention_scores, dim=1)
        aggregated = (x * attention_weights).sum(dim=1)
        return aggregated  
    
class VideoMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(VideoMLPClassifier, self).__init__()
        self.attention_pooling = AttentionPooling(input_dim) 
        self.fc1 = nn.Linear(input_dim, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
