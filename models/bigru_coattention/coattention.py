import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from models.graph.graph import GraphContructor, GRNN



from models.bigru_coattention.bigru import BiGRUWithAttention
from models.bigru_coattention.bilstm import BiLSTMWithAttention

class CoAttentionFusion(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128, dropout_rate=0.6):
        super(CoAttentionFusion, self).__init__()
        self.audio_projection = nn.Linear(input_dim_audio, 256)
        self.text_projection = nn.Linear(input_dim_text, 256)
        self.video_projection = nn.Linear(input_dim_video, 256)

        self.audio_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.text_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.video_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.co_attention = nn.MultiheadAttention(embed_dim=384, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(384)

        self.fc = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def forward(self, audio, text, video):
        audio_feat = self.audio_projection(audio).unsqueeze(1)
        text_feat = self.text_projection(text).unsqueeze(1) 
        video_feat = self.video_projection(video) 

        audio_feat = self.audio_attention(audio_feat) 
        text_feat = self.text_attention(text_feat)    
        video_feat = self.video_attention(video_feat) 

        video_feat = video_feat.mean(dim=1, keepdim=False)  
        audio_feat = audio_feat.squeeze(1)  
        text_feat = text_feat.squeeze(1)    

        combined = torch.cat([audio_feat, text_feat, video_feat], dim=-1)  

        combined = combined.view(combined.size(0), -1, 384)
        attn_output, _ = self.co_attention(combined, combined, combined)
        x = self.layer_norm(attn_output + combined)  

        x = x.mean(dim=1)
        return self.fc(x) 
    

class CoAttentionFusionWithGraph(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128, dropout_rate=0.6):
        super(CoAttentionFusionWithGraph, self).__init__()
        self.audio_projection = nn.Linear(input_dim_audio, 256)
        self.text_projection = nn.Linear(input_dim_text, 256)
        self.video_projection = nn.Linear(input_dim_video, 256)

        self.audio_attention = BiLSTMWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.text_attention = BiLSTMWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.video_attention = BiLSTMWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.graph_constructor = GraphContructor(input_dim=hidden_dim * 2)
        self.grnn = GRNN(input_dim=hidden_dim, hidden_dim=hidden_dim)

        self.co_attention = nn.MultiheadAttention(embed_dim=384, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(384)

        self.fc = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=-1)
        )

        self.modality_weights = nn.Parameter(torch.ones(3))


    def forward(self, audio, text, video, node_features, adjacency_matrix, edge_index):
        audio_feat = self.audio_projection(audio).unsqueeze(1)
        text_feat = self.text_projection(text).unsqueeze(1)
        video_feat = self.video_projection(video)

        audio_feat = self.audio_attention(audio_feat)
        text_feat = self.text_attention(text_feat)
        video_feat = self.video_attention(video_feat)

        video_feat = video_feat.mean(dim=1, keepdim=False)
        audio_feat = audio_feat.squeeze(1)
        text_feat = text_feat.squeeze(1)

        modality_weights = torch.softmax(self.modality_weights, dim=0)
        combined = torch.cat([
            modality_weights[0] * audio_feat,
            modality_weights[1] * text_feat,
            modality_weights[2] * video_feat
        ], dim=-1)

        graph_features = self.graph_constructor(node_features, adjacency_matrix)
        graph_output = self.grnn(graph_features, edge_index)
        combined = torch.cat([combined, graph_output], dim=-1)
        combined = combined.view(combined.size(0), -1, 384)
        attn_output, _ = self.co_attention(combined, combined, combined)
        x = self.layer_norm(attn_output + combined)

        x = x.mean(dim=1)
        return self.fc(x)


import torch
import torch.nn as nn
from models.bigru_coattention.bigru import BiGRUWithAttention


class CoAttentionFusion2(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128, dropout_rate=0.3):
        super(CoAttentionFusion, self).__init__()

        # Projection layers to unify dimensions
        self.audio_projection = nn.Linear(input_dim_audio, 128)  # Project audio to 128
        self.text_projection = nn.Linear(input_dim_text, 256)   # Project text to 256
        self.video_projection = nn.Linear(input_dim_video, 256) # Project video to 256

        # Attention layers
        self.audio_attention = BiGRUWithAttention(input_dim=128, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.text_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.video_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        # Co-attention mechanism
        self.co_attention = nn.MultiheadAttention(embed_dim=hidden_dim * 3, num_heads=8, batch_first=True)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 3)

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),  # Combined feature size is hidden_dim * 3
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)  # Output size is num_classes
        )

    def forward(self, audio, text, video):
        # Project input features to fixed dimensions
        audio_feat = self.audio_projection(audio)  # [batch_size, seq_len, 128]
        text_feat = self.text_projection(text)     # [batch_size, seq_len, 256]
        video_feat = self.video_projection(video)  # [batch_size, seq_len, 256]

        # Apply BiGRU with attention to each modality
        audio_feat = self.audio_attention(audio_feat)  # [batch_size, hidden_dim]
        text_feat = self.text_attention(text_feat)     # [batch_size, hidden_dim]
        video_feat = self.video_attention(video_feat)  # [batch_size, hidden_dim]

        # Combine features
        combined = torch.cat([audio_feat, text_feat, video_feat], dim=-1)  # [batch_size, hidden_dim * 3]

        # Reshape for multi-head attention
        combined = combined.unsqueeze(1)  # Add a sequence length of 1: [batch_size, 1, hidden_dim * 3]
        attn_output, _ = self.co_attention(combined, combined, combined)  # Self-attention
        x = self.layer_norm(attn_output.squeeze(1) + combined.squeeze(1))  # Residual connection and layer norm

        # Classification
        return self.fc(x)  # [batch_size, num_classes]


import torch
import torch.nn as nn

class CoAttentionFusionReguNorm(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128, dropout_rate=0.5):
        super(CoAttentionFusionReguNorm, self).__init__()

        # Projection layers for each modality
        self.audio_projection = nn.Linear(input_dim_audio, 256)
        self.text_projection = nn.Linear(input_dim_text, 256)
        self.video_projection = nn.Linear(input_dim_video, 256)

        # BiGRU with attention for each modality
        self.audio_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.text_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.video_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        # Multihead attention for cross-modality fusion
        self.co_attention = nn.MultiheadAttention(embed_dim=384, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(384)

        # Fully connected layers with additional regularization and normalization
        self.fc = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),  # Add BatchNorm for better generalization
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Regularization to prevent overfitting
        self.audio_dropout = nn.Dropout(dropout_rate)
        self.text_dropout = nn.Dropout(dropout_rate)
        self.video_dropout = nn.Dropout(dropout_rate)

    def forward(self, audio, text, video):
        # Projection
        audio_feat = self.audio_projection(audio)
        text_feat = self.text_projection(text)
        video_feat = self.video_projection(video)

        # Dropout for modality-specific features
        audio_feat = self.audio_dropout(audio_feat).unsqueeze(1)
        text_feat = self.text_dropout(text_feat).unsqueeze(1)
        video_feat = self.video_dropout(video_feat)

        # Attention-based feature extraction
        audio_feat = self.audio_attention(audio_feat)
        text_feat = self.text_attention(text_feat)
        video_feat = self.video_attention(video_feat)

        # Global average pooling for video features
        video_feat = video_feat.mean(dim=1, keepdim=False)  # (batch_size, hidden_dim)
        audio_feat = audio_feat.squeeze(1)  # (batch_size, hidden_dim)
        text_feat = text_feat.squeeze(1)  # (batch_size, hidden_dim)

        # Combine features from all modalities
        combined = torch.cat([audio_feat, text_feat, video_feat], dim=-1)  # (batch_size, 384)

        # Multihead attention for cross-modal fusion
        combined = combined.view(combined.size(0), -1, 384)
        attn_output, _ = self.co_attention(combined, combined, combined)
        x = self.layer_norm(attn_output + combined)  # Residual connection

        # Mean pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, 384)

        # Fully connected layers
        return self.fc(x)
    
