import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from models.graph.graph import  GRNN, GraphConstructor



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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.6), 
            nn.Linear(64, num_classes)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(128, num_classes),
        # )
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
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128, dropout_rate=0.6, num_speakers=100):
        super(CoAttentionFusionWithGraph, self).__init__()

        self.audio_projection = nn.Linear(input_dim_audio, 256)
        self.text_projection = nn.Linear(input_dim_text, 256)
        self.video_projection = nn.Linear(input_dim_video, 256)

        self.audio_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.text_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.video_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.graph_constructor = GraphConstructor(input_dim=2304, hidden_dim=hidden_dim)
        self.grnn = GRNN(input_dim=hidden_dim, hidden_dim=hidden_dim)

        self.co_attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(128)

        self.fc = nn.Sequential(
            nn.Linear(128, num_classes)
        )


        # self.attn_projection = nn.Linear(896, 384)
        self.attn_projection = nn.Linear(768, 384)  
    def forward(self, audio, text, video, node_features, edge_index, batch_speaker_ids):
        device = audio.device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)

        audio_feat = self.audio_projection(audio).unsqueeze(1)
        text_feat = self.text_projection(text).unsqueeze(1)
        video_feat = self.video_projection(video)

        audio_feat = self.audio_attention(audio_feat).squeeze(1)
        text_feat = self.text_attention(text_feat).squeeze(1)
        video_feat = self.video_attention(video_feat).mean(dim=1, keepdim=False)
        
        combined = torch.cat([ audio_feat, text_feat, video_feat], dim=-1)
        graph_features = self.graph_constructor(node_features, edge_index)
        graph_output = self.grnn(graph_features, edge_index)

        graph_output = (graph_output - graph_output.mean()) / graph_output.std()
        graph_output = graph_output * 0.05
        
        if batch_speaker_ids.max() >= graph_output.shape[0]:
            raise IndexError(f"batch_speaker_ids contient un indice hors limite ! Max: {graph_output.shape[0]-1}, Trouvé: {batch_speaker_ids.max()}")
        batch_graph_features = graph_output[batch_speaker_ids]
        # combined = torch.cat([combined, batch_graph_features], dim=-1)  
        # combined = self.attn_projection(combined)
        combined = combined.view(combined.size(0), -1, 128)
        # print(f"Combined: {combined.shape}") # Combined: torch.Size([32, 2, 128])
        # print(f" batch_graph_features: {batch_graph_features.shape}") #   torch.Size([32, 128])
        batch_graph_features = batch_graph_features.unsqueeze(1)
        
        combined = torch.cat([combined, batch_graph_features], dim=1)
        # combine features from all modalities and graph features (combined batch_graph_features)
        attn_output, _ = self.co_attention(combined, combined, combined)
        x = self.layer_norm(attn_output + combined)
        x = x.mean(dim=1)
        x = self.fc(x) 
        return x




import torch
import torch.nn as nn
from models.bigru_coattention.bigru import BiGRUWithAttention


class CoAttentionFusion2(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128, dropout_rate=0.3):
        super(CoAttentionFusion, self).__init__()

        self.audio_projection = nn.Linear(input_dim_audio, 128) 
        self.text_projection = nn.Linear(input_dim_text, 256)  
        self.video_projection = nn.Linear(input_dim_video, 256) 

        self.audio_attention = BiGRUWithAttention(input_dim=128, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.text_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.video_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.co_attention = nn.MultiheadAttention(embed_dim=hidden_dim * 3, num_heads=4, batch_first=True)

        self.layer_norm = nn.LayerNorm(hidden_dim * 3)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes) 
        )

    def forward(self, audio, text, video):
        # Project input features to fixed dimensions
        audio_feat = self.audio_projection(audio)   
        text_feat = self.text_projection(text)      
        video_feat = self.video_projection(video)   

        audio_feat = self.audio_attention(audio_feat)  
        text_feat = self.text_attention(text_feat)     
        video_feat = self.video_attention(video_feat)  

        combined = torch.cat([audio_feat, text_feat, video_feat], dim=-1)

        combined = combined.unsqueeze(1) 
        attn_output, _ = self.co_attention(combined, combined, combined)  
        x = self.layer_norm(attn_output.squeeze(1) + combined.squeeze(1))  

        return self.fc(x)  


import torch
import torch.nn as nn

class CoAttentionFusionReguNorm(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128, dropout_rate=0.5):
        super(CoAttentionFusionReguNorm, self).__init__()

        self.audio_projection = nn.Linear(input_dim_audio, 256)
        self.text_projection = nn.Linear(input_dim_text, 256)
        self.video_projection = nn.Linear(input_dim_video, 256)

        self.audio_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.text_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.video_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.co_attention = nn.MultiheadAttention(embed_dim=384, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(384)

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
        combined = torch.cat([audio_feat, text_feat, video_feat], dim=-1) 
        combined = combined.view(combined.size(0), -1, 384)
        attn_output, _ = self.co_attention(combined, combined, combined)
        x = self.layer_norm(attn_output + combined)
        x = x.mean(dim=1)  
        return self.fc(x)
    
