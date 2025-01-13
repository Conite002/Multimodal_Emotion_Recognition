import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
from models.bigru_coattention.bigru import BiGRUWithAttention

class CoAttentionFusion(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128, dropout_rate=0.3):
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
    