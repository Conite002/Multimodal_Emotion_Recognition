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
    


import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, RGCNConv, global_add_pool

class CoAttentionFusionWithGraph(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128, dropout_rate=0.6, num_speakers=100):
        super(CoAttentionFusionWithGraph, self).__init__()

        # **Projections Initiales**
        self.audio_projection = nn.Linear(input_dim_audio, 256)
        self.text_projection = nn.Linear(input_dim_text, 256)
        self.video_projection = nn.Linear(input_dim_video, 256)

        # **LSTM pour Chaque Modalité**
        self.audio_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.text_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.video_attention = BiGRUWithAttention(input_dim=256, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.audio_lstm = nn.LSTM(256, hidden_dim, bidirectional=True, batch_first=True)
        self.text_lstm = nn.LSTM(256, hidden_dim, bidirectional=True, batch_first=True)
        self.video_lstm = nn.LSTM(256, hidden_dim, bidirectional=True, batch_first=True)

        # **Graph Construction et Propagation**
        self.graph_constructor = GraphConstructor(input_dim=2304, hidden_dim=hidden_dim)
        self.grnn = GRNN(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # **Fusion Multimodale**
        self.fusion_projection = nn.Linear(896, 256)  # 128 * 3 (modalités) + 128 (graphe)
        self.global_bilstm = nn.LSTM(256, hidden_dim, bidirectional=True, batch_first=True)

        # **Co-Attention**
        self.co_attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # **Classification Finale**
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, num_classes),
        )
        self.modality_weights = nn.Parameter(torch.ones(3))

    def forward(self, audio, text, video, node_features, edge_index, edge_type, batch_speaker_ids):

        device = audio.device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)

        # **1️⃣ Projection des features**
        audio_feat = self.audio_projection(audio).unsqueeze(1)
        text_feat = self.text_projection(text).unsqueeze(1)
        video_feat = self.video_projection(video)

        # **2️⃣ Extraction avec BiGRU**
        audio_feat = self.audio_attention(audio_feat).squeeze(1)
        text_feat = self.text_attention(text_feat).squeeze(1)
        video_feat = self.video_attention(video_feat).mean(dim=1, keepdim=False)

        # **3️⃣ Appliquer BiLSTM Individuel**
        audio_feat, _ = self.audio_lstm(audio_feat.unsqueeze(1))
        text_feat, _ = self.text_lstm(text_feat.unsqueeze(1))
        video_feat, _ = self.video_lstm(video_feat.unsqueeze(1))

        audio_feat = audio_feat.mean(dim=1)
        text_feat = text_feat.mean(dim=1)
        video_feat = video_feat.mean(dim=1)

        # **4️⃣ Propagation Graphique**
        graph_features = self.graph_constructor(node_features, edge_index)
        graph_output = self.grnn(graph_features, edge_index, edge_type)

        graph_output = (graph_output - graph_output.mean()) / graph_output.std()
        graph_output = graph_output * 0.05  # Augmenter l'impact du graphe

        # Sélection des features des speakers
        if batch_speaker_ids.max() >= graph_output.shape[0]:
            raise IndexError(f"batch_speaker_ids contient un indice hors limite ! Max: {graph_output.shape[0]-1}, Trouvé: {batch_speaker_ids.max()}")

        batch_graph_features = graph_output[batch_speaker_ids]

        # **5️⃣ Fusion Multimodale**
        combined = torch.cat([audio_feat, text_feat, video_feat, batch_graph_features], dim=-1)
        combined = self.fusion_projection(combined)
        combined = combined.unsqueeze(1)

        # **6️⃣ Appliquer `BiLSTM` Global**
        combined, _ = self.global_bilstm(combined)
        combined = combined.mean(dim=1)

        # **7️⃣ Co-Attention sur la Fusion**
        attn_output, _ = self.co_attention(combined.unsqueeze(1), combined.unsqueeze(1), combined.unsqueeze(1))
        combined = self.layer_norm(attn_output.squeeze(1) + combined)

        # **8️⃣ Classification Finale**
        x = self.fc(combined)

        return x

# --------------------------------------------------------------------------------------------------------------------------
class CoAttentionFusion_Baseline(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128):
        super().__init__()

        self.audio_attention = BiGRUWithAttention(input_dim=input_dim_audio, hidden_dim=hidden_dim)
        self.text_attention = BiGRUWithAttention(input_dim=input_dim_text, hidden_dim=hidden_dim)
        self.video_attention = BiGRUWithAttention(input_dim=input_dim_video, hidden_dim=hidden_dim)

        self.graph_constructor = GraphConstructor(input_dim=2304, hidden_dim=hidden_dim)
        self.grnn = GRNN(input_dim=hidden_dim, hidden_dim=hidden_dim)

        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        self.projection_layer = nn.Linear(896, 512)  # Réduit 896 → 512 avant FC

    def forward(self, audio, text, video, node_features, edge_index, edge_type, batch_speaker_ids):
        # **1️⃣ Passage des Modalités dans BiGRU**
        audio_feat = self.audio_attention(audio)  # (batch_size, 256)
        text_feat = self.text_attention(text)  # (batch_size, 256)
        video_feat = self.video_attention(video)  # (batch_size, seq_len, 256)

        # **2️⃣ Correction : Moyenne Temporelle pour `video_feat`**
        video_feat = video_feat.mean(dim=1)  # (batch_size, 256)

        # **3️⃣ Propagation Graphique**
        graph_features = self.graph_constructor(node_features, edge_index)
        graph_output = self.grnn(graph_features, edge_index, edge_type)  # (260, 128)

        # **4️⃣ Correction : Sélection des Features Graphiques pour le Batch**
        if batch_speaker_ids.max() >= graph_output.shape[0]:
            raise IndexError(f"batch_speaker_ids contient un indice hors limite ! Max: {graph_output.shape[0]-1}, Trouvé: {batch_speaker_ids.max()}")

        batch_graph_features = graph_output[batch_speaker_ids]  # (batch_size, 128)

        # **6️⃣ Fusion Multimodale**
        combined = torch.cat([audio_feat, text_feat, video_feat, batch_graph_features], dim=-1)  # (batch_size, 896)
        combined = self.projection_layer(combined)  # (batch_size, 512)

        # **7️⃣ Classification Finale**
        x = self.fc(combined)
        return x


# --------------------------------------------------------------------------------------------------------------------------
class CoAttentionFusion_SelfAttn(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128):
        super().__init__()

        self.audio_attention = BiGRUWithAttention(input_dim=input_dim_audio, hidden_dim=hidden_dim)
        self.text_attention = BiGRUWithAttention(input_dim=input_dim_text, hidden_dim=hidden_dim)
        self.video_attention = BiGRUWithAttention(input_dim=input_dim_video, hidden_dim=hidden_dim)

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        self.projection_layer = nn.Linear(768, 512)
    def forward(self, audio, text, video, node_features, edge_index, edge_type, batch_speaker_ids):
        device = audio.device  # Récupère automatiquement le device utilisé
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        batch_speaker_ids = batch_speaker_ids.to(device)  # S'assurer que `batch_speaker_ids` est bien sur le même device

        # **1️⃣ Passage des Modalités dans BiGRU**
        audio_feat = self.audio_attention(audio.to(device))
        text_feat = self.text_attention(text.to(device))
        video_feat = self.video_attention(video.to(device))
        # **2️⃣ Passage dans Self-Attention**
        audio_feat, _ = self.self_attn(audio_feat, audio_feat, audio_feat)
        text_feat, _ = self.self_attn(text_feat, text_feat, text_feat)
        video_feat, _ = self.self_attn(video_feat, video_feat, video_feat)
        video_feat = video_feat.mean(dim=1)

        # **3️⃣ Fusion des Modalités**
        combined = torch.cat([audio_feat, text_feat, video_feat], dim=-1)
        combined = self.projection_layer(combined)
        # **4️⃣ Passage au Fully Connected (FC)**
        x = self.fc(combined.to(device))

        return x
# --------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, RGCNConv, global_add_pool

class CoAttentionFusion_SelfAttn_GNN(nn.Module): 
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128):
        super().__init__()

        # **1️⃣ BiGRU avec Self-Attention pour Chaque Modalité**
        self.audio_attention = BiGRUWithAttention(input_dim=input_dim_audio, hidden_dim=hidden_dim)
        self.text_attention = BiGRUWithAttention(input_dim=input_dim_text, hidden_dim=hidden_dim)
        self.video_attention = BiGRUWithAttention(input_dim=input_dim_video, hidden_dim=hidden_dim)

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)

        # **2️⃣ Projection pour Réduire la Dimension de la Fusion Initiale**
        self.projection_layer = nn.Linear(768, hidden_dim)

        self.graph_constructor = GraphConstructor(input_dim=2304, hidden_dim=hidden_dim)
        self.grnn = GRNN(input_dim=hidden_dim, hidden_dim=hidden_dim)

        self.final_projection = nn.Linear(hidden_dim * 2, hidden_dim)  # Fusion finale GNN + multimodal

        # **5️⃣ Classification Finale**
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio, text, video, node_features, edge_index, edge_type, batch_speaker_ids):
        device = audio.device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        batch_speaker_ids = batch_speaker_ids.to(device)

        audio_feat = self.audio_attention(audio.to(device)) 
        text_feat = self.text_attention(text.to(device))
        video_feat = self.video_attention(video.to(device))

        audio_feat, _ = self.self_attn(audio_feat, audio_feat, audio_feat)
        text_feat, _ = self.self_attn(text_feat, text_feat, text_feat)
        video_feat, _ = self.self_attn(video_feat, video_feat, video_feat)

        video_feat = video_feat.mean(dim=1)
        combined = torch.cat([audio_feat, text_feat, video_feat], dim=-1)
        combined = self.projection_layer(combined) 
        graph_features = self.graph_constructor(node_features, edge_index)
        graph_output = self.grnn(graph_features, edge_index, edge_type)

        graph_output = (graph_output - graph_output.mean()) / (graph_output.std() + 1e-6)  # Normalisation
        graph_output = graph_output * 0.1 
        if batch_speaker_ids.max() >= graph_output.shape[0]:
            raise IndexError(f"batch_speaker_ids contient un indice hors limite ! Max: {graph_output.shape[0]-1}, Trouvé: {batch_speaker_ids.max()}")

        batch_graph_features = graph_output[batch_speaker_ids]
        combined = torch.cat([combined, batch_graph_features], dim=-1)  
        combined = self.final_projection(combined) 

        x = self.fc(combined)
        return x

# --------------------------------------------------------------------------------------------------------------------------
class CoAttentionFusion_MultiPhase(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128):
        super().__init__()

        self.audio_attention = BiGRUWithAttention(input_dim=input_dim_audio, hidden_dim=hidden_dim)
        self.text_attention = BiGRUWithAttention(input_dim=input_dim_text, hidden_dim=hidden_dim)
        self.video_attention = BiGRUWithAttention(input_dim=input_dim_video, hidden_dim=hidden_dim)

        self.graph_constructor = GraphConstructor(input_dim=2304, hidden_dim=hidden_dim)
        self.grnn = GRNN(input_dim=hidden_dim, hidden_dim=hidden_dim)

        self.fusion_projection = nn.Linear(768, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(896, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, audio, text, video, node_features, edge_index, edge_type, batch_speaker_ids):
        audio_feat = self.audio_attention(audio)
        text_feat = self.text_attention(text)
        video_feat = self.video_attention(video)
        video_feat = video_feat.mean(dim=1)

        combined = torch.cat([audio_feat, text_feat, video_feat], dim=-1)
        graph_features = self.graph_constructor(node_features, edge_index)
        graph_output = self.grnn(graph_features, edge_index, edge_type)

        graph_output = (graph_output - graph_output.mean()) / graph_output.std()
        graph_output = graph_output * 0.05
        if batch_speaker_ids.max() >= graph_output.shape[0]:
            raise IndexError(f"batch_speaker_ids contient un indice hors limite ! Max: {graph_output.shape[0]-1}, Trouvé: {batch_speaker_ids.max()}")

        batch_graph_features = graph_output[batch_speaker_ids] 
        combined = torch.cat([combined, batch_graph_features], dim=-1)
        x = self.fc(combined)
        return x

# --------------------------------------------------------------------------------------------------------------------------
class CoAttentionFusion_ModalAttention(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128):
        super().__init__()

        self.co_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, batch_first=True)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, audio, text, video, node_features, edge_index, edge_type, batch_speaker_ids):
        video = video.mean(dim=1)
        modality_features = torch.stack([audio, text, video], dim=1)
        modality_features, _ = self.co_attention(modality_features, modality_features, modality_features)
        modality_features = modality_features.mean(dim=1)


        x = self.fc(modality_features)
        return x
# --------------------------------------------------------------------------------------------------------------------------
class CoAttentionFusion_Gated(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128):
        super().__init__()
        self.fusion_gate = nn.Linear(hidden_dim, 3)
        # self.fc = nn.Sequential(
        #     nn.Linear(768, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.6),
        #     nn.Linear(64, num_classes)
        # )
        self.fc = nn.Linear(768, num_classes)
        self.fusion_projection = nn.Linear(768*3, hidden_dim)

    def forward(self, audio, text, video, node_features, edge_index, edge_type, batch_speaker_ids):
        device = audio.device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        batch_speaker_ids = batch_speaker_ids.to(device)
        video = video.mean(dim=1)  
        concatenated = torch.cat([audio, text, video], dim=-1) 
        concatenated = self.fusion_projection(concatenated) 

        fusion_weights = torch.sigmoid(self.fusion_gate(concatenated)) 
        combined = fusion_weights[:, 0].unsqueeze(1) * audio + \
                fusion_weights[:, 1].unsqueeze(1) * text + \
                fusion_weights[:, 2].unsqueeze(1) * video


        x = self.fc(combined)
        return x
   
# --------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, RGCNConv, global_add_pool

class CoAttentionFusion_Gated_Graph(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128):
        super().__init__()

        # **1️⃣ Gated Fusion Multimodale**
        self.fusion_gate = nn.Linear(hidden_dim, 3)
        
        # **2️⃣ Projection des Modalités**
        self.fusion_projection = nn.Linear(768 * 3, hidden_dim)

        # **3️⃣ Graphe Relationnel pour la Fusion Finale**
        self.graph_constructor = GraphConstructor(input_dim=2304, hidden_dim=hidden_dim)
        self.grnn = GRNN(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # **4️⃣ Classification Finale avec Ajustement du Graphe**
        self.fc = nn.Sequential(
            nn.Linear(896, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, audio, text, video, node_features, edge_index, edge_type, batch_speaker_ids):
        device = audio.device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        batch_speaker_ids = batch_speaker_ids.to(device)

        # **1️⃣ Réduction Temporelle pour Vidéo**
        video = video.mean(dim=1)  

        # **2️⃣ Fusion Gated**
        concatenated = torch.cat([audio, text, video], dim=-1)  
        concatenated = self.fusion_projection(concatenated)  

        fusion_weights = torch.sigmoid(self.fusion_gate(concatenated))  
        combined = fusion_weights[:, 0].unsqueeze(1) * audio + \
                   fusion_weights[:, 1].unsqueeze(1) * text + \
                   fusion_weights[:, 2].unsqueeze(1) * video

        graph_features = self.graph_constructor(node_features, edge_index)
        graph_output = self.grnn(graph_features, edge_index, edge_type)

        graph_output = (graph_output - graph_output.mean()) / (graph_output.std() + 1e-6)  
        graph_output = graph_output * 0.2 
        if batch_speaker_ids.max() >= graph_output.shape[0]:
            raise IndexError(f"batch_speaker_ids contient un indice hors limite ! Max: {graph_output.shape[0]-1}, Trouvé: {batch_speaker_ids.max()}")

        batch_graph_features = graph_output[batch_speaker_ids] 
        combined = torch.cat([combined, batch_graph_features], dim=-1) 
        x = self.fc(combined)
        return x

# --------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, RGCNConv, global_add_pool

class CoAttentionFusion_ELRGNN(nn.Module):
    def __init__(self, input_dim_audio, input_dim_text, input_dim_video, num_classes, hidden_dim=128):
        super().__init__()

        # **1️⃣ Projections Initiales**
        self.audio_projection = nn.Linear(input_dim_audio, 256)
        self.text_projection = nn.Linear(input_dim_text, 256)
        self.video_projection = nn.Linear(input_dim_video, 256)

        # **2️⃣ LSTMs pour Chaque Modalité**
        self.audio_lstm = nn.LSTM(256, hidden_dim, bidirectional=True, batch_first=True)
        self.text_lstm = nn.LSTM(256, hidden_dim, bidirectional=True, batch_first=True)
        self.video_lstm = nn.LSTM(256, hidden_dim, bidirectional=True, batch_first=True)

        # **3️⃣ Convolutions 1D pour Extraction de Patterns**
        self.conv1d_audio = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.conv1d_text = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.conv1d_video = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)

        # **4️⃣ Mécanisme de Fusion par Modulation Sigmoid/Tanh**
        self.sigmoid_fc = nn.Linear(hidden_dim, hidden_dim)
        self.tanh_fc = nn.Linear(hidden_dim, hidden_dim)

        # **5️⃣ Construction et Propagation Graphique**
        self.graph_constructor = GraphConstructor(input_dim=2304, hidden_dim=hidden_dim)
        self.grnn = GRNN(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # **6️⃣ Fusion Multimodale Adaptative**
        self.fusion_projection = nn.Linear(hidden_dim * 4, hidden_dim)  # 128 * 3 (mod.) + 128 (graphe)
        self.adaptive_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # **7️⃣ Classification Finale**
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)            
        )

    def forward(self, audio, text, video, node_features, edge_index, edge_type, batch_speaker_ids):
        device = audio.device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)

        audio_feat = self.audio_projection(audio)
        text_feat = self.text_projection(text)
        video_feat = self.video_projection(video)
        video_feat = video_feat.mean(dim=1)

        audio_feat, _ = self.audio_lstm(audio_feat.unsqueeze(1))
        text_feat, _ = self.text_lstm(text_feat.unsqueeze(1))
        video_feat, _ = self.video_lstm(video_feat.unsqueeze(1))

        audio_feat = audio_feat.mean(dim=1)  # (batch, 128)
        text_feat = text_feat.mean(dim=1)
        video_feat = video_feat.mean(dim=1)

        audio_feat = self.conv1d_audio(audio_feat.unsqueeze(-1)).squeeze(-1)
        text_feat = self.conv1d_text(text_feat.unsqueeze(-1)).squeeze(-1)
        video_feat = self.conv1d_video(video_feat.unsqueeze(-1)).squeeze(-1)

        audio_feat = torch.sigmoid(self.sigmoid_fc(audio_feat)) * torch.tanh(self.tanh_fc(audio_feat))
        text_feat = torch.sigmoid(self.sigmoid_fc(text_feat)) * torch.tanh(self.tanh_fc(text_feat))
        video_feat = torch.sigmoid(self.sigmoid_fc(video_feat)) * torch.tanh(self.tanh_fc(video_feat))

        graph_features = self.graph_constructor(node_features, edge_index)
        graph_output = self.grnn(graph_features, edge_index, edge_type)

        graph_output = (graph_output - graph_output.mean()) / graph_output.std()
        graph_output = graph_output * 0.05

        if batch_speaker_ids.max() >= graph_output.shape[0]:
            raise IndexError(f"batch_speaker_ids contient un indice hors limite ! Max: {graph_output.shape[0]-1}, Trouvé: {batch_speaker_ids.max()}")

        batch_graph_features = graph_output[batch_speaker_ids]

        combined = torch.cat([audio_feat, text_feat, video_feat, batch_graph_features], dim=-1)
        combined = self.fusion_projection(combined)
        combined = self.adaptive_mlp(combined)  
        x = self.fc(combined)
        return x
# --------------------------------------------------------------------------------------------------------------------------
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
    
