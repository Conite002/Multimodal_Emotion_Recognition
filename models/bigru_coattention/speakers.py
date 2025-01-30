import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class SpeakerAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpeakerAttention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, features):
        attention_weights = torch.softmax(self.attention(features), dim=0)
        weights_sum = torch.sum(attention_weights * features, dim=0)
        return weights_sum
        

def get_speaker_node_features(data):
    node_features = {}
    attention_layer = SpeakerAttention(input_dim=2304)

    for split in ['train', 'val', 'test']:
        audio_data = data[split]['audio'].tensors[0]
        text_data = data[split]['text'].tensors[0]
        video_data = data[split]['video'].tensors[0]
        speaker_ids = data[split]['speaker']

        num_samples = audio_data.shape[0]
        unique_speakers = torch.unique(speaker_ids)
        speaker_node_features = []

        for speaker in unique_speakers:
            speaker_indices = (speaker_ids == speaker).nonzero(as_tuple=True)[0]
            speaker_indices = speaker_indices[speaker_indices < num_samples]

            if len(speaker_indices) == 0:
                continue

            speaker_audio = audio_data[speaker_indices]
            speaker_text = text_data[speaker_indices]
            speaker_video = video_data[speaker_indices].mean(dim=1) 
            speaker_features = torch.cat([speaker_audio, speaker_text, speaker_video], dim=-1)
            speaker_representation = attention_layer(speaker_features)
            speaker_node_features.append(speaker_representation)

        if speaker_node_features:
            node_features[split] = torch.stack(speaker_node_features)
        else:
            node_features[split] = torch.empty((0, 2304))

    return node_features
     
def compute_adjacency_matrix(node_features):
    normalized_features = nn.functional.normalize(node_features, dim=-1)
    adjacency_matrix = torch.matmul(normalized_features, normalized_features.T)
    return adjacency_matrix

def get_edge_index(adjacency_matrix, threshold=0.5):
    edge_indices = (adjacency_matrix > threshold).nonzero(as_tuple=False).T  
    return edge_indices

