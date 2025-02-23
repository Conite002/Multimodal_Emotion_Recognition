import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import BatchNorm1d


class GraphConstructor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphConstructor, self).__init__()
        self.projection = nn.Linear(input_dim, 256)  
        self.gcn1 = GATConv(256, hidden_dim, heads=2, concat=False)
        # self.gcn2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, node_features, edge_index):
        device = node_features.device
        edge_index = edge_index.to(device)
        num_nodes = node_features.shape[0]

        if edge_index.max() >= num_nodes:
            raise ValueError(f" Indice hors limite : max {num_nodes-1}, trouvé {edge_index.max()}")

        adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        degree = adj_matrix.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1
        adj_matrix = adj_matrix / degree

        x = self.projection(node_features)
        x = self.norm1(self.gcn1(x, edge_index).relu())
        # x = self.norm2(self.gcn2(x, edge_index).relu())
        x = self.dropout(x)
        x = self.norm3(self.gcn3(x, edge_index).relu())
        x = self.dropout(x)
        x, _ = self.gru(x.unsqueeze(1))  
        return x.squeeze(1)

class GRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRNN, self).__init__()
        self.gcn1 = RGCNConv(input_dim, hidden_dim, num_relations=3)
        self.gcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x, edge_index, edge_type):
        device = x.device
        edge_index, edge_type = edge_index.to(device), edge_type.to(device)
        drop_mask = torch.bernoulli(torch.full((edge_index.shape[1],), 0.9, device=device)).bool()
        edge_index, edge_type = edge_index[:, drop_mask], edge_type[drop_mask]
        adj_matrix = compute_adjacency_matrix(x)
        propagation_matrix = compute_gfpush_matrix(adj_matrix)

        x = self.norm1(self.gcn1(x, edge_index, edge_type).relu())
        x = self.dropout(x)
        x = self.norm2(self.gcn2(x, edge_index, edge_type).relu())
        x = self.dropout(x)
        x = forward_push(edge_index, x, alpha=0.1, num_iterations=3)
        # if edge_index.dtype != torch.long:
        #     edge_index = edge_index.long()
        # pooled_x = global_add_pool(x, torch.arange(x.size(0), device=device)) 
        x = torch.matmul(propagation_matrix, x)
        return x
        # return pooled_x

# def batch_graphify(lengths, n_modals=1, edge_multi=True, edge_temp=True, P=2, F=2, temp_sample_rate=0.5, multi_modal_sample_rate=0.5, edge_reduction_rate=0.75):

#     """
#     lengths: list of lengths of each sample in the batch
#     n_modals: number of modalities
#     edge_multi: whether to add multi modal edges
#     edge_temp: whether to add temporal edges
#     P: number of previous frames to connect to
#     F: number of future frames to connect to
#     temp_sample_rate: rate at which to sample temporal edges
#     multi_modal_sample_rate: rate at which to sample multi modal edges
#     edge_reduction_rate: rate at which to reduce the number of edges
#     """
#     batch_size = len(lengths)
#     edge_index = []
#     edge_type = []
#     for i in range(batch_size):
#         for j in range(lengths[i]):
#             if edge_multi and torch.rand(1) < multi_modal_sample_rate:
#                 edge_index.append([i*n_modals, i*n_modals+1])
#                 edge_index.append([i*n_modals+1, i*n_modals])
#                 edge_type.append(0)  
#                 edge_type.append(0)  
#             if edge_temp and torch.rand(1) < temp_sample_rate:
#                 for p in range(1, P+1):
#                     if j-p >= 0:
#                         edge_index.append([i*n_modals+j, i*n_modals+j-p])
#                         edge_type.append(1)
#                     if j+p < lengths[i]:
#                         edge_index.append([i*n_modals+j, i*n_modals+j+p])
#                         edge_type.append(1)
#                 for f in range(1, F+1):
#                     if j-f >= 0:
#                         edge_index.append([i*n_modals+j, i*n_modals+j-f])
#                         edge_type.append(1)
#                     if j+f < lengths[i]:
#                         edge_index.append([i*n_modals+j, i*n_modals+j+f])
#                         edge_type.append(1)
#     edge_index = torch.tensor(edge_index).t().contiguous()
#     edge_type = torch.tensor(edge_type)
#     perm = torch.randperm(edge_index.size(1))[:int(edge_reduction_rate*edge_index.size(1))]
#     edge_index = edge_index[:, perm]
#     edge_type = edge_type[perm]
#     return edge_index, edge_type


def forward_push(edge_index, node_features, alpha=0.1, num_iterations=3):
    num_nodes = node_features.shape[0]
    adj_matrix = torch.zeros((num_nodes, num_nodes), device=node_features.device)
    adj_matrix[edge_index[0], edge_index[1]] = 1

    degree = adj_matrix.sum(dim=1, keepdim=True)
    degree[degree == 0] = 1
    adj_matrix = adj_matrix / degree

    for _ in range(num_iterations):
        node_features = node_features.clone() * (1 - alpha) + alpha * torch.matmul(adj_matrix, node_features)

    return node_features

class GRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRNN, self).__init__()
        self.gcn1 = RGCNConv(input_dim, hidden_dim, num_relations=3)
        self.gcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x, edge_index, edge_type):
        device = x.device
        edge_index, edge_type = edge_index.to(device), edge_type.to(device)
        
        drop_mask = torch.bernoulli(torch.full((edge_index.shape[1],), 0.9, device=device)).bool()
        edge_index, edge_type = edge_index[:, drop_mask], edge_type[drop_mask]
        
        adj_matrix = compute_adjacency_matrix(x)
        propagation_matrix = compute_gfpush_matrix(adj_matrix)

        x = self.norm1(self.gcn1(x, edge_index, edge_type).relu())
        x = self.dropout(x)
        x = self.norm2(self.gcn2(x, edge_index, edge_type).relu())
        x = self.dropout(x)

        x = forward_push(edge_index, x, alpha=0.1, num_iterations=3)
        x = torch.matmul(propagation_matrix, x)

        return x

def compute_gfpush_matrix(adj_matrix, r_max=0.1, sparsification_ratio=0.5):
    num_nodes = adj_matrix.shape[0]
    propagation_matrix = torch.zeros_like(adj_matrix)

    degree_matrix = adj_matrix.sum(dim=1, keepdim=True)
    degree_matrix[degree_matrix == 0] = 1
    normalized_adj = adj_matrix / degree_matrix

    for i in range(num_nodes):
        push_vector = torch.zeros(num_nodes, device=adj_matrix.device)
        push_vector[i] = 1
        residual_vector = push_vector.clone()

        while residual_vector.sum() > r_max:
            max_index = torch.argmax(residual_vector)
            push_amount = residual_vector[max_index] * normalized_adj[max_index]

            propagation_matrix = propagation_matrix.clone()
            residual_vector = residual_vector.clone()

            propagation_matrix[i] = propagation_matrix[i] + push_amount
            residual_vector[max_index] = 0

    num_keep = int(sparsification_ratio * num_nodes)
    _, top_indices = torch.topk(propagation_matrix, num_keep, dim=1)
    sparse_matrix = torch.zeros_like(propagation_matrix)

    for i in range(num_nodes):
        sparse_matrix[i, top_indices[i]] = propagation_matrix[i, top_indices[i]]

    return sparse_matrix

def get_edge_type(edge_index, node_features):
    num_edges = edge_index.shape[1]
    src_features = node_features[edge_index[0]]  
    dest_features = node_features[edge_index[1]] 

    src_norm = torch.nn.functional.normalize(src_features, dim=1)
    dest_norm = torch.nn.functional.normalize(dest_features, dim=1)

    cosine_sim = torch.sum(src_norm * dest_norm, dim=1)  
    euclidean_dist = torch.norm(src_features - dest_features, dim=1)  

    edge_types = torch.zeros(num_edges, dtype=torch.long)

    edge_types[cosine_sim > 0.9] = 1 
    edge_types[euclidean_dist < 0.5] = 2  

    return edge_types



class SpeakerAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpeakerAttention, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, features):
        attention_weights = torch.softmax(self.attention(features), dim=0)
        weights_sum = torch.sum(attention_weights * features, dim=0)
        return weights_sum
        

def get_edge_index(adjacency_matrix, threshold=0.3):
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError(f" La matrice d'adjacence doit être carrée ! Shape: {adjacency_matrix.shape}")

    edge_indices = (adjacency_matrix > threshold).nonzero(as_tuple=False).T  
    num_nodes = adjacency_matrix.shape[0]
    valid_edges = (edge_indices[0] < num_nodes) & (edge_indices[1] < num_nodes)
    edge_indices = edge_indices[:, valid_edges]

    return edge_indices

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
            num_speakers = len(torch.unique(speaker_ids))
            node_features[split] = torch.zeros((num_speakers, 2304))

        expected_num_nodes = len(torch.unique(data[split]['speaker']))
        if node_features[split].shape[0] != expected_num_nodes:
            print(f"Nombre de nœuds incohérent : Attendu {expected_num_nodes}, Obtenu {node_features[split].shape[0]}")

    return node_features

def compute_adjacency_matrix(node_features):

    num_nodes = node_features.shape[0]
    adj_matrix = torch.matmul(node_features, node_features.T) 
    adj_matrix.fill_diagonal_(0)

    return adj_matrix

