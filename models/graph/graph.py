import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool



class GraphConstructor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphConstructor, self).__init__()
        self.projection = nn.Linear(input_dim, 256)  
        self.gcn1 = GCNConv(256, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
             
    def forward(self, node_features, edge_index):
        device = node_features.device
        edge_index = edge_index.to(device)
        x = self.projection(node_features)
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = global_max_pool(x, torch.arange(x.size(0), device=device))
        return x 
    
    


class GRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRNN, self).__init__()
        self.gcn1 = RGCNConv(input_dim, hidden_dim, num_relations=9)
        self.gcn2 = RGCNConv(hidden_dim, hidden_dim, num_relations=9)

    def forward(self, x, edge_index):
        device = x.device
        edge_index, edge_type = batch_graphify([x.size(0)], edge_multi=True, edge_temp=True, P=3, F=3, temp_sample_rate=0.4, multi_modal_sample_rate=0.4, edge_reduction_rate=0.5)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        
        x = self.gcn1(x, edge_index, edge_type).relu()
        x = self.gcn2(x, edge_index, edge_type).relu()
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        pooled_x = global_mean_pool(x, torch.arange(x.size(0), device=device)) 
        return pooled_x
    
def batch_graphify(lengths, n_modals=1, edge_multi=True, edge_temp=True, P=2, F=2, temp_sample_rate=0.5, multi_modal_sample_rate=0.5, edge_reduction_rate=0.75):
    """
    lengths: list of lengths of each sample in the batch
    n_modals: number of modalities
    edge_multi: whether to add multi modal edges
    edge_temp: whether to add temporal edges
    P: number of previous frames to connect to
    F: number of future frames to connect to
    temp_sample_rate: rate at which to sample temporal edges
    multi_modal_sample_rate: rate at which to sample multi modal edges
    edge_reduction_rate: rate at which to reduce the number of edges
    """
    batch_size = len(lengths)
    edge_index = []
    edge_type = []
    for i in range(batch_size):
        for j in range(lengths[i]):
            if edge_multi and torch.rand(1) < multi_modal_sample_rate:
                edge_index.append([i*n_modals, i*n_modals+1])
                edge_index.append([i*n_modals+1, i*n_modals])
                edge_type.append(0)  
                edge_type.append(0)  
            if edge_temp and torch.rand(1) < temp_sample_rate:
                for p in range(1, P+1):
                    if j-p >= 0:
                        edge_index.append([i*n_modals+j, i*n_modals+j-p])
                        edge_type.append(1)
                    if j+p < lengths[i]:
                        edge_index.append([i*n_modals+j, i*n_modals+j+p])
                        edge_type.append(1)
                for f in range(1, F+1):
                    if j-f >= 0:
                        edge_index.append([i*n_modals+j, i*n_modals+j-f])
                        edge_type.append(1)
                    if j+f < lengths[i]:
                        edge_index.append([i*n_modals+j, i*n_modals+j+f])
                        edge_type.append(1)
    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_type = torch.tensor(edge_type)
    perm = torch.randperm(edge_index.size(1))[:int(edge_reduction_rate*edge_index.size(1))]
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    return edge_index, edge_type


# class GraphConstructor(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(GraphConstructor, self).__init__()
#         self.projection = nn.Linear(input_dim, 256)  
#         self.gcn1 = GCNConv(256, hidden_dim)
#         self.gcn2 = GCNConv(hidden_dim, hidden_dim)

#     def forward(self, node_features, edge_index):
#         device = node_features.device
#         edge_index = edge_index.to(device)
#         x = self.projection(node_features)
#         x = self.gcn1(x, edge_index).relu()
#         x = self.gcn2(x, edge_index).relu()
#         return x  # (num_nodes, hidden_dim)


# class GRNN(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(GRNN, self).__init__()
#         self.gcn1 = GCNConv(input_dim, hidden_dim)
#         self.gcn2 = GCNConv(hidden_dim, hidden_dim)

#     def forward(self, x, edge_index):
#         device = x.device
#         edge_index = edge_index.to(device)
#         x = self.gcn1(x, edge_index).relu()
#         x = self.gcn2(x, edge_index).relu()
#         if edge_index.dtype != torch.long:
#             edge_index = edge_index.long()
#         pooled_x = global_mean_pool(x, torch.arange(x.size(0), device=device)) 
#         return pooled_x
