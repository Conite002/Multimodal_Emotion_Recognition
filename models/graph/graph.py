import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


# class GraphConstructor(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(GraphConstructor, self).__init__()
#         self.projection = nn.Linear(input_dim, 256)  
#         self.gcn = GCNConv(256, hidden_dim)

#     def forward(self, node_features, edge_index):
#         print(f"On forward: {node_features.shape} {edge_index.shape}")
#         device = node_features.device
#         edge_index = edge_index.to(device)
#         x = self.projection(node_features) 
#         print(f"Projection: {x.shape}")
#         x = self.gcn(x, edge_index).relu() 
#         return x 

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
        return x  # (num_nodes, hidden_dim)


class GRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRNN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        device = x.device
        edge_index = edge_index.to(device)
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        pooled_x = global_mean_pool(x, torch.arange(x.size(0), device=device)) 
        return pooled_x
