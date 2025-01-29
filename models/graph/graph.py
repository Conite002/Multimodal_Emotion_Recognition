import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GraphConstructor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super(GraphConstructor, self).__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.num_nodes = num_nodes

    def forward(self, node_features, adjacency_matrix):
        x = self.gcn(node_features, adjacency_matrix)
        return x


class GRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRNN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        pooled_x = global_mean_pool(x, torch.arange(x.size(0))) 
        return pooled_x
