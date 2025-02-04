import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# -------------------------------------------
# Step 1: Create Cross-Modal Temporal Graph
# -------------------------------------------
def create_crossmodal_temporal_graph(audio_data, video_data, text_data, window_size=5):
    """
    Create a graph with temporal and cross-modal relations.
    :param audio_data: Tensor (num_samples, feature_dim_audio)
    :param video_data: Tensor (num_samples, feature_dim_video)
    :param text_data: Tensor (num_samples, feature_dim_text)
    :param window_size: Temporal window size
    :return: PyTorch Geometric Data object
    """
    num_samples = audio_data.size(0)

    audio_projection = nn.Linear(audio_data.size(1), video_data.size(1)).to(audio_data.device)
    text_projection = nn.Linear(text_data.size(1), video_data.size(1)).to(text_data.device)

    audio_data = audio_projection(audio_data)  
    text_data = text_projection(text_data)  

    # Concatenate node features
    node_features = torch.cat([audio_data, video_data, text_data], dim=0)
    edge_index = []

    for i in range(num_samples):
        for j in range(1, window_size + 1):
            if i + j < num_samples:
                edge_index.append([i, i + j])  
                edge_index.append([i + num_samples, i + num_samples + j])  # Video temporal
                edge_index.append([i + 2 * num_samples, i + 2 * num_samples + j])  # Text temporal

    for i in range(num_samples):
        edge_index.append([i, i + num_samples]) 
        edge_index.append([i, i + 2 * num_samples])  
        edge_index.append([i + num_samples, i]) 
        edge_index.append([i + num_samples, i + 2 * num_samples])  
        edge_index.append([i + 2 * num_samples, i]) 
        edge_index.append([i + 2 * num_samples, i + num_samples])  

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  
    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data

# -------------------------------------------
# Step 2: Define the GNN Model (GAT-based)
# -------------------------------------------
class CrossModalGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, dropout_rate=0.3):
        super(CrossModalGAT, self).__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout_rate)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_dim)  
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge index (2, num_edges)
        Returns:
            out: Node-level predictions (num_nodes, output_dim)
        """
        x = self.gat1(x, edge_index)  
        x = nn.ReLU()(x)
        x = self.gat2(x, edge_index) 
        out = self.fc(x) 
        return out

# -------------------------------------------
# Step 3: Training the GNN with Classification
# -------------------------------------------
def train_crossmodal_gnn(model, graph, labels, optimizer, criterion, num_epochs, device):
    model.to(device)
    graph = graph.to(device)
    labels = labels.to(device)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(graph.x, graph.edge_index) 
        num_samples = labels.size(0) 
        loss = criterion(out[:num_samples], labels)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

