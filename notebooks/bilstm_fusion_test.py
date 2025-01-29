import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.abspath('..'))
from models.bigru_coattention.speakers import get_speaker_node_features, compute_adjacency_matrix, get_edge_index
from models.graph.graph import GraphConstructor
from utils.dataloader import create_dataloader_with_graph_features
from models.bigru_coattention.coattention import CoAttentionFusionWithGraph
from pipelines.training.training_pipeline import train_coattention_graph


data = torch.load(os.path.join("..", "outputs", "embeddings", "loaders_datasets_speakers.pt"))
node_features = get_speaker_node_features(data)

# Access the node features for each split
train_node_features = node_features['train']
val_node_features = node_features['val']
test_node_features = node_features['test']

print("Train Node Features Shape:", train_node_features.shape)
print("Validation Node Features Shape:", val_node_features.shape)
print("Test Node Features Shape:", test_node_features.shape)

train_adj_matrix = compute_adjacency_matrix(node_features['train'])
val_adj_matrix = compute_adjacency_matrix(node_features['val'])
test_adj_matrix = compute_adjacency_matrix(node_features['test'])

print("Train Adjacency Matrix Shape:", train_adj_matrix.shape)  
print("Validation Adjacency Matrix Shape:", val_adj_matrix.shape)  
print("Test Adjacency Matrix Shape:", test_adj_matrix.shape)

train_edge_index = get_edge_index(train_adj_matrix, threshold=0.5)
val_edge_index = get_edge_index(val_adj_matrix, threshold=0.5)
test_edge_index = get_edge_index(test_adj_matrix, threshold=0.5)

print("Train Edge Index Shape:", train_edge_index.shape)  # Example: (2, num_edges)
print("Validation Edge Index Shape:", val_edge_index.shape)
print("Test Edge Index Shape:", test_edge_index.shape)


graph_constructor = GraphConstructor(input_dim=2304, hidden_dim=128, num_nodes=train_node_features.shape[0])

train_graph_features = graph_constructor(node_features['train'], train_edge_index)
val_graph_features = graph_constructor(node_features['val'], val_edge_index)
test_graph_features = graph_constructor(node_features['test'], test_edge_index)

print("Train Graph Features Shape:", train_graph_features.shape)  # Expected: (100, 128)
print("Validation Graph Features Shape:", val_graph_features.shape)
print("Test Graph Features Shape:", test_graph_features.shape)
# add to data graph features
data['train']['graph_features'] = train_graph_features
data['val']['graph_features'] = val_graph_features
data['test']['graph_features'] = test_graph_features
torch.save(data, os.path.join("..", "outputs", "embeddings", "loaders_datasets_speakers_graph.pt"))


train_loaders, val_loaders, label_mapping = create_dataloader_with_graph_features(data)
print("Train Loaders:", train_loaders)
print("Validation Loaders:", val_loaders)
print("Label Mapping:", label_mapping)

# Define the model
model = CoAttentionFusionWithGraph(
    input_dim_audio=768,
    input_dim_text=768,
    input_dim_video=768,
    num_classes=7,  
    hidden_dim=128,
    dropout_rate=0.6
)


train_coattention_graph(model, train_loaders, val_loaders, num_epochs=20, lr=1e-3, verbose=False, logfile="coattention_graph.log", save_model="coattention_graph.pth")
