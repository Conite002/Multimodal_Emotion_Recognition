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


train_node_features = node_features['train']
val_node_features = node_features['val']
test_node_features = node_features['test']

train_adj_matrix = compute_adjacency_matrix(train_node_features)
val_adj_matrix = compute_adjacency_matrix(val_node_features)
test_adj_matrix = compute_adjacency_matrix(test_node_features)

train_edge_index = get_edge_index(train_adj_matrix, threshold=0.5)
val_edge_index = get_edge_index(val_adj_matrix, threshold=0.5)
test_edge_index = get_edge_index(test_adj_matrix, threshold=0.5)

# Valid edges
valid_edges = (train_edge_index[0] < train_node_features.shape[0]) & (train_edge_index[1] < train_node_features.shape[0])
train_edge_index = train_edge_index[:, valid_edges]
valid_edges = (val_edge_index[0] < val_node_features.shape[0]) & (val_edge_index[1] < val_node_features.shape[0])
val_edge_index = val_edge_index[:, valid_edges]
valid_edges = (test_edge_index[0] < test_node_features.shape[0]) & (test_edge_index[1] < test_node_features.shape[0])
test_edge_index = test_edge_index[:, valid_edges]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph_constructor = GraphConstructor(input_dim=2304, hidden_dim=128).to(device)

train_graph_features = graph_constructor(train_node_features.to(device), train_edge_index.to(device))
val_graph_features = graph_constructor(val_node_features.to(device), val_edge_index.to(device))
test_graph_features = graph_constructor(test_node_features.to(device), test_edge_index.to(device))


data['train']['graph_features'] = train_graph_features
data['val']['graph_features'] = val_graph_features
data['test']['graph_features'] = test_graph_features

torch.save(data, os.path.join("..", "outputs", "embeddings", "loaders_datasets_speakers_graph.pt"))

num_speakers = len(torch.unique(data['train']['speaker']))
train_utterance_graph_features = train_graph_features[data['train']['speaker']]
val_utterance_graph_features = val_graph_features[data['val']['speaker']]
test_utterance_graph_features = test_graph_features[data['test']['speaker']]

train_loaders, val_loaders, test_loaders = create_dataloader_with_graph_features(data)

nodes_edges = {
    'node_features': {
        'train': train_node_features,
        'val': val_node_features,
        'test': test_node_features
    },
    'edge_index': {
        'train': train_edge_index,
        'val': val_edge_index,
        'test': test_edge_index
    }
}

model = CoAttentionFusionWithGraph(
    input_dim_audio=768,
    input_dim_text=768,
    input_dim_video=768,
    num_classes=7,
    hidden_dim=128,
    dropout_rate=0.6,
    num_speakers=num_speakers,

).to(device)

train_coattention_graph(
    model,
    train_loaders,
    val_loaders,
    num_epochs=20,
    lr=1e-3,
    verbose=False,
    logfile="coattention_graph_2.log",
    model_name='best_model_graph_coattention_2.pth',
    num_classes=7,
    nodes_edges=nodes_edges
)

# load the best model
model.load_state_dict(torch.load(os.path.join("best_model_graph_coattention_2.pth")))

# evaluate the model
from pipelines.evaluation.evaluation_pipeline import evaluate_model_coattention_graph
val_loss, val_accuracy, precision, recall, f1, accuracies = evaluate_model_coattention_graph(
    model,
    test_loaders,
    nn.CrossEntropyLoss(),
    device,
    num_classes=7,
    logfile="evaluation_log_graph_coattention",
    node_features=nodes_edges['node_features']['test'],
    edge_index=nodes_edges['edge_index']['test'],
    verbose=True
)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%")

