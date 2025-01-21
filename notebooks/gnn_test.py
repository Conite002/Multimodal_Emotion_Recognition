import torch
import numpy as np
from torch_geometric.data import Data

import torch
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn as nn
import os, json, sys
from models.fusion_modals.transformerMultimodal import TransformerMultimodal, validate_model_TF_FUSION_MULTIMODAL, test_model_TF_FUSION_MULTIMODAL, train_model_TF_FUSION_MULTIMODAL
from preprocessing.text.preprocess_text import preprocess_text_for_model, load_text_model
from preprocessing.audio.preprocess_audio import preprocess_audio_for_model, load_audio_model, extract_audio
from pipelines.preprocessing.data_pipeline import generate_metadata
import numpy as np
import torch.nn as nn
from pipelines.training.training_pipeline import train_model, train_model_coattention
from models.audio.audio_model import AudioCNNClassifier
from pipelines.training.training_pipeline import evaluate_model

from torch import nn, optim
from tqdm import tqdm
from models.bigru_coattention.coattention import CoAttentionFusion
from utils.logger import create_logger
from sklearn.metrics import precision_score, recall_score, f1_score
from pipelines.evaluation.evaluation_pipeline import test_model_coattention, evaluate_model_coattention
from models.bigru_coattention.multimodal import MultiModalDataset, MultiModalDatasetWithSpeaker
from utils.dataloader import extract_tensors_from_tensordataset


saved_data = torch.load(os.path.join("..", "outputs", "embeddings", "loaders_datasets.pt"))

train_dataset = MultiModalDataset(
    saved_data['train']['audio'],
    saved_data['train']['text'],
    saved_data['train']['video'],
    saved_data['train']['labels']
)

val_dataset = MultiModalDataset(
    saved_data['val']['audio'],
    saved_data['val']['text'],
    saved_data['val']['video'],
    saved_data['val']['labels']
)

test_dataset = MultiModalDataset(
    saved_data['test']['audio'],
    saved_data['test']['text'],
    saved_data['test']['video'],
    saved_data['test']['labels']
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------
# Generate model
# --------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv

class MultimodalGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, heads, dropout):
        super(MultimodalGraphModel, self).__init__()
        self.rgcn = RGCNConv(input_dim, hidden_dim, num_relations=num_relations)
        self.transformer = TransformerConv(hidden_dim, output_dim, heads=heads, dropout=dropout)
        self.fc = nn.Linear(output_dim * heads, 3)  # Pour 3 classes de sortie

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        x = self.rgcn(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.transformer(x, edge_index)
        x = torch.relu(x)

        x = self.fc(x.mean(dim=0, keepdim=True)) 
        return x

# --------------------------------------------------------------------------------------------------------
# Training and testing functions
# --------------------------------------------------------------------------------------------------------
import torch.optim as optim

# Hyperparamètres
input_dim = 768
hidden_dim = 128
output_dim = 64
num_relations = 6  # Types de relations : 3 temporelles + 3 cross-modales
heads = 4
dropout = 0.1

# Modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalGraphModel(input_dim, hidden_dim, output_dim, num_relations, heads, dropout).to(device)

# Optimiseur et fonction de perte
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

#--------------------------------------------------------------------------------------------------------
# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # Create graph for the current batch
        train_graph = create_graph(batch, wp=1, wf=1, edge_type_to_idx={
            'temporal_audio': 0,
            'temporal_text': 1,
            'temporal_video': 2,
            'crossmodal_audio_text': 3,
            'crossmodal_audio_video': 4,
            'crossmodal_text_video': 5
        }, device=device)

        logits = model(train_graph)
        loss = loss_fn(logits, train_graph.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Testing
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        test_graph = create_graph(batch, wp=1, wf=1, edge_type_to_idx={
            'temporal_audio': 0,
            'temporal_text': 1,
            'temporal_video': 2,
            'crossmodal_audio_text': 3,
            'crossmodal_audio_video': 4,
            'crossmodal_text_video': 5
        }, device=device)

        logits = model(test_graph)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(test_graph.y.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# Calculate accuracy
accuracy = (all_preds == all_labels).float().mean().item()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
