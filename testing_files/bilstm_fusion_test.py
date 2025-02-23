# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import sys, os
# sys.path.append(os.path.abspath('..'))
# from models.graph.graph import GraphConstructor, get_speaker_node_features, compute_adjacency_matrix, get_edge_index, get_edge_type
# from utils.dataloader import create_dataloader_with_graph_features
# from models.bigru_coattention.coattention import CoAttentionFusionWithGraph
# from pipelines.training.training_pipeline import train_coattention_graph
# import random


# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False




# data = torch.load(os.path.join("..", "outputs", "embeddings", "loaders_datasets_speakers.pt"))
# node_features = get_speaker_node_features(data)

# train_node_features = node_features['train']
# val_node_features = node_features['val']
# test_node_features = node_features['test']

# train_adj_matrix = compute_adjacency_matrix(train_node_features)
# val_adj_matrix = compute_adjacency_matrix(val_node_features)
# test_adj_matrix = compute_adjacency_matrix(test_node_features)

# train_edge_index = get_edge_index(train_adj_matrix, threshold=0.4)
# val_edge_index = get_edge_index(val_adj_matrix, threshold=0.5)
# test_edge_index = get_edge_index(test_adj_matrix, threshold=0.5)

# train_edge_type = get_edge_type(train_edge_index, train_node_features)
# val_edge_type = get_edge_type(val_edge_index, val_node_features)
# test_edge_type = get_edge_type(test_edge_index, test_node_features)

# torch.save(data, os.path.join("..", "outputs", "embeddings", "loaders_datasets_speakers_graph.pt"))
# num_speakers = len(torch.unique(data['train']['speaker']))
# train_loaders, val_loaders, test_loaders = create_dataloader_with_graph_features(data)
# nodes_edges = {
#     'node_features': {
#         'train': train_node_features,
#         'val': val_node_features,
#         'test': test_node_features
#     },
#     'edge_index': {
#         'train': train_edge_index,
#         'val': val_edge_index,
#         'test': test_edge_index
#     },
#     'edge_type': {
#         'train': train_edge_type,
#         'val': val_edge_type,
#         'test': test_edge_type
#     }
# }
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CoAttentionFusionWithGraph(
#     input_dim_audio=768,
#     input_dim_text=768,
#     input_dim_video=768,
#     num_classes=7,
#     hidden_dim=128,
#     dropout_rate=0.6,
#     num_speakers=num_speakers,

# ).to(device)

# train_coattention_graph(
#     model,
#     train_loaders,
#     val_loaders,
#     num_epochs=50,
#     lr=1e-3,
#     verbose=False,
#     logfile="coattention_graph_2_aug_audio.log",
#     model_name='best_model_graph_coattention_2_aug_audio.pth',
#     num_classes=7,
#     nodes_edges=nodes_edges
# )

# model.load_state_dict(torch.load(os.path.join("best_model_graph_coattention_2_aug_audio.pth")))

# from pipelines.evaluation.evaluation_pipeline import evaluate_model_coattention_graph
# val_loss, val_accuracy, precision, recall, f1, accuracies = evaluate_model_coattention_graph(
#     model,
#     test_loaders,
#     nn.CrossEntropyLoss(),
#     device,
#     num_classes=7,
#     logfile="evaluation_log_graph_coattention__sampler",
#     node_features=nodes_edges['node_features']['test'],
#     edge_index=nodes_edges['edge_index']['test'],
#     edge_type=nodes_edges['edge_type']['test'],
#     verbose=True
# )
# print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%")
# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import sys, os
# sys.path.append(os.path.abspath('..'))
# from models.graph.graph import GraphConstructor, get_speaker_node_features, compute_adjacency_matrix, get_edge_index, get_edge_type
# from utils.dataloader import create_dataloader_with_graph_features
# from models.bigru_coattention.coattention import CoAttentionFusionWithGraph
# from pipelines.training.training_pipeline import train_coattention_graph
# import random


# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False




# data = torch.load(os.path.join("..", "outputs", "embeddings", "loaders_datasets_speakers.pt"))
# node_features = get_speaker_node_features(data)

# train_node_features = node_features['train']
# val_node_features = node_features['val']
# test_node_features = node_features['test']

# train_adj_matrix = compute_adjacency_matrix(train_node_features)
# val_adj_matrix = compute_adjacency_matrix(val_node_features)
# test_adj_matrix = compute_adjacency_matrix(test_node_features)

# train_edge_index = get_edge_index(train_adj_matrix, threshold=0.4)
# val_edge_index = get_edge_index(val_adj_matrix, threshold=0.5)
# test_edge_index = get_edge_index(test_adj_matrix, threshold=0.5)

# train_edge_type = get_edge_type(train_edge_index, train_node_features)
# val_edge_type = get_edge_type(val_edge_index, val_node_features)
# test_edge_type = get_edge_type(test_edge_index, test_node_features)

# torch.save(data, os.path.join("..", "outputs", "embeddings", "loaders_datasets_speakers_graph.pt"))
# num_speakers = len(torch.unique(data['train']['speaker']))
# train_loaders, val_loaders, test_loaders = create_dataloader_with_graph_features(data)
# nodes_edges = {
#     'node_features': {
#         'train': train_node_features,
#         'val': val_node_features,
#         'test': test_node_features
#     },
#     'edge_index': {
#         'train': train_edge_index,
#         'val': val_edge_index,
#         'test': test_edge_index
#     },
#     'edge_type': {
#         'train': train_edge_type,
#         'val': val_edge_type,
#         'test': test_edge_type
#     }
# }
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CoAttentionFusionWithGraph(
#     input_dim_audio=768,
#     input_dim_text=768,
#     input_dim_video=768,
#     num_classes=7,
#     hidden_dim=128,
#     dropout_rate=0.6,
#     num_speakers=num_speakers,

# ).to(device)

# train_coattention_graph(
#     model,
#     train_loaders,
#     val_loaders,
#     num_epochs=50,
#     lr=1e-3,
#     verbose=False,
#     logfile="coattention_graph_2_aug_audio.log",
#     model_name='best_model_graph_coattention_2_aug_audio.pth',
#     num_classes=7,
#     nodes_edges=nodes_edges
# )

# model.load_state_dict(torch.load(os.path.join("best_model_graph_coattention_2_aug_audio.pth")))

# from pipelines.evaluation.evaluation_pipeline import evaluate_model_coattention_graph
# val_loss, val_accuracy, precision, recall, f1, accuracies = evaluate_model_coattention_graph(
#     model,
#     test_loaders,
#     nn.CrossEntropyLoss(),
#     device,
#     num_classes=7,
#     logfile="evaluation_log_graph_coattention__sampler",
#     node_features=nodes_edges['node_features']['test'],
#     edge_index=nodes_edges['edge_index']['test'],
#     edge_type=nodes_edges['edge_type']['test'],
#     verbose=True
# )
# print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%")


import sys, os
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn as nn
from pipelines.evaluation.evaluation_pipeline import evaluate_model_coattention_graph
import pandas as pd
import numpy as np

from models.graph.graph import GraphConstructor, get_speaker_node_features, compute_adjacency_matrix, get_edge_index, get_edge_type
from utils.dataloader import create_dataloader_with_graph_features
from models.bigru_coattention.coattention import CoAttentionFusionWithGraph, CoAttentionFusion_Baseline, CoAttentionFusion_SelfAttn, CoAttentionFusion_MultiPhase, CoAttentionFusion_ModalAttention, CoAttentionFusion_Gated, CoAttentionFusion_ELRGNN, CoAttentionFusion_SelfAttn_GNN, CoAttentionFusion_Gated_Graph
from pipelines.training.training_pipeline import train_coattention_graph
import random


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



import torch
import os

file_path = os.path.join("..", "outputs", "embeddings", "loaders_datasets_speakers.pt")

data = torch.load(file_path, weights_only=False)  
node_features = get_speaker_node_features(data)

train_node_features = node_features['train']
val_node_features = node_features['val']
test_node_features = node_features['test']

train_adj_matrix = compute_adjacency_matrix(train_node_features)
val_adj_matrix = compute_adjacency_matrix(val_node_features)
test_adj_matrix = compute_adjacency_matrix(test_node_features)

train_edge_index = get_edge_index(train_adj_matrix, threshold=0.4)
val_edge_index = get_edge_index(val_adj_matrix, threshold=0.5)
test_edge_index = get_edge_index(test_adj_matrix, threshold=0.5)

train_edge_type = get_edge_type(train_edge_index, train_node_features)
val_edge_type = get_edge_type(val_edge_index, val_node_features)
test_edge_type = get_edge_type(test_edge_index, test_node_features)

torch.save(data, os.path.join("..", "outputs", "embeddings", "loaders_datasets_speakers_graph.pt"))
num_speakers = len(torch.unique(data['train']['speaker']))
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
    },
    'edge_type': {
        'train': train_edge_type,
        'val': val_edge_type,
        'test': test_edge_type
    }
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


models = {
    # "Baseline_BiGRU": CoAttentionFusion_Baseline,
    # "BiGRU_SelfAttn": CoAttentionFusion_SelfAttn,
    # "BiGRU_selfAttnGNN":CoAttentionFusion_SelfAttn_GNN,
    # "Fusion_MultiPhase": CoAttentionFusion_MultiPhase,
    # "Modal_Attention": CoAttentionFusion_ModalAttention,
    # "Gated_Fusion_GNN": CoAttentionFusion_Gated_Graph,
    "Gated_Fusion": CoAttentionFusion_Gated,
    # "ELR_GNN": CoAttentionFusion_ELRGNN,
}

results = {}

num_epochs = 50
learning_rate = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name, ModelClass in models.items():
    print(f"\ Entraînement du modèle : {model_name}")

    model = ModelClass(
        input_dim_audio=768,
        input_dim_text=768,
        input_dim_video=768,
        num_classes=7,
        hidden_dim=128
    ).to(device)

    # **Entraînement**
    model_save_path = f"best_model_{model_name}.pth"
    model = train_coattention_graph(
        model,
        train_loaders,
        val_loaders,
        original_data=data,
        num_epochs=num_epochs,
        lr=learning_rate,
        verbose=False,
        logfile=f"{model_name}_training.log",
        model_name=model_save_path,
        num_classes=7,
        nodes_edges=nodes_edges,
    )

    # model.load_state_dict(torch.load(model_save_path, map_location=device))

    # **Évaluation**
    print(f"\n Évaluation du modèle : {model_name}")
    best_model_path = os.path.join('..', 'outputs', 'models', model_name)

    val_loss, val_accuracy, precision, recall, f1, accuracies = evaluate_model_coattention_graph(
        model,
        test_loaders,
        nn.CrossEntropyLoss(),
        device,
        num_classes=7,
        logfile=f"{best_model_path}_evaluation.log",
        node_features=nodes_edges['node_features']['test'],
        edge_index=nodes_edges['edge_index']['test'],
        edge_type=nodes_edges['edge_type']['test'],
        verbose=True
    )

    results[model_name] = {
        "Validation Loss": val_loss,
        "Validation Accuracy": val_accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100
    }

    print(f"\n🔍 Résultats pour {model_name}:")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%, "
          f"Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1 Score: {f1 * 100:.2f}%")

print("\n Comparaison des performances des modèles :")
for model_name, metrics in results.items():
    print(f"\n🔹 {model_name}:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.2f}%")

# save results to csv
results_df = pd.DataFrame(results).T
results_df.to_csv(os.path.join('..', 'outputs', 'models', 'results.csv'))
