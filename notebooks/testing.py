import os
import torch
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.abspath('..'))
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
from models.bigru_coattention.multimodal import MultiModalDataset



saving_dir = os.path.join("..", "outputs", "embeddings")
saved_data = torch.load(os.path.join(saving_dir, "loaders_datasets.pt"))

train_loaders = {
    'audio': DataLoader(saved_data['train']['audio'], batch_size=32, shuffle=True),
    'text': DataLoader(saved_data['train']['text'], batch_size=32, shuffle=True),
    'video': DataLoader(saved_data['train']['video'], batch_size=32, shuffle=True),
    'label': DataLoader(saved_data['train']['labels'], batch_size=32, shuffle=True)
}

val_loaders = {
    'audio': DataLoader(saved_data['val']['audio'], batch_size=32, shuffle=False),
    'text': DataLoader(saved_data['val']['text'], batch_size=32, shuffle=False),
    'video': DataLoader(saved_data['val']['video'], batch_size=32, shuffle=False),
    'label': DataLoader(saved_data['val']['labels'], batch_size=32, shuffle=False)
}

test_loaders = {
    'audio': DataLoader(saved_data['test']['audio'], batch_size=32, shuffle=False),
    'text': DataLoader(saved_data['test']['text'], batch_size=32, shuffle=False),
    'video': DataLoader(saved_data['test']['video'], batch_size=32, shuffle=False),
    'label': DataLoader(saved_data['test']['labels'], batch_size=32, shuffle=False)
}

import torch
import numpy as np


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

# --------------------------------------------------------------------------------
# Train Model
# --------------------------------------------------------------------------------


co_attention_model = CoAttentionFusion(input_dim_audio=768, input_dim_text=768, input_dim_video=768, num_classes=7)
trained_model = train_model_coattention(co_attention_model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3, device="cpu", num_classes=7, logfile=os.path.join('..','logs','training_logs', "coattention_training.log"), verbose=False)
test_accuracy, precision, recall, f1 = test_model_coattention(trained_model, test_loader, device="cpu", num_classes=7, verbose=True, logfile=os.path.join('..','logs','training_logs', "coattention_evaluation.log"))

