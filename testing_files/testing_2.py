import torch
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.abspath('..'))
import torch
import torch.nn as nn
import os, json, sys
from models.fusion_modals.transformerMultimodal import TransformerMultimodal, validate_model_TF_FUSION_MULTIMODAL, test_model_TF_FUSION_MULTIMODAL, train_model_TF_FUSION_MULTIMODAL
import torch.nn as nn
from models.bigru_coattention.multimodal import MultiModalDataset, MultiModalDatasetWithSpeaker
from utils.dataloader import extract_tensors_from_tensordataset


saving_dir = os.path.join("..", "outputs", "embeddings")
saved_data = torch.load(os.path.join(saving_dir, "loaders_datasets.pt"))

train_audio = extract_tensors_from_tensordataset(saved_data['train']['audio'])
train_text = extract_tensors_from_tensordataset(saved_data['train']['text'])
train_video = extract_tensors_from_tensordataset(saved_data['train']['video'])

val_audio = extract_tensors_from_tensordataset(saved_data['val']['audio'])
val_text = extract_tensors_from_tensordataset(saved_data['val']['text'])
val_video = extract_tensors_from_tensordataset(saved_data['val']['video'])

test_audio = extract_tensors_from_tensordataset(saved_data['test']['audio'])
test_text = extract_tensors_from_tensordataset(saved_data['test']['text'])
test_video = extract_tensors_from_tensordataset(saved_data['test']['video'])



# File paths
train_json_path = os.path.join("..", "outputs", "preprocessed", "train_data.json")
val_json_path = os.path.join("..", "outputs", "preprocessed", "dev_data.json")
test_json_path = os.path.join("..", "outputs", "preprocessed", "test_data.json")
train_dataset = MultiModalDatasetWithSpeaker(
    audio_data=train_audio,
    text_data=train_text,
    video_data=train_video,
    labels=saved_data['train']['labels'],
    json_path=train_json_path
)

val_dataset = MultiModalDatasetWithSpeaker(
    audio_data=val_audio,
    text_data=val_text,
    video_data=val_video,
    labels=saved_data['val']['labels'],
    json_path=val_json_path,
    speaker_to_id=train_dataset.speaker_to_id 
)

test_dataset = MultiModalDatasetWithSpeaker(
    audio_data=test_audio,
    text_data=test_text,
    video_data=test_video,
    labels=saved_data['test']['labels'],
    json_path=test_json_path,
    speaker_to_id=train_dataset.speaker_to_id 
)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# --------------------------------------------------------------------------------
# Workflow for Training
# --------------------------------------------------------------------------------
# Model parameters
num_classes = 7
num_speakers = len(train_dataset.speaker_to_id)
hidden_dim = 128

# Instantiate the model
model_TransformerMultimodal = TransformerMultimodal(
    text_dim=768,
    audio_dim=768,
    video_dim=768,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    num_speakers=num_speakers,
    speaker_dim=64
)

# --------------------------------------------------------------------------------

from tqdm import tqdm
trained_model = train_model_TF_FUSION_MULTIMODAL(
    model_TransformerMultimodal,
    train_loader,
    val_loader,
    num_epochs=20,
    learning_rate=1e-3,
    device="cpu" if torch.cuda.is_available() else "cpu",
    logfile=os.path.join("..", "logs", "training_logs", "transformer_multimodal_train.log"),
    verbose=False
)