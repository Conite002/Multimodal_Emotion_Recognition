import os
import torch
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.abspath('..'))
from preprocessing.text.preprocess_text import preprocess_text_for_model, load_text_model
from preprocessing.audio.preprocess_audio import preprocess_audio_for_model, load_audio_model, extract_audio
from pipelines.preprocessing.data_pipeline import generate_metadata
import numpy as np
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

# ---------------------------------------------------------------------------------------------------------------------------------

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------------------------------------------------------------------------------------------------------------
def create_temporal_windows(data, labels, window_size=5):
    """
    Crée des fenêtres temporelles autour de chaque sample.
    :param data: Tensor de taille (num_samples, feature_dim)
    :param labels: Tensor de labels associés (num_samples,)
    :param window_size: Taille de la fenêtre temporelle
    :return: data_windows, labels
    """
    half_window = window_size // 2
    num_samples, feature_dim = data.size()
    
    padded_data = torch.cat([
        torch.zeros((half_window, feature_dim)),
        data,
        torch.zeros((half_window, feature_dim)) 
    ], dim=0)
    
    data_windows = []
    for i in range(num_samples):
        window = padded_data[i:i+window_size]
        data_windows.append(window)
    
    return torch.stack(data_windows), labels

# ---------------------------------------------------------------------------------------------------------------------------------
class TemporalAttentionModel(nn.Module):
    def __init__(self, input_dim_audio, input_dim_video, input_dim_text, hidden_dim, num_heads=4, num_layers=2, dropout_rate=0.3):
        super(TemporalAttentionModel, self).__init__()

        self.audio_projection = nn.Linear(input_dim_audio, 100)  
        self.video_projection = nn.Linear(input_dim_video, 512)
        
        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=100, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout_rate),
            num_layers=num_layers
        )
        self.video_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout_rate),
            num_layers=num_layers
        )
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout_rate),
            num_layers=num_layers
        )

        self.fusion = nn.Linear(100 + 512 + 768, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 7)  
        )

    def forward(self, audio, video, text):
        audio = audio.to(self.audio_projection.weight.device)  
        video = video.to(self.video_projection.weight.device)
        text = text.to(self.text_transformer.layers[0].linear1.weight.device) 
        
        audio = self.audio_projection(audio)
        video = self.video_projection(video)
        audio = self.audio_transformer(audio.permute(1, 0, 2))
        video = self.video_transformer(video.permute(1, 0, 2))
        text = self.text_transformer(text.permute(1, 0, 2))

        audio = audio.mean(dim=0)
        video = video.mean(dim=0)
        text = text.mean(dim=0)

        combined = torch.cat([audio, video, text], dim=-1)
        combined = self.fusion(combined)

        output = self.classifier(combined)
        return output


#-----------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        """
        Module pour appliquer une attention pondérée sur la dimension temporelle.
        Args:
            input_dim (int): Taille des features (embedding size).
        """
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim) 
        self.key = nn.Linear(input_dim, input_dim)    
        self.value = nn.Linear(input_dim, input_dim)  
        self.scale = input_dim ** 0.5

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor de taille (batch_size, seq_length, feature_dim)
        Returns:
            torch.Tensor: Tensor réduit avec attention (batch_size, feature_dim)
        """
        Q = self.query(x) 
        K = self.key(x)   
        V = self.value(x) 

        # Calcul de l'attention : scores = softmax(Q * K^T / sqrt(d))
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  
        attn_weights = F.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_weights, V)

        context = context.mean(dim=1) 
        return context

class VideoAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VideoAttention, self).__init__()
        self.temporal_attention = TemporalAttention(input_dim)
        self.fc = nn.Linear(input_dim, hidden_dim) 

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Vidéo Tensor de taille (batch_size, seq_length, feature_dim)
        Returns:
            torch.Tensor: Tensor réduit avec attention (batch_size, hidden_dim)
        """
        context = self.temporal_attention(x)
        context = self.fc(context) 
        return context

#-----------------------------------------------------------------------------------------------------------------------------

class MultiModalTemporalDataset(torch.utils.data.Dataset):
    def __init__(self, audio, video, text, labels):
        self.audio = audio
        self.video = video
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.audio[idx], self.video[idx], self.text[idx], self.labels[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_attention_module = VideoAttention(input_dim=768, hidden_dim=512)
video_attention_module = video_attention_module.to(device) 

reduced_video_data_train = torch.stack([
    video_attention_module(video_sample.unsqueeze(0).to(device)).squeeze(0).cpu()
    for video_sample in train_dataset.video_data
])

reduced_video_data_val = torch.stack([
    video_attention_module(video_sample.unsqueeze(0).to(device)).squeeze(0).cpu()
    for video_sample in val_dataset.video_data
])

reduced_video_data_test = torch.stack([
    video_attention_module(video_sample.unsqueeze(0).to(device)).squeeze(0).cpu()
    for video_sample in test_dataset.video_data
])

train_dataset.video_data = reduced_video_data_train
print("Train video data shape after reduction:", train_dataset.video_data.shape)
val_dataset.video_data = reduced_video_data_val
print("Validation video data shape after reduction:", val_dataset.video_data.shape)  

test_dataset.video_data = reduced_video_data_test
print("Test video data shape after reduction:", test_dataset.video_data.shape)

#---------------------------------------------------------------------------------------------------------------------
window_size = 5

audio_windows_train, labels_train = create_temporal_windows(train_dataset.audio_data, train_dataset.labels, window_size=window_size)
video_windows_train, _ = create_temporal_windows(train_dataset.video_data, train_dataset.labels, window_size=window_size)
text_windows_train, _ = create_temporal_windows(train_dataset.text_data, train_dataset.labels, window_size=window_size)

audio_windows_val, labels_val = create_temporal_windows(val_dataset.audio_data, val_dataset.labels, window_size=window_size)
video_windows_val, _ = create_temporal_windows(val_dataset.video_data, val_dataset.labels, window_size=window_size)
text_windows_val, _ = create_temporal_windows(val_dataset.text_data, val_dataset.labels, window_size=window_size)

audio_windows_test, labels_test = create_temporal_windows(test_dataset.audio_data, test_dataset.labels, window_size=window_size)
video_windows_test, _ = create_temporal_windows(test_dataset.video_data, test_dataset.labels, window_size=window_size)
text_windows_test, _ = create_temporal_windows(test_dataset.text_data, test_dataset.labels, window_size=window_size)


train_dataset = MultiModalTemporalDataset(audio_windows_train, video_windows_train, text_windows_train, labels_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = MultiModalTemporalDataset(audio_windows_val, video_windows_val, text_windows_val, labels_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = MultiModalTemporalDataset(audio_windows_test, video_windows_test, text_windows_test, labels_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------------------------------------------------------------------------------------------------------
model = TemporalAttentionModel(
    input_dim_audio=768,  
    input_dim_video=512,  
    input_dim_text=768,   
    hidden_dim=512,       
    num_heads=4,          
    num_layers=2,         
    dropout_rate=0.3      
)
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for audio, video, text, labels in train_loader:
            audio, video, text, labels = audio.to(device), video.to(device), text.to(device), labels.to(device)

            optimizer.zero_grad()  
            outputs = model(audio, video, text)  
            loss = criterion(outputs, labels)  
            loss.backward()

            optimizer.step()  

            train_loss += loss.item()
            
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for audio, video, text, labels in dataloader:
            audio, video, text, labels = audio.to(device), video.to(device), text.to(device), labels.to(device)
            outputs = model(audio, video, text)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return total_loss / len(dataloader), accuracy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,         
    learning_rate=1e-3,    
    device=device          
)
