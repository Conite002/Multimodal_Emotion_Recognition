import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Classe Dataset pour générer des données factices
class MultiModalDataset(Dataset):
    def __init__(self, num_samples, seq_len, text_dim, audio_dim, video_dim, num_classes, num_speakers):
        self.text_data = torch.rand(num_samples, seq_len, text_dim)
        self.audio_data = torch.rand(num_samples, seq_len, audio_dim)
        self.video_data = torch.rand(num_samples, seq_len, video_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.speaker_ids = torch.randint(0, num_speakers, (num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_data[idx], self.audio_data[idx], self.video_data[idx], self.speaker_ids[idx], self.labels[idx]

# Classe du modèle
class TransformerMultimodal(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, hidden_dim, num_classes, speaker_dim=64, num_speakers=10, num_heads=8, num_layers=4, dropout=0.3):
        super(TransformerMultimodal, self).__init__()
        
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=text_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.audio_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=audio_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.video_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=video_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_dim)
        self.projection_layer = nn.Linear(text_dim + audio_dim + video_dim + speaker_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, text, audio, video, speaker_ids):
        text_features = self.text_encoder(text).mean(dim=1)  # [batch_size, text_dim]
        audio_features = self.audio_encoder(audio).mean(dim=1)  # [batch_size, audio_dim]
        video_features = self.video_encoder(video).mean(dim=1)  # [batch_size, video_dim]
        speaker_embeddings = self.speaker_embedding(speaker_ids)  # [batch_size, speaker_dim]
        
        print(f"text_features: {text_features.shape} | audio_features: {audio_features.shape} | video_features: {video_features.shape} | speaker_embeddings: {speaker_embeddings.shape}")
        combined_features = torch.cat([text_features, audio_features, video_features, speaker_embeddings], dim=-1)
        combined_features = self.projection_layer(combined_features)  # [batch_size, hidden_dim]
        combined_features = combined_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        attn_output, _ = self.cross_attention(combined_features, combined_features, combined_features)
        logits = self.fc(attn_output.squeeze(1))  # [batch_size, num_classes]
        return logits

# Fonction d'entraînement
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            text, audio, video, speaker_ids, labels = [item.to(device) for item in batch]

            optimizer.zero_grad()
            outputs = model(text, audio, video, speaker_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        validate_model(model, val_loader, device, criterion)

# Fonction de validation
def validate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            text, audio, video, speaker_ids, labels = [item.to(device) for item in batch]
            outputs = model(text, audio, video, speaker_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Instanciation des paramètres
text_dim = 768
audio_dim = 768
video_dim = 768
hidden_dim = 128
num_classes = 7
num_speakers = 10
seq_len = 16
batch_size = 32
num_samples = 1000
num_epochs = 5
learning_rate = 1e-3

# Création des datasets et loaders
train_dataset = MultiModalDataset(num_samples, seq_len, text_dim, audio_dim, video_dim, num_classes, num_speakers)
val_dataset = MultiModalDataset(num_samples // 10, seq_len, text_dim, audio_dim, video_dim, num_classes, num_speakers)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Instanciation du modèle
model = TransformerMultimodal(text_dim, audio_dim, video_dim, hidden_dim, num_classes, num_speakers=num_speakers)

# Entraînement
train_model(model, train_loader, val_loader, num_epochs, learning_rate, device="cuda" if torch.cuda.is_available() else "cpu")
