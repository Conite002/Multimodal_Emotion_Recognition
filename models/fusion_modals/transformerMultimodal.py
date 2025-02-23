import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from utils.logger import create_logger



class TransformerMultimodal(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, hidden_dim, num_classes, num_speakers, speaker_dim=64, num_heads=8, num_layers=4, dropout=0.3):
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
        # Encode chaque modalité
        text_features = self.text_encoder(text)
        audio_features = self.audio_encoder(audio)
        video = video.mean(dim=1)
        video_features = self.video_encoder(video)
        
        unique_speakers = sorted(set(speaker_ids.tolist()))
        speaker_id_map = {id_: idx for idx, id_ in enumerate(unique_speakers)}
        mapped_speaker_ids = torch.tensor([speaker_id_map[id_.item()] for id_ in speaker_ids])
        speaker_embeddings = self.speaker_embedding(mapped_speaker_ids)

        text_features = text_features.view(text_features.size(0), -1)  
        audio_features = audio_features.view(audio_features.size(0), -1)  
        video_features = video_features.view(video_features.size(0), -1) 
        speaker_embeddings = speaker_embeddings.view(speaker_embeddings.size(0), -1)

        combined_features = torch.cat([text_features, audio_features, video_features, speaker_embeddings], dim=-1)  
        combined_features = self.projection_layer(combined_features)  
        combined_features = combined_features.unsqueeze(1)  
        attn_output, _ = self.cross_attention(combined_features, combined_features, combined_features)
        
        logits = self.fc(attn_output.squeeze(1)) 
        return logits


from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def validate_model_TF_FUSION_MULTIMODAL(model, val_loader, criterion, device, verbose=True, logfile=None):
    """
    Validate the TransformerMultimodal model and compute metrics.
    """
    logfile = create_logger(logfile)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            text, audio, video, speaker_ids, labels = [item.to(device) for item in batch]
            outputs = model(text, audio, video, speaker_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

    unique_classes = sorted(set(all_labels + all_predictions))


    # Compute overall metrics
    val_accuracy = 100 * correct / total
    unique_classes = sorted(set(all_labels + all_predictions))
    precision = precision_score(all_labels, all_predictions, labels=unique_classes, average="weighted", zero_division=0)*100
    recall = recall_score(all_labels, all_predictions, labels=unique_classes, average="weighted", zero_division=0)*100
    f1 = f1_score(all_labels, all_predictions,labels=unique_classes, average="weighted", zero_division=0)*100
    class_report = classification_report(
            all_labels,
            all_predictions,
            labels=unique_classes,
            target_names=[f"Class {i}" for i in range(len(set(all_labels)))],
            output_dict=True
     )
    # print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%, Precision: {precision:.4f}%, Recall: {recall:.4f}%, F1: {f1:.4f}%")
    # Per-class metrics
    class_metrics = classification_report(all_labels, all_predictions, target_names=[f"Class {i}" for i in range(len(set(all_labels)))], output_dict=True)
    for class_name, metrics in class_report.items():
        if class_name.startswith('Class'):
            if verbose:
                print(f"{class_name}: Precision: {metrics['precision']*100:.2f}%, Recall: {metrics['recall']*100:.2f}%, F1-Score: {metrics['f1-score']*100:.2f}%")
            logfile.info(f"{class_name}: Precision: {metrics['precision']*100:.2f}%, Recall: {metrics['recall']*100:.2f}%, F1-Score: {metrics['f1-score']*100:.2f}%")
    return val_loss / len(val_loader), val_accuracy, class_metrics, 



def test_model_TF_FUSION_MULTIMODAL(model, test_loader, device):
    """
    Test the TransformerMultimodal model and evaluate per-class performance.
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            text, audio, video, speaker_ids, labels = [item.to(device) for item in batch]

            outputs = model(text, audio, video, speaker_ids)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

    # Compute overall metrics
    accuracy = 100 * correct / total
    unique_classes = sorted(set(all_labels + all_predictions))
    precision = precision_score(all_labels, all_predictions, labels=unique_classes, average="weighted")
    recall = recall_score(all_labels, all_predictions,labels=unique_classes, average="weighted")
    f1 = f1_score(all_labels, all_predictions,labels=unique_classes, average="weighted")

    # Per-class metrics
    unique_classes = sorted(set(all_labels + all_predictions))
    class_metrics = classification_report(all_labels, all_predictions, labels=unique_classes, target_names=[f"Class {i}" for i in range(len(set(all_labels)))], output_dict=True, zero_division=0)
    
    print(f"Test Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    for class_name, metrics in class_metrics.items():
        if class_name.startswith('Class'):
            print(f"{class_name}: Precision: {metrics['precision']*100:.2f}%, Recall: {metrics['recall']*100:.2f}%, F1-Score: {metrics['f1-score']* 100:.2f}%")

                                                                                                                               
    return accuracy, class_metrics


def train_model_TF_FUSION_MULTIMODAL(model, train_loader, val_loader, num_epochs, learning_rate, device,verbose=True, logfile=None):
    """
    Train the TransformerMultimodal model with speaker embeddings.
    """

    log_file = create_logger(logfile)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            processed_batch = []
            for idx, item in enumerate(batch):
                if isinstance(item, list): 
                    item = torch.cat([torch.tensor(part, dtype=torch.float32) for part in item], dim=0)
                elif isinstance(item, torch.Tensor):
                    item = item  
                else:
                    raise TypeError(f"Unsupported type for item {idx}: {type(item)}")
                processed_batch.append(item.to(device)) 
            text, audio, video, speaker_ids, labels = processed_batch

            # Forward pass
            optimizer.zero_grad()
            outputs = model(text, audio, video, speaker_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy, val_metrics = validate_model_TF_FUSION_MULTIMODAL(model, val_loader, criterion, device, verbose=verbose, logfile=logfile)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        log_file.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        log_file.info(f"Validation Metrics: {val_metrics}")
    return model
