import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pipelines.evaluation.evaluation_pipeline import evaluate_model, evaluate_model_coattention
import torch 
import logging
from utils.logger import create_logger
from sklearn.metrics import precision_recall_fscore_support
from pipelines.evaluation.evaluation_pipeline import evaluate_model_coattention_graph
import os, sys

sys.path.append(os.path.abspath(os.path.join('..', '..')))

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, modal=None, logfile="training.log", verbose=True, num_classes=7):
    """
    Train the model using the provided training and validation data loaders.
    """
    logger = create_logger(logfile)

    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}")

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
                
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)

            # if not modal: 
                # inputs = inputs.view(inputs.size(0), -1)
                # inputs = inputs[:, :768]
                # inputs = inputs.mean(dim=1)
            if modal != "video":
                # inputs = inputs.squeeze(1)
                inputs = inputs.unsqueeze(1)
            # 

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy, precision, recall, f1 = evaluate_model(model, val_loader, criterion, device, modal=modal, verbose=verbose, logfile=logfile, num_classes=num_classes)
        # logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss/len(train_loader) :.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")
        # recal, precision, f1 * 100
        recall = recall * 100
        precision = precision * 100
        f1 = f1 * 100
        logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss/len(train_loader) :.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%, Precision = {precision:.2f}%, Recall = {recall:.2f}%, F1 = {f1:.2f}")
        # print(f"Epoch {epoch + 1}: Train Loss = {train_loss/len(train_loader) :.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")

    return model


# --------------------------------------------------------------------------------------------------------------------------
# train_model_coattention
# --------------------------------------------------------------------------------------------------------------------------
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

def train_coattention_graph(model, train_loader, val_loader, original_data=None, num_epochs=10, lr=1e-3, logfile="training_log_graph_coattention.log", num_classes=7, nodes_edges=None, model_name='best_model_graph_coattention.pth', verbose=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    nodes_edges = {key: {k: v.to(device) for k, v in value.items()} for key, value in nodes_edges.items()}

    # train_labels = original_data["train"]["labels"]
    # class_counts = torch.bincount(train_labels)
    # class_weights = 1.0 / (class_counts.float() + 1e-6)  
    # class_weights = class_weights / class_weights.sum()

    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=0.25, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    logfile = os.path.join('..', 'logs', 'training_logs', logfile)
    print(f"Logfile: {logfile}")
    logger = create_logger(logfile)

    best_val_loss = float('inf')
    best_model_path = os.path.join( '..', 'outputs', 'models', model_name)
    print(f"Best model path: {best_model_path}")
    early_stopping_patience = 5
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            audio, text, video, labels, batch_speaker_ids = [item.to(device) for item in batch]
            optimizer.zero_grad()
            
            outputs = model(
                audio, text, video, 
                node_features=nodes_edges['node_features']['train'], 
                edge_index=nodes_edges['edge_index']['train'], 
                edge_type=nodes_edges['edge_type']['train'],
                batch_speaker_ids=batch_speaker_ids
            )

            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_samples += labels.size(0)
        
        train_accuracy = total_correct / total_samples
        val_loss, val_accuracy, precision, recall, f1, accuracies = evaluate_model_coattention_graph(
            model, val_loader, criterion, device, num_classes=num_classes, logfile=logfile,
            node_features=nodes_edges['node_features']['val'], edge_index=nodes_edges['edge_index']['val'], 
            edge_type=nodes_edges['edge_type']['val'],
            verbose=verbose
        )
        
        scheduler.step(val_loss)

        val_accuracy *= 100
        precision *= 100
        recall *= 100
        f1 *= 100

        print(
            f"Epoch => {epoch + 1}: Train Loss = {total_loss / len(train_loader):.4f}, "
            f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%, "
            f"Precision = {precision:.2f}%, Recall = {recall:.2f}%, F1 = {f1:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved at epoch {epoch + 1} with val loss: {val_loss:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"Loaded best model with val loss: {best_val_loss:.4f}")

    return model

class FocalLoss(nn.Module):
    """
    Implémente la Focal Loss pour améliorer la gestion des classes sous-représentées.
    """
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        p_t = torch.exp(-ce_loss)  # Probabilité de la classe correcte
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss  # Modulation

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
