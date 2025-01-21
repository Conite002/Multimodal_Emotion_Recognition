import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pipelines.evaluation.evaluation_pipeline import evaluate_model, evaluate_model_coattention
import torch 
import logging
from utils.logger import create_logger


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
def train_model_coattention(model, train_loader, val_loader, num_epochs, learning_rate, device, modal=None, logfile="training.log", verbose=True, num_classes=7):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger = create_logger(logfile)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            # Unpack and move tensors to the device
            audio, text, video, labels = [item.to(device) for item in batch]

            optimizer.zero_grad()
            outputs = model(audio, text, video)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy, precision, recall, f1, accuracies = evaluate_model_coattention(
            model, val_loader, criterion, device, verbose=False, num_classes=num_classes, logfile=logfile
        )

        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):.4f}, "
            f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%, "
            f"Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}"
        )

    return model