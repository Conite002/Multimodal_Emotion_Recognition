import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pipelines.evaluation.evaluation_pipeline import evaluate_model, evaluate_model_coattention
import torch 
import logging
from utils.logger import create_logger
from sklearn.metrics import precision_recall_fscore_support
from pipelines.evaluation.evaluation_pipeline import evaluate_model_coattention_graph


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

def train_model_coattention(model, train_loader, val_loader, num_epochs, learning_rate, device=None, modal=None, logfile="training.log", verbose=True, num_classes=7, save_model=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    logger = create_logger(logfile)

    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_patience = 10
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
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

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):.4f}, "
            f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%, "
            f"Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), save_model)
        logger.info("Best model saved with validation loss: {:.4f}".format(best_val_loss))

    return model


def train_coattention_graph(model, train_loader, val_loader, num_epochs=10, lr=1e-3, logfile="training_log_graph_coattention.log", num_classes=7, model_name='best_model_graph_coattention.pth', verbose=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    logger = create_logger(logfile)
    best_val_loss = float('inf')
    best_model_state = None
    early_stopping_patience = 10
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        logger.info(f"Starting Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            audio, text, video, graph_features, labels = [item.to(device) for item in batch]
            optimizer.zero_grad()
            outputs = model(audio, text, video, graph_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_samples += labels.size(0)
        
        val_loss, val_accuracy, precision, recall, f1, accuracies = evaluate_model_coattention_graph(
            model, val_loader, criterion, device, verbose=False, num_classes=num_classes, logfile=logfile
        )
        train_accuracy = total_correct / total_samples
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}: Train Loss = {total_loss / len(train_loader):.4f}, "
            f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%, "
            f"Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), model_name)
        logger.info("Best model saved with validation loss: {:.4f}".format(best_val_loss))

    return model
