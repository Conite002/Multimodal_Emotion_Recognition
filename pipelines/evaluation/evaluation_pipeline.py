import torch
import logging
from utils.logger import create_logger


import torch
import logging
from utils.logger import create_logger
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, val_loader, criterion, device, num_classes=7, modal=None, logfile="evaluation.log", verbose=True):
    """
    Evaluate the model on validation data.

    Args:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation data loader.
        criterion: Loss function.
        device (str): Device to evaluate on ('cpu' or 'cuda').

    Returns:
        tuple: (validation loss, validation accuracy, precision, recall, f1-score)
    """
    logger = create_logger(logfile)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)

            inputs, labels = inputs.to(device), labels.to(device)
            if modal != "video":
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for precision, recall, and F1
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                class_correct[label] += (predicted[i].item() == label)

    # Calculate overall metrics
    val_accuracy = 100 * correct / total if total > 0 else 0
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    # Log and print metrics
    logger.info(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    logger.info(f"Validation Accuracy: {val_accuracy:.2f}%")
    logger.info(f"Validation Precision: {precision:.2f}")
    logger.info(f"Validation Recall: {recall:.2f}")
    logger.info(f"Validation F1-Score: {f1:.2f}")

    if verbose:
        print(f"Validation Metrics: Loss = {val_loss / len(val_loader):.4f}, Accuracy = {val_accuracy:.2f}%, Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}")

    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        if verbose:
            print(f"Accuracy of class {i}: {accuracy:.2f}%")
        logger.info(f"Accuracy of class {i}: {accuracy:.2f}%")

    return val_loss / len(val_loader), val_accuracy, precision, recall, f1

