import torch
import logging
from utils.logger import create_logger
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import os, sys
sys.path.append(os.path.abspath(os.path.join('..', '..')))


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
    class_accuracies = []
    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        class_accuracies.append(accuracy)
        if verbose:
            print(f"Accuracy of class {i}: {accuracy:.2f}%")
        logger.info(f"Accuracy of class {i}: {accuracy:.2f}%")
    logger.info(f"Validation Loss: {val_loss / len(val_loader):.4f}")
    logger.info(f"Validation Accuracy: {val_accuracy:.2f}%")
    logger.info(f"Validation Precision: {precision:.2f}")
    logger.info(f"Validation Recall: {recall:.2f}")
    logger.info(f"Validation F1-Score: {f1:.2f}")

    if verbose:
        print(f"Validation Metrics: Loss = {val_loss / len(val_loader):.4f}, Accuracy = {val_accuracy:.2f}%, Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}")


    return val_loss / len(val_loader), val_accuracy, precision, recall, f1

# -----------------------------------------------------------------------------------------------------------------------------------
# Evaluation with test Co-Attention Model
# -----------------------------------------------------------------------------------------------------------------------------------
def test_model_coattention(model, test_loader, device, num_classes=7, verbose=True, logfile="evaluation.log"):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    logfile = create_logger(logfile)
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    with torch.no_grad():
        for batch in test_loader:
            audio, text, video, labels = [item.to(device) for item in batch]

            outputs = model(audio, text, video)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

    test_accuracy = 100 * correct / total if total > 0 else 0

    for i in range(len(all_labels)):
        label = all_labels[i]
        class_total[label] += 1
        class_correct[label] += (all_predictions[i] == label)

    class_accuracies = []
    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        class_accuracies.append(accuracy)
        print(f"Accuracy of class {i}: {accuracy:.2f}%")
        logfile.info(f"Accuracy of class {i}: {accuracy:.2f}%")
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")


    print(f"Test Metrics: Accuracy = {test_accuracy:.2f}%, Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}")
    logfile.info(f"Test Accuracy: {test_accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    return test_accuracy, precision, recall, f1


# -----------------------------------------------------------------------------------------------------------------------------------
# Evaluate Co-Attention Model
# -----------------------------------------------------------------------------------------------------------------------------------
def evaluate_model_coattention(model, val_loader, criterion, device, verbose=True, num_classes=7, logfile="evaluation.log"):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    logfile = create_logger(logfile)


    with torch.no_grad():
        for batch in val_loader:
            audio, text, video, labels = [item.to(device) for item in batch]

            outputs = model(audio, text, video)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())

            for  label, prediction in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1

    # Compute overall accuracy
    val_accuracy = 100 * correct / total if total > 0 else 0
    class_accuracies = []
    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        class_accuracies.append(accuracy)

        if verbose:
            print(f"Accuracy of class {i}: {accuracy:.2f}%")
        logfile.info(f"Accuracy of class {i}: {accuracy:.2f}%")

    # Compute precision, recall, F1
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    if verbose:
        print(f"Validation Metrics: Loss = {val_loss / len(val_loader):.4f}, Accuracy = {val_accuracy:.2f}%, Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}")
    logfile.info(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    return val_loss / len(val_loader), val_accuracy, precision, recall, f1, class_accuracies




def evaluate_model_coattention_graph(model, val_loader, criterion, device, verbose=True, num_classes=7, logfile="evaluation_log_graph_coattention", node_features=None, edge_index=None, edge_type=None):
    model = model.to(device)
    model.eval()

    if node_features is not None:
        node_features = node_features.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)
    if edge_type is not None:
        edge_type = edge_type.to(device)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            audio, text, video, labels, batch_speaker_ids = [item.to(device) for item in batch]

            outputs = model(audio, text, video,  node_features=node_features, edge_index=edge_index, edge_type=edge_type,  batch_speaker_ids=batch_speaker_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    val_accuracy = total_correct / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    if verbose:
        print(f"Val Loss = {total_loss / len(val_loader):.4f}, Val Accuracy = {val_accuracy * 100:.2f}%, Precision = {precision * 100:.2f}%, Recall = {recall * 100:.2f}%, F1 = {f1 * 100:.2f}%")



    logger = create_logger(logfile)

    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    for label, prediction in zip(all_labels, all_predictions):
        class_total[label] += 1
        if label == prediction:
            class_correct[label] += 1

    class_accuracies = []
    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        class_accuracies.append(accuracy)
        if verbose:
            print(f"Accuracy of class {i}: {accuracy:.2f}%")
        logger.info(f"Accuracy of class {i}: {accuracy:.2f}%")
    logger.info(f"Val Loss = {total_loss / len(val_loader):.4f}, Val Accuracy = {val_accuracy:.2f}%, Precision = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}")
    return total_loss / len(val_loader), val_accuracy, precision, recall, f1, total_correct / total_samples