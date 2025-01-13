import torch
import logging

def evaluate_model(model, val_loader, criterion, device, num_classes=7, modal=None, logfile="evaluation.log"):
    """
    Evaluate the model on validation data.

    Args:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation data loader.
        criterion: Loss function.
        device (str): Device to evaluate on ('cpu' or 'cuda').

    Returns:
        tuple: (validation loss, validation accuracy)
    """
    logging.basicConfig(
        filename=logfile,  # Use the logfile passed as a parameter
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]
    with torch.no_grad():
        for inputs, labels in val_loader:
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=1)
            
            # inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            if modal != "video":
                inputs = inputs.unsqueeze(1)
                
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(num_classes):
                label = labels[i].item()
                class_total[label] += 1
                class_correct[label] += (predicted[i].item() == label)


    val_accuracy = 100 * correct / total if total > 0 else 0

    for i in range(num_classes):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Accuracy of class {i}: {accuracy:.2f}%")
        logging.info(f"Accuracy of class {i}: {accuracy:.2f}%")

    return val_loss/ len(val_loader), val_accuracy
