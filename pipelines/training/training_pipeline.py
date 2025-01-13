import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pipelines.evaluation.evaluation_pipeline import evaluate_model
import torch 


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, modal=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
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
            print(f"Inputs shape: {inputs.shape}")

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print(f"outputs shape: {outputs.shape}")   
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")

    return model
