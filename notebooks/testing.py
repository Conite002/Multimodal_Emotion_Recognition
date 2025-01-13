import os
import torch
from torch.utils.data import DataLoader
import sys, os
sys.path.append(os.path.abspath('..'))
from preprocessing.text.preprocess_text import preprocess_text_for_model, load_text_model
from preprocessing.audio.preprocess_audio import preprocess_audio_for_model, load_audio_model, extract_audio
from pipelines.preprocessing.data_pipeline import generate_metadata

import torch.nn as nn
from pipelines.training.training_pipeline import train_model
from models.audio.audio_model import AudioCNNClassifier
from pipelines.training.training_pipeline import evaluate_model



saving_dir = os.path.join("..", "outputs", "embeddings")
saved_data = torch.load(os.path.join(saving_dir, "loaders_datasets.pt"))

train_loaders = {
    'audio': DataLoader(saved_data['train']['audio'], batch_size=32, shuffle=True),
    'text': DataLoader(saved_data['train']['text'], batch_size=32, shuffle=True),
    'video': DataLoader(saved_data['train']['video'], batch_size=32, shuffle=True),
    'label': DataLoader(saved_data['train']['labels'], batch_size=32, shuffle=True)
}

val_loaders = {
    'audio': DataLoader(saved_data['val']['audio'], batch_size=32, shuffle=False),
    'text': DataLoader(saved_data['val']['text'], batch_size=32, shuffle=False),
    'video': DataLoader(saved_data['val']['video'], batch_size=32, shuffle=False),
    'label': DataLoader(saved_data['val']['labels'], batch_size=32, shuffle=False)
}

test_loaders = {
    'audio': DataLoader(saved_data['test']['audio'], batch_size=32, shuffle=False),
    'text': DataLoader(saved_data['test']['text'], batch_size=32, shuffle=False),
    'video': DataLoader(saved_data['test']['video'], batch_size=32, shuffle=False),
    'label': DataLoader(saved_data['test']['labels'], batch_size=32, shuffle=False)
}




# audio_model = AudioCNNClassifier(input_dim=768, num_classes=7)
# train_model_audio = train_model(audio_model, train_loaders["audio"], val_loaders["audio"], num_epochs=5, learning_rate=1e-3, device="cpu")
# test_audio_loss, test_audio_acc = evaluate_model(train_model_audio, test_loaders["audio"], device="cpu", criterion=nn.CrossEntropyLoss())

from models.text.text_model import TextLSTMClassifier

text_model = TextLSTMClassifier(input_dim=768, num_classes=7)

trained_text_model = train_model(
    text_model,
    train_loaders["text"],
    val_loaders["text"],
    num_epochs=5,
    learning_rate=1e-3,
    device="cpu",
    modal="text"
)

test_loss, test_accuracy = evaluate_model(
    trained_text_model,
    test_loaders["text"],
    criterion=nn.CrossEntropyLoss(),
    device="cpu"
)

# print(f"Text Test Loss: {test_loss:.4f}, Text Test Accuracy: {test_accuracy:.2f}%")

# from models.video.video_model import VideoMLPClassifier

# # Define the video model
# video_model = VideoMLPClassifier(input_dim=768, num_classes=7)

# # Train and validate the video model
# trained_video_model = train_model(
#     video_model,
#     train_loaders["video"],
#     val_loaders["video"],
#     num_epochs=5,
#     learning_rate=1e-3,
#     device="cpu",
# )

# # Evaluate the video model on the test set
# test_loss, test_accuracy = evaluate_model(
#     trained_video_model,
#     test_loaders["video"],
#     criterion=nn.CrossEntropyLoss(),
#     device="cpu"
# )

# print(f"Video Test Loss: {test_loss:.4f}, Video Test Accuracy: {test_accuracy:.2f}%")
