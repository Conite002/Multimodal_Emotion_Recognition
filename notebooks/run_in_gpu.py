import sys, os
sys.path.append(os.path.abspath('..'))
from preprocessing.text.preprocess_text import preprocess_text_for_model, load_text_model
from preprocessing.audio.preprocess_audio import preprocess_audio_for_model, load_audio_model, extract_audio
import torch.nn as nn
from pipelines.training.training_pipeline import train_model
from models.audio.audio_model import AudioCNNClassifier
from pipelines.training.training_pipeline import evaluate_model


from pipelines.preprocessing.data_pipeline import generate_metadata
from utils.dataloader import create_data_loaders

train_data_json_path = os.path.join("..", "outputs", "preprocessed", "train_data.json")
dev_data_json_path = os.path.join("..", "outputs", "preprocessed", "dev_data.json")
test_data_json_path = os.path.join("..", "outputs", "preprocessed", "test_data.json")

dims = {"audio": 100, "text": 768, "video": 512}
train_loaders, val_loaders, label_mapping = create_data_loaders(train_data_json_path, dev_data_json_path, dims, batch_size=32)
test_loaders, _, _ = create_data_loaders(test_data_json_path, test_data_json_path, dims, batch_size=32)


# total length of the dataset
print(len(train_loaders["audio"].dataset))
# total length of labels
print(len(train_loaders["audio"].dataset.tensors[1]))
train_loaders["audio"].dataset.tensors[1]

print(train_loaders, val_loaders, test_loaders)

print(f"================================SAVING================================")
# Save train, val, and test datasets
saving_dir = os.path.join("..", "outputs", "embeddings")
import torch

torch.save({
    'train': {
        'audio': train_loaders['audio'].dataset,
        'text': train_loaders['text'].dataset,
        'video': train_loaders['video'].dataset,
        'labels': train_loaders['audio'].dataset.tensors[1]
    },
    'val': {
        'audio': val_loaders['audio'].dataset,
        'text': val_loaders['text'].dataset,
        'video': val_loaders['video'].dataset,
        'labels': val_loaders['audio'].dataset.tensors[1]
    },
    'test': {
        'audio': test_loaders['audio'].dataset,
        'text': test_loaders['text'].dataset,
        'video': test_loaders['video'].dataset,
        'labels': test_loaders['audio'].dataset.tensors[1]
    }
}, os.path.join(saving_dir, "loaders_datasets.pt"))
