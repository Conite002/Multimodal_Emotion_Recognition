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
from utils.data_augmentation import create_data_loaders_augmented


train_data_json_path = os.path.join("..", "outputs", "preprocessed", "train_data.json")
dev_data_json_path = os.path.join("..", "outputs", "preprocessed", "dev_data.json")
test_data_json_path = os.path.join("..", "outputs", "preprocessed", "test_data.json")

train_loaders = create_data_loaders_augmented(train_data_json_path, batch_size=32)

