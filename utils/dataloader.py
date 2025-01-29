import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from preprocessing.audio.preprocess_audio import preprocess_audio_for_model, load_audio_model
from preprocessing.video.preprocess_video import preprocess_video_for_model, load_vit_model
from preprocessing.text.preprocess_text import preprocess_text_for_model, load_text_model
import numpy as np
from sklearn.decomposition import PCA

import os

def reduce_dimensionality(features, target_dim):
    features_np = np.array(features)
    if features_np.ndim == 3:
        num_samples, time_steps, feature_dim = features_np.shape
        features_np = features_np.reshape(num_samples, -1)
    pca = PCA(n_components=target_dim)
    reduced_features = pca.fit_transform(features_np)

    return [reduced for reduced in reduced_features]

def create_data_loaders(train_path, val_path, dims, batch_size=32, reduce_labels=None):
    """
    Create DataLoaders for training and validation with dynamic dimensions.

    Args:
        train_path (str): Path to training data JSON.
        val_path (str): Path to validation data JSON.
        dims (dict): Dictionary specifying dimensions for audio, text, and video.
                     Example: {"audio": 100, "text": 768, "video": 512}
        batch_size (int): Batch size for DataLoaders.

    Returns:
        dict: Training and validation DataLoaders for each modality.
        dict: Label mapping for class indices.
    """
    def load_data_with_preprocessing(json_path, output_dir, save_every=200, reduce_labels=None):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        print(f"Number of samples in {json_path}: {len(data)}")
        if reduce_labels is not None:
            data = [item for item in data if item["label"] in reduce_labels]


        print(f"Number of samples in {json_path}: {len(data)}")
        import pandas as pd
        df = pd.DataFrame(data)
        labels_unique = df["label"].unique()
        print(f"Unique labels: {labels_unique}")


        audio_embeddings, text_embeddings, video_embeddings, labels = [], [], [], []
        processor_audio, model_audio = load_audio_model()
        tokenizer_text, model_text = load_text_model()
        feature_extractor_video, model_video = load_vit_model()

        os.makedirs(output_dir, exist_ok=True)
        for i, item in enumerate(tqdm(data, desc=f"Processing {json_path}")):
                try:
                    audio_path = item["audio"]
                    audio_embedding = preprocess_audio_for_model(
                        audio_path, processor=processor_audio, model=model_audio, target_sample_rate=16000, target_duration=8.0
                    )
                    audio_embeddings.append(audio_embedding)

                    text = item["text"]
                    text_embedding = preprocess_text_for_model(
                        text, tokenizer=tokenizer_text, model=model_text, max_length=128
                    )
                    text_embeddings.append(text_embedding)

                    video_path = item["video"]
                    video_embedding = preprocess_video_for_model(
                        video_path, feature_extractor=feature_extractor_video, model=model_video, num_frames=16, frame_size=(224, 224)
                    )
                    video_embeddings.append(video_embedding)

                    labels.append(item["label"])

                except Exception as e:
                    print(f"Error processing row {i}: {e}")

                if (i + 1) % save_every == 0 or (i + 1) == len(data): 
                    torch.save({
                        "audio": audio_embeddings,
                        "text": text_embeddings,
                        "video": video_embeddings,
                        "labels": labels,
                    }, os.path.join(output_dir, f"progress_{i+1}.pt"))
                    print(f"Saved progress after {i+1} rows.")
        # audio_embeddings = reduce_dimensionality(audio_embeddings, dims["audio"])
        # text_embeddings = reduce_dimensionality(text_embeddings, dims["text"])
        # video_embeddings = reduce_dimensionality(video_embeddings, dims["video"])
        return audio_embeddings, video_embeddings, text_embeddings, labels
    

    train_audio, train_video, train_text, train_labels = load_data_with_preprocessing(json_path=train_path, output_dir="../outputs/preprocessed/train", reduce_labels=reduce_labels)
    val_audio, val_video, val_text, val_labels = load_data_with_preprocessing(json_path=val_path, output_dir="../outputs/preprocessed/val", reduce_labels=reduce_labels)

    label_mapping = {label: idx for idx, label in enumerate(set(train_labels + val_labels))}
    train_labels = torch.tensor([label_mapping[label] for label in train_labels], dtype=torch.long)
    val_labels = torch.tensor([label_mapping[label] for label in val_labels], dtype=torch.long)

    train_audio_tensors = torch.tensor(train_audio, dtype=torch.float32)
    train_video_tensors = torch.tensor(train_video, dtype=torch.float32)
    train_text_tensors = torch.tensor(train_text, dtype=torch.float32)

    val_audio_tensors = torch.tensor(val_audio, dtype=torch.float32)
    val_text_tensors = torch.tensor(val_text, dtype=torch.float32)
    val_video_tensors = torch.tensor(val_video, dtype=torch.float32)

    train_loaders = {
        "audio": DataLoader(TensorDataset(train_audio_tensors, train_labels), batch_size=batch_size, shuffle=True),
        "text": DataLoader(TensorDataset(train_text_tensors, train_labels), batch_size=batch_size, shuffle=True),
        "video": DataLoader(TensorDataset(train_video_tensors, train_labels), batch_size=batch_size, shuffle=True),
    }
    val_loaders = {
        "audio": DataLoader(TensorDataset(val_audio_tensors, val_labels), batch_size=batch_size),
        "text": DataLoader(TensorDataset(val_text_tensors, val_labels), batch_size=batch_size),
        "video": DataLoader(TensorDataset(val_video_tensors, val_labels), batch_size=batch_size),
    }
    return train_loaders, val_loaders, label_mapping

def extract_tensors_from_tensordataset(tensor_dataset):
    features = []
    for item in tensor_dataset:
        features.append(item[0])
    return torch.stack(features)


from torch.utils.data import Dataset

class MultimodalGraphDataset(Dataset):
    def __init__(self, audio, text, video, labels, graph_features):
        self.audio = audio
        self.text = text
        self.video = video
        self.labels = labels
        self.graph_features = graph_features    
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.audio[idx], self.text[idx], self.video[idx], self.graph_features[idx], self.labels[idx]
    
def create_dataloader_with_graph_features(data):
    train_dataset = MultimodalGraphDataset(
        audio=data["train"]["audio"],
        text=data["train"]["text"],
        video=data["train"]["video"],
        labels=data["train"]["labels"],
        graph_features=data["train"]["graph_features"]
    )
    val_dataset = MultimodalGraphDataset(
        audio=data["val"]["audio"],
        text=data["val"]["text"],
        video=data["val"]["video"],
        labels=data["val"]["labels"],
        graph_features=data["val"]["graph_features"]
    )
    test_dataset = MultimodalGraphDataset(
        audio=data["test"]["audio"],
        text=data["test"]["text"],
        video=data["test"]["video"],
        labels=data["test"]["labels"],
        graph_features=data["test"]["graph_features"]
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    return train_loader, val_loader, test_loader