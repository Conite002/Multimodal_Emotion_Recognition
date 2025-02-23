import json
import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as transforms
import torchvision.io as io
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import os
import multiprocessing as mp
from collections import Counter
import random

# Your custom imports
from preprocessing.audio.preprocess_audio import preprocess_audio_for_model, load_audio_model
from preprocessing.video.preprocess_video import (
    preprocess_video_for_model, 
    load_vit_model,
    preprocess_video_for_visualbert, 
    load_visualbert_model
)
from preprocessing.text.preprocess_text import preprocess_text_for_model, load_text_model

# Global references (populated once per process)
processor_audio, model_audio = None, None
tokenizer_text, model_text = None, None
feature_extractor_video, model_video = None, None

################################################################################
#                         AUDIO AUGMENTATION
################################################################################
def augment_audio(audio_path):
    """
    Load audio from `audio_path`, apply a random transform,
    return the augmented waveform (tensor) and sample rate.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Erreur chargement audio {audio_path}: {e}")
        return None, None

    choice = random.choice(["noise", "pitch", "speed", "shift"])
    if choice == "noise":
        noise = torch.randn_like(waveform) * 0.005
        waveform += noise
    elif choice == "pitch":
        pitch_shift = T.PitchShift(sr, n_steps=random.uniform(-2, 2))
        waveform = pitch_shift(waveform)
    elif choice == "speed":
        speed_factor = random.uniform(0.9, 1.1)
        resampler = T.Resample(sr, int(sr * speed_factor))
        waveform = resampler(waveform)
    elif choice == "shift":
        shift = np.random.randint(0, waveform.shape[1] // 10)
        waveform = torch.roll(waveform, shifts=shift, dims=1)

    return waveform, sr

################################################################################
#                         TEXT AUGMENTATION
################################################################################
def augment_text(text):
    """
    Use nlpaug's SynonymAug or return text if augmentation fails.
    """
    try:
        from nlpaug.augmenter.word import SynonymAug
        aug = SynonymAug(aug_src='wordnet')
        return aug.augment(text)
    except:
        return text

################################################################################
#                         VIDEO AUGMENTATION
################################################################################
def augment_video(video_path):
    """
    Load video from `video_path`, apply random flip / color jitter,
    return the augmented frames (tensor) in shape [T, C, H, W].
    """
    if not os.path.exists(video_path):
        print(f"Vidéo introuvable: {video_path}")
        return None

    try:
        video, _, _ = io.read_video(video_path, pts_unit="sec")
        video = video.permute(0, 3, 1, 2).float() / 255.0
    except Exception as e:
        print(f"Erreur chargement vidéo {video_path}: {e}")
        return None

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    return transform(video)

################################################################################
#                         AUGMENT A SINGLE ITEM
################################################################################
def augment_item(item):
    """
    Takes a data sample (with paths for audio/video, raw text),
    performs in-memory augmentation, and returns a new dict with
    'waveform', 'sample_rate', 'text_aug', 'video_frames', etc.
    """
    # Audio
    audio_waveform, sr = augment_audio(item["audio"])
    if audio_waveform is None or sr is None:
        return None

    # Text
    text_aug = augment_text(item["text"])
    if text_aug is None:
        return None

    # Video
    video_frames = augment_video(item["video"])
    if video_frames is None:
        return None

    new_item = dict(item)
    new_item["waveform"] = audio_waveform  # Tensor
    new_item["sample_rate"] = sr
    new_item["text_aug"] = text_aug        # Possibly string or list of tokens
    new_item["video_frames"] = video_frames  # Tensor [T, C, H, W]
    return new_item

################################################################################
#                OVERSAMPLING UNDERREPRESENTED CLASSES
################################################################################
def replicate_underrepresented_classes(data, factor=1.0, classes_to_augment=None):
    """
    Oversample underrepresented classes up to (max_class_count * factor).
    Instead of calling `augment_item` here, we simply replicate items
    and mark them with `item["need_aug"] = True` so augmentation happens
    in parallel inside process_sample().
    """
    label_counts = Counter(item["label"] for item in data)
    if not label_counts:
        return data

    max_count = max(label_counts.values())
    new_data = []

    for label, count in tqdm(label_counts.items(), desc="Replicating classes"):
        items_of_label = [d for d in data if d["label"] == label]

        if classes_to_augment is not None and label not in classes_to_augment:
            new_data.extend(items_of_label)
            continue

        target = int(max_count * factor)
        if count < target:
            needed = target - count
            for _ in tqdm(range(needed), desc=f"Replicating '{label}'", leave=False):
                original = random.choice(items_of_label)
                new_item = dict(original)
                new_item["need_aug"] = True
                new_data.append(new_item)

        new_data.extend(items_of_label)

    return new_data

################################################################################
#           MULTIPROCESSING INITIALIZATION & SAMPLE PROCESSING
################################################################################
def init_worker():
    """
    Initializer for each process: load the models once per process.
    This ensures each process has its own model copies.
    """
    global processor_audio, model_audio, tokenizer_text, model_text, feature_extractor_video, model_video
    processor_audio, model_audio = load_audio_model()
    tokenizer_text, model_text = load_text_model()
    feature_extractor_video, model_video = load_vit_model()

def process_sample(item):
    """
    Process a single data sample.
    - If item contains 'waveform'/'video_frames', we assume augmentation was done
      and pass them directly to the model.
    - Otherwise, we use the file paths as usual.

    Returns a tuple of (audio_embedding, text_embedding, video_embedding, label, speaker).
    """
    try:
        if item.get("need_aug", False):
            item = augment_item(item)
            if item is None:
                return None
        # AUDIO
        if "waveform" in item and "sample_rate" in item:
            audio_embedding = preprocess_audio_for_model(
                audio_path=None,
                processor=processor_audio,
                model=model_audio,
                target_sample_rate=item["sample_rate"],
                target_duration=13,
                waveform_in_memory=item["waveform"]
            )
        else:
            audio_embedding = preprocess_audio_for_model(
                audio_path=item["audio"],
                processor=processor_audio,
                model=model_audio,
                target_sample_rate=16000,
                target_duration=13
            )

        # TEXT
        if "text_aug" in item:
            text_embedding = preprocess_text_for_model(
                item["text_aug"],
                tokenizer=tokenizer_text,
                model=model_text,
                max_length=70
            )
        else:
            text_embedding = preprocess_text_for_model(
                item["text"],
                tokenizer=tokenizer_text,
                model=model_text,
                max_length=70
            )

        # VIDEO
        if "video_frames" in item:
            video_embedding = preprocess_video_for_model(
                video_path=None,
                image_processor=feature_extractor_video,
                model=model_video,
                num_frames=200,
                frame_size=(224, 224),
                frames_in_memory=item["video_frames"]
            )
        else:
            video_embedding = preprocess_video_for_model(
                video_path=item["video"],
                image_processor=feature_extractor_video,
                model=model_video,
                num_frames=200,
                frame_size=(224, 224)
            )

        # If video embedding is 2D (num_frames, 768), average-pool it
        if video_embedding is not None and video_embedding.ndim == 2:
            video_embedding = video_embedding.mean(axis=0)

        return (audio_embedding, text_embedding, video_embedding, item["label"], item["speaker"])

    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

################################################################################
#        PARALLEL LOADING + OPTIONAL FRACTION + AUGMENTATION
################################################################################
def load_data_with_preprocessing_parallel(
    json_path,
    output_dir,
    reduce_labels=None,
    data_fraction=1.0,
    augmentation=False,
    augmentation_factor=1.0,
    classes_to_fix=None,
    nber_cpu=1
):

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Number of samples in {json_path}: {len(data)}")
    
    if reduce_labels is not None:
        data = [item for item in data if item["label"] in reduce_labels]
    print(f"Number of samples after label reduction: {len(data)}")
   
    if 0 < data_fraction < 1.0:
        keep_count = int(len(data) * data_fraction)
        data = data[:keep_count]
        print(f"Number of samples after fraction {data_fraction} reduction: {len(data)}")

    # 3) Augment underrepresented classes if requested
    if augmentation and len(data) > 0:
        data = replicate_underrepresented_classes(
            data,
            factor=augmentation_factor,
            classes_to_augment=classes_to_fix
        )
        print(f"Number of samples after augmentation: {len(data)}")

    os.makedirs(output_dir, exist_ok=True)

    nb_cpu = min(mp.cpu_count(), nber_cpu)
    with mp.Pool(processes=nb_cpu, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_sample, data, chunksize=1),
            total=len(data),
            desc=f"Processing {json_path}"
        ))
    results = [r for r in results if r is not None]
    if not results:
        return [], [], [], [], []

    audio_embeddings, text_embeddings, video_embeddings, labels, speakers = zip(*results)

    # 6) Save progress
    torch.save({
        "audio": audio_embeddings,
        "text": text_embeddings,
        "video": video_embeddings,
        "labels": labels,
        "speakers": speakers
    }, os.path.join(output_dir, "progress.pt"))

    return audio_embeddings, text_embeddings, video_embeddings, labels, speakers

################################################################################
#                     CREATE DATA LOADERS
################################################################################
def create_data_loaders(
    train_path,
    val_path,
    dims,
    batch_size=32,
    reduce_labels=None,
    data_fraction=1.0,
    augmentation=False,
    augmentation_factor=1.0,
    classes_to_fix=None,
    nber_cpu=1
):
    """
    1) load_data_with_preprocessing_parallel for train/val
    2) convert to PyTorch tensors
    3) build DataLoaders
    """
    # --- Training set ---
    train_audio, train_text, train_video, train_labels, train_speakers = load_data_with_preprocessing_parallel(
        json_path=train_path,
        output_dir="../outputs/preprocessed/train",
        reduce_labels=reduce_labels,
        data_fraction=data_fraction,
        augmentation=augmentation,
        augmentation_factor=augmentation_factor,
        classes_to_fix=classes_to_fix,
        nber_cpu=nber_cpu
    )
    # --- Validation set ---
    val_audio, val_text, val_video, val_labels, val_speakers = load_data_with_preprocessing_parallel(
        json_path=val_path,
        output_dir="../outputs/preprocessed/val",
        reduce_labels=reduce_labels,
        data_fraction=data_fraction,
        augmentation=False,
        augmentation_factor=augmentation_factor,
        classes_to_fix=classes_to_fix,
        nber_cpu=nber_cpu
    )

    # Create label and speaker mappings
    label_mapping = {lbl: idx for idx, lbl in enumerate(set(train_labels + val_labels))}
    train_label_ids = torch.tensor([label_mapping[x] for x in train_labels], dtype=torch.long)
    val_label_ids   = torch.tensor([label_mapping[x] for x in val_labels],   dtype=torch.long)

    speaker_mapping = {spk: idx for idx, spk in enumerate(set(train_speakers + val_speakers))}
    train_speaker_ids = torch.tensor([speaker_mapping[x] for x in train_speakers], dtype=torch.long)
    val_speaker_ids   = torch.tensor([speaker_mapping[x] for x in val_speakers],   dtype=torch.long)

    # Convert embeddings to Tensors
    def to_tensor(data_list, name="unknown"):
        if not data_list:
            print(f"{name} is empty, returning an empty tensor.")
            return torch.empty(0)
        arr = np.array(data_list)
        return torch.tensor(arr, dtype=torch.float32)

    train_audio_tensors = to_tensor(train_audio, name="train_audio")
    train_text_tensors  = to_tensor(train_text,  name="train_text")
    train_video_tensors = to_tensor(train_video, name="train_video")

    val_audio_tensors   = to_tensor(val_audio,   name="val_audio")
    val_text_tensors    = to_tensor(val_text,    name="val_text")
    val_video_tensors   = to_tensor(val_video,   name="val_video")

    # Build DataLoaders
    train_loaders = {
        "audio":   DataLoader(TensorDataset(train_audio_tensors, train_label_ids),   batch_size=batch_size, shuffle=True),
        "text":    DataLoader(TensorDataset(train_text_tensors, train_label_ids),    batch_size=batch_size, shuffle=True),
        "video":   DataLoader(TensorDataset(train_video_tensors, train_label_ids),   batch_size=batch_size, shuffle=True),
        "speaker": DataLoader(TensorDataset(train_speaker_ids,  train_label_ids),    batch_size=batch_size, shuffle=True),
    }
    val_loaders = {
        "audio":   DataLoader(TensorDataset(val_audio_tensors, val_label_ids),       batch_size=batch_size),
        "text":    DataLoader(TensorDataset(val_text_tensors, val_label_ids),        batch_size=batch_size),
        "video":   DataLoader(TensorDataset(val_video_tensors, val_label_ids),       batch_size=batch_size),
        "speaker": DataLoader(TensorDataset(val_speaker_ids,  val_label_ids),        batch_size=batch_size),
    }

    return train_loaders, val_loaders, label_mapping

################################################################################
#                   OPTIONAL PCA DIMENSION REDUCTION
################################################################################
def reduce_dimensionality(features, target_dim):
    """
    If features are 2D: (num_samples, feature_dim), apply PCA to reduce to target_dim.
    If features are 3D: (num_samples, time_steps, feature_dim), flatten first, then PCA.
    """
    features_np = np.array(features)
    if features_np.ndim == 3:
        num_samples, time_steps, feature_dim = features_np.shape
        features_np = features_np.reshape(num_samples, -1)

    pca = PCA(n_components=target_dim)
    reduced_features = pca.fit_transform(features_np)
    return [reduced for reduced in reduced_features]



# def create_data_loaders(train_path, val_path, dims, batch_size=32, reduce_labels=None):
#     """
#     Create DataLoaders for training and validation with dynamic dimensions.

#     Args:
#         train_path (str): Path to training data JSON.
#         val_path (str): Path to validation data JSON.
#         dims (dict): Dictionary specifying dimensions for audio, text, and video.
#                      Example: {"audio": 100, "text": 768, "video": 512}
#         batch_size (int): Batch size for DataLoaders.

#     Returns:
#         dict: Training and validation DataLoaders for each modality.
#         dict: Label mapping for class indices.
#     """
#     def load_data_with_preprocessing(json_path, output_dir, save_every=200, reduce_labels=None):
#         with open(json_path, "r") as f:
#             data = json.load(f)
        
#         print(f"Number of samples in {json_path}: {len(data)}")
#         if reduce_labels is not None:
#             data = [item for item in data if item["label"] in reduce_labels]


#         print(f"Number of samples in {json_path}: {len(data)}")
#         import pandas as pd
#         df = pd.DataFrame(data)
#         labels_unique = df["label"].unique()
#         print(f"Unique labels: {labels_unique}")


#         audio_embeddings, text_embeddings, video_embeddings, labels, speakers = [], [], [], [], []
#         processor_audio, model_audio = load_audio_model()
#         tokenizer_text, model_text = load_text_model()
#         feature_extractor_video, model_video = load_vit_model()
        

#         os.makedirs(output_dir, exist_ok=True)
#         for i, item in enumerate(tqdm(data, desc=f"Processing {json_path}")):
#                 try:
#                     audio_path = item["audio"]
#                     audio_embedding = preprocess_audio_for_model(
#                         audio_path, processor=processor_audio, model=model_audio, target_sample_rate=16000, target_duration=8.0
#                     )
#                     audio_embeddings.append(audio_embedding)

#                     text = item["text"]
#                     text_embedding = preprocess_text_for_model(
#                         text, tokenizer=tokenizer_text, model=model_text, max_length=128
#                     )
#                     text_embeddings.append(text_embedding)

#                     video_path = item["video"]
#                     video_embedding = preprocess_video_for_model(
#                         video_path=video_path, image_processor=feature_extractor_video, model=model_video, num_frames=16, frame_size=(224, 224),
#                         )
#                     video_embeddings.append(video_embedding)

#                     labels.append(item["label"])
#                     speakers.append(item["speaker"])

#                 except Exception as e:
#                     print(f"Error processing row {i}: {e}")

#                 if (i + 1) % save_every == 0 or (i + 1) == len(data): 
#                     torch.save({
#                         "audio": audio_embeddings,
#                         "text": text_embeddings,
#                         "video": video_embeddings,
#                         "labels": labels,
#                         "speakers": speakers
#                     }, os.path.join(output_dir, f"progress_{i+1}.pt"))
#                     print(f"Saved progress after {i+1} rows.")
#         return audio_embeddings, video_embeddings, text_embeddings, labels, speakers
    

#     train_audio, train_video, train_text, train_labels, train_speakers = load_data_with_preprocessing(json_path=train_path, output_dir="../outputs/preprocessed/train", reduce_labels=reduce_labels)
#     val_audio, val_video, val_text, val_labels, val_speakers = load_data_with_preprocessing(json_path=val_path, output_dir="../outputs/preprocessed/val", reduce_labels=reduce_labels)

#     label_mapping = {label: idx for idx, label in enumerate(set(train_labels + val_labels))}
#     train_labels = torch.tensor([label_mapping[label] for label in train_labels], dtype=torch.long)
#     val_labels = torch.tensor([label_mapping[label] for label in val_labels], dtype=torch.long)

#     speaker_mapping = {speaker: idx for idx, speaker in enumerate(set(train_speakers + val_speakers))}
#     train_speakers = torch.tensor([speaker_mapping[speaker] for speaker in train_speakers], dtype=torch.long)
#     val_speakers = torch.tensor([speaker_mapping[speaker] for speaker in val_speakers], dtype=torch.long)


#     train_audio_tensors = torch.tensor(train_audio, dtype=torch.float32)
#     train_video_tensors = torch.tensor(train_video, dtype=torch.float32)
#     train_text_tensors = torch.tensor(train_text, dtype=torch.float32)

#     val_audio_tensors = torch.tensor(val_audio, dtype=torch.float32)
#     val_text_tensors = torch.tensor(val_text, dtype=torch.float32)
#     val_video_tensors = torch.tensor(val_video, dtype=torch.float32)

#     train_loaders = {
#         "audio": DataLoader(TensorDataset(train_audio_tensors, train_labels), batch_size=batch_size, shuffle=True),
#         "text": DataLoader(TensorDataset(train_text_tensors, train_labels), batch_size=batch_size, shuffle=True),
#         "video": DataLoader(TensorDataset(train_video_tensors, train_labels), batch_size=batch_size, shuffle=True),
#         "speaker": DataLoader(TensorDataset(train_speakers, train_labels), batch_size=batch_size, shuffle=True)
#     }
#     val_loaders = {
#         "audio": DataLoader(TensorDataset(val_audio_tensors, val_labels), batch_size=batch_size),
#         "text": DataLoader(TensorDataset(val_text_tensors, val_labels), batch_size=batch_size),
#         "video": DataLoader(TensorDataset(val_video_tensors, val_labels), batch_size=batch_size),
#         "speaker": DataLoader(TensorDataset(val_speakers, val_labels), batch_size=batch_size)
#     }
#     return train_loaders, val_loaders, label_mapping

def extract_tensors_from_tensordataset(tensor_dataset):
    features = []
    for item in tensor_dataset:
        features.append(item[0])
    return torch.stack(features)




def create_weighted_sampler(labels):
    class_counts = torch.bincount(labels)
    class_weights = 1.0/class_counts.float()
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def create_weighted_sampler_dynamique(labels, smoothing_factor=0.5):
    """
    Crée un WeightedRandomSampler avec un facteur d'ajustement dynamique.
    """
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / (class_counts.float() ** smoothing_factor)  
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


class MultimodalGraphDataset(Dataset):
    def __init__(self, audio, text, video, labels, speaker_ids):
        self.audio = audio
        self.text = text
        self.video = video
        self.labels = labels
        self.speaker_ids = speaker_ids
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.audio[idx], self.text[idx], self.video[idx],   self.labels[idx], self.speaker_ids[idx]


def create_dataloader_with_graph_features(data):
    train_labels = data["train"]["labels"]
    val_labels = data["val"]["labels"]
    test_labels = data["test"]["labels"]

    weights_sampler_train = create_weighted_sampler_dynamique(train_labels)
    weights_sampler_val = create_weighted_sampler_dynamique(val_labels)
    weights_sampler_test = create_weighted_sampler_dynamique(test_labels)

    train_dataset = MultimodalGraphDataset(
        audio=data["train"]["audio"].tensors[0],
        text=data["train"]["text"].tensors[0],
        video=data["train"]["video"].tensors[0],
        labels=data["train"]["labels"],
        speaker_ids=data["train"]["speaker"]
    )
    val_dataset = MultimodalGraphDataset(
        audio=data["val"]["audio"].tensors[0],
        text=data["val"]["text"].tensors[0],
        video=data["val"]["video"].tensors[0],
        labels=data["val"]["labels"],
        speaker_ids=data["val"]["speaker"]
    )
    test_dataset = MultimodalGraphDataset(
        audio=data["test"]["audio"].tensors[0],
        text=data["test"]["text"].tensors[0],
        video=data["test"]["video"].tensors[0],
        labels=data["test"]["labels"],
        speaker_ids=data["test"]["speaker"]
    )
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=weights_sampler_train)
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)
    return train_loader, val_loader, test_loader


