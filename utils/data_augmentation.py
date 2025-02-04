import torchaudio
import torchaudio.transforms as T
import random
import torch
import torchvision.transforms as transforms
import torchvision.io as io
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np
from nlpaug.augmenter.word import SynonymAug
from utils.dataloader import load_audio_model, load_text_model, load_visualbert_model
from preprocessing.audio.preprocess_audio import preprocess_audio_for_model
from preprocessing.video.preprocess_video import preprocess_video_for_model
from preprocessing.text.preprocess_text import preprocess_text_for_model
from tqdm import tqdm



# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and move them to GPU
processor_audio, model_audio = load_audio_model()
model_audio.to(device)

tokenizer_text, model_text = load_text_model()
model_text.to(device)

feature_extractor_video, model_video = load_visualbert_model()
model_video.to(device)

# --- Optimized Audio Augmentation ---
def augment_audio(audio_path, sample_rate=16000):
    try:
        waveform, sr = torchaudio.load(audio_path)  # Faster than librosa
        print(f"Audio loaded from {audio_path} with shape: {waveform.shape}")
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    choice = random.choice(["noise", "pitch", "speed", "shift"])

    if choice == "noise":
        noise = torch.randn_like(waveform) * 0.005
        waveform += noise
    elif choice == "pitch":
        pitch_shift = T.PitchShift(sample_rate, n_steps=random.uniform(-2, 2))
        waveform = pitch_shift(waveform)
    elif choice == "speed":
        speed_factor = random.uniform(0.9, 1.1)
        resampler = T.Resample(orig_freq=sr, new_freq=int(sr * speed_factor))
        waveform = resampler(waveform)
    elif choice == "shift":
        shift = np.random.randint(0, waveform.shape[1] // 10)
        waveform = torch.roll(waveform, shifts=shift, dims=1)

    print(f"Audio augmented, shape: {waveform.shape}")
    return waveform

# --- Optimized Text Augmentation ---
def augment_text(text):
    try:
        aug = SynonymAug(aug_src='wordnet')
        augmented_text = aug.augment(text)
        print(f"Text augmented: {augmented_text}")
        return augmented_text
    except Exception as e:
        print(f"Error augmenting text: {e}")
        return text

# --- Optimized Video Augmentation ---
def augment_video(video_path):
    try:
        video, _, _ = io.read_video(video_path, pts_unit="sec")  # Read video efficiently
        video = video.permute(0, 3, 1, 2).float() / 255.0  # Convert to (T, C, H, W)
        print(f"Video loaded from {video_path} with shape: {video.shape}")
    except Exception as e:
        print(f"Error loading video file {video_path}: {e}")
        return None

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    augmented_video = transform(video)  # Apply transformations
    print(f"Video augmented, shape: {augmented_video.shape}")
    return augmented_video


def augment_underrepresented_classes(data):
    initial_count = len(data)
    print(f"📌 Initial training samples before augmentation: {initial_count}")

    class_counts = {label: sum(1 for x in data if x["label"] == label) for label in set(x["label"] for x in data)}
    max_samples = max(class_counts.values())

    augmented_samples = []
    underrepresented_classes = sorted(class_counts, key=class_counts.get)[:4]  # Focus on the 4 least represented classes

    # ✅ Use tqdm to show augmentation progress
    for label in tqdm(underrepresented_classes, desc="Augmenting underrepresented classes", unit="class"):
        class_samples = [x for x in data if x["label"] == label]
        num_samples_to_add = max_samples - class_counts[label]

        selected_samples = random.choices(class_samples, k=num_samples_to_add)

        # ✅ Use tqdm to track augmentation steps
        for item in tqdm(selected_samples, desc=f"Processing label {label}", unit="sample"):
            audio_aug = augment_audio(item["audio"])
            text_aug = augment_text(item["text"])
            video_aug = augment_video(item["video"])

            if audio_aug is not None and text_aug is not None and video_aug is not None:
                augmented_samples.append({
                    "audio": audio_aug,
                    "text": text_aug,
                    "video": video_aug,
                    "label": item["label"],
                    "speaker": item["speaker"]
                })

    data.extend(augmented_samples)  # ✅ Append new augmented data

    final_count = len(data)
    print(f"✅ Number of training samples AFTER augmentation: {final_count}")

    if final_count <= initial_count:
        print("⚠ WARNING: Data augmentation did NOT increase the number of samples! Check augmentation logic.")

    return data



# --- Optimized DataLoader Creation with Debug ---
def create_data_loaders_augmented(train_path, batch_size=32):
    try:
        with open(train_path, "r") as f:
            train_data = json.load(f)
    except Exception as e:
        print(f"Error loading train data from {train_path}: {e}")
        return None

    initial_count = len(train_data)
    print(f"📌 Initial number of training samples: {initial_count}")

    train_data = augment_underrepresented_classes(train_data)

    augmented_count = len(train_data)
    print(f"✅ Number of training samples after augmentation: {augmented_count}")

    # Vérifier si l'augmentation a bien augmenté le nombre d'échantillons
    if augmented_count <= initial_count:
        print("⚠ WARNING: Data augmentation did NOT increase the number of samples! Check augmentation logic.")

    audio_embeddings, text_embeddings, video_embeddings, labels, speakers = [], [], [], [], []
    speakers_mapping = {speaker: idx for idx, speaker in enumerate(set(x["speaker"] for x in train_data))}

    try:
        for item in tqdm(train_data, desc="Processing Data", unit="sample"):
            try:
                audio_embedding = preprocess_audio_for_model(
                    item["audio"], processor=processor_audio, model=model_audio, target_sample_rate=16000, target_duration=8.0
                )
                text_embedding = preprocess_text_for_model(
                    item["text"], tokenizer=tokenizer_text, model=model_text, max_length=128
                )
                video_embedding = preprocess_video_for_model(
                    item["video"], feature_extractor_video, model=model_video, num_frames=16, frame_size=(224, 224)
                )
                speakers.append(speakers_mapping[item["speaker"]])

                audio_embeddings.append(audio_embedding)
                text_embeddings.append(text_embedding)
                video_embeddings.append(video_embedding)
                labels.append(item["label"])
            except Exception as e:
                print(f"❌ Error processing sample {item}: {e}")
        
    except Exception as e:
        print(f"❌ Error during data processing: {e}")
        return None

    print(f"✅ Total processed: {len(audio_embeddings)} audio, {len(text_embeddings)} text, {len(video_embeddings)} video")

    try:
        if len(audio_embeddings) == 0 or len(text_embeddings) == 0 or len(video_embeddings) == 0:
            print("⚠ ERROR: One of the modalities has 0 samples.")
            return None

        train_audio_tensors = torch.stack(audio_embeddings).float()
        train_text_tensors = torch.stack(text_embeddings).float()
        train_video_tensors = torch.stack(video_embeddings).float()
        train_speakers = torch.tensor(speakers, dtype=torch.long)
        train_labels = torch.tensor(labels, dtype=torch.long)
    except Exception as e:
        print(f"❌ Error during tensor stacking: {e}")
        return None

    train_loaders = {
        "audio": DataLoader(TensorDataset(train_audio_tensors, train_labels), batch_size=batch_size, shuffle=True),
        "text": DataLoader(TensorDataset(train_text_tensors, train_labels), batch_size=batch_size, shuffle=True),
        "video": DataLoader(TensorDataset(train_video_tensors, train_labels), batch_size=batch_size, shuffle=True),
        "speaker": DataLoader(TensorDataset(train_speakers, train_labels), batch_size=batch_size, shuffle=True)
    }
    import pickle

    # --- Save Train Loaders ---
    save_path = "train_loaders_augmented.pth"
    save_data = {
        "audio": train_audio_tensors,
        "text": train_text_tensors,
        "video": train_video_tensors,
        "speaker": train_speakers,
        "labels": train_labels
    }

    torch.save(save_data, save_path)
    print(f"✅ Train loaders saved successfully at {save_path}")

    return train_loaders