import torchaudio
import torchaudio.transforms as T
import random
import torch
import torchvision.transforms as transforms
import torchvision.io as io
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np
import os
import psutil
import gc
from tqdm import tqdm
from utils.dataloader import load_audio_model, load_text_model, load_visualbert_model
from preprocessing.audio.preprocess_audio import preprocess_audio_for_model
from preprocessing.video.preprocess_video import preprocess_video_for_model
from preprocessing.text.preprocess_text import preprocess_text_for_model

CHECKPOINT_FILE = "progress_checkpoint.json"
AUGMENTED_DATA_FILE = "train_data_augmented.json"

def check_memory_usage():
    """Retourne le pourcentage de mémoire utilisée."""
    return psutil.virtual_memory().percent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement des modèles
processor_audio, model_audio = load_audio_model()
model_audio.to(device)

tokenizer_text, model_text = load_text_model()
model_text.to(device)

feature_extractor_video, model_video = load_visualbert_model()
model_video.to(device)

def save_progress(index):
    """Sauvegarde la progression dans un fichier JSON."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_index": index}, f)

def load_progress():
    """Charge le dernier index traité s'il existe."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f).get("last_index", 0)
    return 0

def augment_audio(audio_path):
    """Applique une augmentation aléatoire à l'audio."""
    if check_memory_usage() > 85:
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Erreur chargement audio {audio_path}: {e}")
        return None

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
    
    return waveform

def augment_text(text):
    """Augmente le texte via des synonymes."""
    if check_memory_usage() > 85:
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        from nlpaug.augmenter.word import SynonymAug
        aug = SynonymAug(aug_src='wordnet')
        return aug.augment(text)
    except:
        return text

def augment_video(video_path):
    """Augmente la vidéo en appliquant des transformations aléatoires."""
    if check_memory_usage() > 85:
        gc.collect()
        torch.cuda.empty_cache()
    
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

def augment_underrepresented_classes(data, batch_size=20):
    """Augmente progressivement les classes sous-représentées en sauvegardant après chaque batch."""
    if os.path.exists(AUGMENTED_DATA_FILE):
        with open(AUGMENTED_DATA_FILE, "r") as f:
            data = json.load(f)

    class_counts = {label: sum(1 for x in data if x["label"] == label) for label in set(x["label"] for x in data)}
    max_samples = max(class_counts.values())
    underrepresented_classes = sorted(class_counts, key=class_counts.get)[:4]

    augmented_samples = []
    last_index = load_progress()

    for label in underrepresented_classes:
        class_samples = [x for x in data if x["label"] == label]
        num_samples_to_add = max_samples - class_counts[label]

        for start in tqdm(range(last_index, num_samples_to_add, batch_size), desc=f"Augmenting {label}"):
            batch = random.choices(class_samples, k=min(batch_size, num_samples_to_add - start))
            batch_augmented = []

            for item in batch:
                audio_aug = augment_audio(item["audio"])
                text_aug = augment_text(item["text"])
                video_aug = augment_video(item["video"])

                if audio_aug is not None and text_aug is not None and video_aug is not None:
                    batch_augmented.append({
                        "audio": item["audio"],
                        "text": text_aug,
                        "video": item["video"],
                        "label": item["label"],
                        "speaker": item["speaker"]
                    })

            augmented_samples.extend(batch_augmented)

            # Sauvegarde progressive
            data.extend(batch_augmented)
            with open(AUGMENTED_DATA_FILE, "w") as f:
                json.dump(data, f)

            save_progress(start + batch_size)

            if check_memory_usage() > 85:
                print("Mémoire saturée, nettoyage...")
                gc.collect()
                torch.cuda.empty_cache()

    return data

def create_data_loaders_augmented(train_path, batch_size=32):
    """Crée des DataLoaders et continue le traitement si interrompu."""
    if os.path.exists(AUGMENTED_DATA_FILE):
        with open(AUGMENTED_DATA_FILE, "r") as f:
            train_data = json.load(f)
    else:
        with open(train_path, "r") as f:
            train_data = json.load(f)
        train_data = augment_underrepresented_classes(train_data)

    audio_embeddings, text_embeddings, video_embeddings, labels, speakers = [], [], [], [], []
    speakers_mapping = {speaker: idx for idx, speaker in enumerate(set(x["speaker"] for x in train_data))}

    for item in tqdm(train_data, desc="Processing Data"):
        try:
            audio_embeddings.append(preprocess_audio_for_model(item["audio"], processor_audio, model_audio))
            text_embeddings.append(preprocess_text_for_model(item["text"], tokenizer_text, model_text))
            video_embeddings.append(preprocess_video_for_model(item["video"], feature_extractor_video, model_video, text=item["text"]))
            speakers.append(speakers_mapping[item["speaker"]])
            labels.append(item["label"])
        except Exception as e:
            print(f"Erreur traitement: {e}")

    torch.save({"audio": audio_embeddings, "text": text_embeddings, "video": video_embeddings, "speaker": speakers, "labels": labels}, "train_loaders_augmented.pth")

    return DataLoader(TensorDataset(torch.stack(audio_embeddings), torch.tensor(labels)), batch_size=batch_size, shuffle=True)
