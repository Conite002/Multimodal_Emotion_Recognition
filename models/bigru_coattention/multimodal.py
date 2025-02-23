import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, audio_data, text_data, video_data, labels):
        """
        A dataset class to combine audio, text, video, and labels into a single dataset.

        Args:
            audio_data (TensorDataset or torch.Tensor): Audio embeddings.
            text_data (TensorDataset or torch.Tensor): Text embeddings.
            video_data (TensorDataset or torch.Tensor): Video embeddings.
            labels (TensorDataset or torch.Tensor): Labels.
        """
        # Extract tensors if input is a TensorDataset
        self.audio_data = torch.cat([item[0].unsqueeze(0) for item in audio_data], dim=0) if isinstance(audio_data, torch.utils.data.Dataset) else audio_data
        self.text_data = torch.cat([item[0].unsqueeze(0) for item in text_data], dim=0) if isinstance(text_data, torch.utils.data.Dataset) else text_data
        self.video_data = torch.cat([item[0].unsqueeze(0) for item in video_data], dim=0) if isinstance(video_data, torch.utils.data.Dataset) else video_data
        self.labels = torch.cat([item[0].unsqueeze(0) for item in labels], dim=0) if isinstance(labels, torch.utils.data.Dataset) else labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.audio_data[idx],  
            self.text_data[idx],   
            self.video_data[idx],  
            self.labels[idx]       
        )

# --------------------------------------------------------------------------------
# MultiModalDataset_MultiSpeaker
# --------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
import json
class MultiModalDatasetWithSpeaker(Dataset):
    def __init__(self, audio_data, text_data, video_data, labels, json_path, speaker_to_id=None):
        """
        Dataset class to load audio, text, video, speakers, and labels.

        Args:
            audio_data (torch.Tensor): Preprocessed audio features.
            text_data (torch.Tensor): Preprocessed text features.
            video_data (torch.Tensor): Preprocessed video features.
            labels (torch.Tensor): Labels for classification.
            json_path (str): Path to the JSON file with speaker information.
            speaker_to_id (dict): Mapping of speaker names to unique IDs.
        """
        self.audio_data = audio_data
        self.text_data = text_data
        self.video_data = video_data
        self.labels = labels

        # Load speakers from JSON
        with open(json_path, "r") as file:
            self.json_data = json.load(file)
        speakers = [entry['Speaker'] for entry in self.json_data]

        # Handle speaker-to-ID mapping
        self.speaker_to_id = speaker_to_id or {speaker: idx for idx, speaker in enumerate(set(speakers))}
        self.speaker_to_id['UNK'] = len(self.speaker_to_id)  # Add UNK (unknown speaker) ID

        # Map speakers to IDs
        self.speaker_ids = torch.tensor(
            [self.speaker_to_id.get(speaker, self.speaker_to_id['UNK']) for speaker in speakers],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.audio_data[idx],  
            self.text_data[idx],   
            self.video_data[idx],  
            self.speaker_ids[idx],  
            self.labels[idx]       
        )


# --------------------------------------------------------------------------------