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
