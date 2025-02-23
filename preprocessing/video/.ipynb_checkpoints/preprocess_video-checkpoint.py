import os
import cv2
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel,  BertTokenizer, VisualBertModel
from transformers import AutoImageProcessor, ViTModel
import numpy as np
import torch

from moviepy.video.io.VideoFileClip import VideoFileClip

def load_video_data_and_extract_audio(video_dir, output_audio_dir):
    """
    Load video data and extract audio tracks.

    Args:
        video_dir (str): Path to the directory containing video files.
        output_audio_dir (str): Path to the directory where extracted audio files will be saved.

    Returns:
        List[dict]: A list of metadata dictionaries with video and audio file paths.
    """
    os.makedirs(output_audio_dir, exist_ok=True)
    metadata = []

    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                audio_filename = os.path.splitext(file)[0] + '.wav'
                audio_path = os.path.join(output_audio_dir, audio_filename)

                try:
                    with VideoFileClip(video_path) as video:
                        video.audio.write_audiofile(audio_path, fps=16000)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue

                metadata.append({
                    "video_path": video_path,
                    "audio_path": audio_path
                })

    return metadata



def load_vit_model(model_name="google/vit-base-patch16-224-in21k"):
    """
    Load the ViT model and feature extractor.

    Args:
        model_name (str): Hugging Face model name for video processing.

    Returns:
        tuple: (feature_extractor, model)
    """
    # feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    # model = ViTModel.from_pretrained(model_name)
    # model.eval()
    # return feature_extractor, model
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    model.eval()
    return image_processor, model

import os
import cv2
import torch
import numpy as np


def preprocess_video_for_model(video_path, image_processor, model, num_frames=16, frame_size=(224, 224)):
    try:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found -> {video_path}")
            return None

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            num_frames = total_frames
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames_list = []

        for idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()
            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, frame_size)
                frames_list.append(frame_resized)

        video.release()

        if len(frames_list) == 0:
            print("Error: No valid frames extracted.")
            return None

        frames_array = np.array(frames_list, dtype=np.float32) / 255.0

        inputs = image_processor(images=frames_array, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

    except Exception as e:
        print(f"Error processing video, reason: {e}")
        return None
# -----------------------------------------------------------------------------------------------------------------------
from tqdm import tqdm
from transformers import BertTokenizer, VisualBertModel

def load_visualbert_model(model_name="uclanlp/visualbert-vqa-coco-pre"):
    """
    Load the VisualBERT model and tokenizer.
    
    Args:
        model_name (str): Pretrained model name.
    
    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = VisualBertModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

import torch
import cv2
import numpy as np
from transformers import BertTokenizer, VisualBertModel

def preprocess_video_for_visualbert(video_path, tokenizer, model, num_frames=16, frame_size=(224, 224)):
    """
        Preprocess a video and extract embeddings using VisualBERT.
        
        Args:
            video_path (str): Path to the video file.
            tokenizer: BERT tokenizer.
            model: VisualBERT model.
            num_frames (int): Number of frames to sample.
            frame_size (tuple): Frame size (height, width).
        
        Returns:
            torch.Tensor: Extracted video embeddings.
    """
    try:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()
            if success:
                resized_frame = cv2.resize(frame, frame_size)
                frames.append(resized_frame)
        
        video.release()
        
        if not frames:
            print(f"Error: No frames extracted from video {video_path}")
            return None
        
        visual_embeds = torch.stack([torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) for frame in frames])
        visual_embeds = visual_embeds.mean(dim=0, keepdim=True)
        inputs = tokenizer("Describe the video content", return_tensors="pt")
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        inputs.update({
            "visual_embeds": visual_embeds.unsqueeze(0), 
            "visual_token_type_ids": visual_token_type_ids.unsqueeze(0),
            "visual_attention_mask": visual_attention_mask.unsqueeze(0),
        })
        
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings
    except Exception as e:
        print(f"Error processing video: {e}")
        return None