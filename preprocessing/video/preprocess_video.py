import os
import cv2
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import cv2
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

                # Extract audio from video
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
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    model.eval()
    return feature_extractor, model


def preprocess_video_for_model(video_path, feature_extractor, model, num_frames=16, frame_size=(224, 224)):
    """
    Preprocess video and extract embeddings using a specified model.

    Args:
        video_path (str): Path to the video file.
        feature_extractor: Hugging Face feature extractor for ViT.
        model: Hugging Face model.
        num_frames (int): Number of frames to sample.
        frame_size (tuple): Frame size (height, width).

    Returns:
        np.ndarray: Extracted video embeddings.
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

        # Prepare input for the model
        pixel_values = feature_extractor(images=frames, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            outputs = model(pixel_values)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return embeddings
    except Exception as e:
        print(f"Error processing video: {e}")
        return None
