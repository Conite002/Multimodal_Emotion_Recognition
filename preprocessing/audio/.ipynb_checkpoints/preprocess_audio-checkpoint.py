import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
import opensmile
import os
import ffmpeg 


def load_audio_model(model_name="facebook/wav2vec2-base"):
    """
    Load the Wav2Vec model and processor.
    Args:
        model_name (str): Hugging Face model name for Wav2Vec.

    Returns:
        tuple: (processor, model)
    """
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    return processor, model

def preprocess_audio_with_opensmile(audio_path, config_path="IS10_paraling" ):

    try:
        smile = opensmile.Smile(feature_set=config_path, feature_level="func")
        features = smile.process_file(audio_path)
        return features.to_numpy().squeeze()
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


def preprocess_audio_for_model(audio_path, processor, model, target_sample_rate=16000, target_duration=8.0):
    """
    Preprocess audio and extract embeddings using a specified model.

    Args:
        audio_path (str): Path to the audio file.
        processor: Hugging Face processor for the model.
        model: Hugging Face model.
        target_sample_rate (int): Sampling rate for the audio.
        target_duration (float): Target audio duration in seconds.

    Returns:
        np.ndarray: Extracted audio embeddings.
    """
    try:
        waveform, _ = librosa.load(audio_path, sr=target_sample_rate)
        max_length = int(target_sample_rate * target_duration)
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
        else:
            waveform = np.pad(waveform, (0, max_length - len(waveform)), mode='constant')

        inputs = processor(waveform, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return embeddings
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# import os
# from ffmpeg import FFmpeg, FFmpegError
# from tqdm import tqdm

# def extract_audio(video_dir, output_audio_dir):
#     """
#     Extrait l'audio des fichiers vidéo dans un répertoire en utilisant python-ffmpeg,
#     en ignorant les fichiers commençant par '._' ou si le fichier WAV existe déjà.

#     Args:
#         video_dir (str): Répertoire contenant les fichiers vidéo.
#         output_audio_dir (str): Répertoire pour enregistrer les fichiers audio extraits.

#     Returns:
#         list: Métadonnées reliant les fichiers vidéo et audio.
#     """
#     os.makedirs(output_audio_dir, exist_ok=True)
#     metadata = []

#     for root, _, files in os.walk(video_dir):
#         for file in tqdm(files, desc="Extraction de l'audio", unit="fichier"):
#             # Ignorer les fichiers cachés ou les fichiers de métadonnées macOS
#             if file.startswith("._") or not file.endswith(".mp4"):
#                 print(f"Fichier ignoré : {file}")
#                 continue

#             video_path = os.path.join(root, file)
#             audio_filename = os.path.splitext(file)[0] + ".wav"
#             audio_path = os.path.join(output_audio_dir, audio_filename)

#             # Ignorer si le fichier audio existe déjà
#             if os.path.exists(audio_path):
#                 print(f"Fichier audio existant ignoré : {audio_path}")
#                 metadata.append({
#                     "video_path": video_path,
#                     "audio_path": audio_path
#                 })
#                 continue

#             try:
#                 # Utiliser python-ffmpeg pour extraire l'audio
#                 ffmpeg = (
#                     FFmpeg()
#                     .option("y")
#                     .input(video_path)
#                     .output(audio_path, format='wav', ar=16000, ac=1)
#                 )
#                 ffmpeg.execute()
#                 print(f"Audio extrait avec succès : {audio_path}")
#             except FFmpegError as e:
#                 print(f"Erreur lors du traitement de {video_path} : {e}")
#                 continue

#             metadata.append({
#                 "video_path": video_path,
#                 "audio_path": audio_path
#             })

#     return metadata

def extract_audio(video_dir, output_audio_dir):
    """
    Extrait l'audio des fichiers vidéo dans un répertoire en utilisant ffmpeg-python,
    en ignorant les fichiers commençant par '._' ou si le fichier WAV existe déjà.

    Args:
        video_dir (str): Répertoire contenant les fichiers vidéo.
        output_audio_dir (str): Répertoire pour enregistrer les fichiers audio extraits.

    Returns:
        list: Métadonnées reliant les fichiers vidéo et audio.
    """
    os.makedirs(output_audio_dir, exist_ok=True)
    metadata = []
    nb_ignore_file = 0
    for root, _, files in os.walk(video_dir):
        for file in tqdm(files, desc="Extraction de l'audio", unit="fichier"):
            # Ignorer les fichiers cachés ou les fichiers de métadonnées macOS
            if file.startswith("._") or not file.endswith(".mp4"):
                # print(f"Fichier ignoré : {file}")
                nb_ignore_file += 0
                continue

            video_path = os.path.join(root, file)
            audio_filename = os.path.splitext(file)[0] + ".wav"
            audio_path = os.path.join(output_audio_dir, audio_filename)
            if os.path.exists(audio_path):
                # print(f"Fichier audio existant ignoré : {audio_path}")
                metadata.append({
                    "video_path": video_path,
                    "audio_path": audio_path
                })
                continue
            import ffmpeg
            
            try:
                (
                    ffmpeg
                    .input(video_path) 
                    .output(audio_path, format='wav', ar=16000, ac=1)
                    .run(overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)
                )            
            except ffmpeg.Error as e:  
                print(f"Erreur lors du traitement de {video_path} : {e}")

            metadata.append({
                "video_path": video_path,
                "audio_path": audio_path
            })
    print(f"Nb de fichier ignoré : {nb_ignore_file}")
    return metadata

# def extract_audio(video_dir, output_audio_dir):
#     """
#     Extract audio from video files in a directory, skipping files starting with '._' or if the WAV file already exists.

#     Args:
#         video_dir (str): Directory containing video files.
#         output_audio_dir (str): Directory to save extracted audio files.

#     Returns:
#         list: Metadata linking video and audio files.
#     """
#     import os
#     from moviepy.video.io.VideoFileClip import VideoFileClip
#     from tqdm import tqdm
#     import os

#     # os.environ["PATH"] += "/home/conite.gbodogbe/lustre/mml_bc-2q5huipcugq/users/conite.gbodogbe/Workspace/ffmpeg-6_1/bin"


#     os.makedirs(output_audio_dir, exist_ok=True)
#     metadata = []

#     for root, _, files in os.walk(video_dir):
#         for file in tqdm(files, desc="Extracting audio", unit="file"):
#             # Skip hidden files or macOS metadata files
#             if file.startswith("._") or not file.endswith(".mp4"):
#                 print(f"Skipping file: {file}")
#                 continue

#             video_path = os.path.join(root, file)
#             audio_filename = os.path.splitext(file)[0] + ".wav"
#             audio_path = os.path.join(output_audio_dir, audio_filename)

#             # Skip if the audio file already exists
#             if os.path.exists(audio_path):
#                 print(f"Skipping existing audio file: {audio_path}")
#                 metadata.append({
#                     "video_path": video_path,
#                     "audio_path": audio_path
#                 })
#                 continue

#             try:
#                 # Extract audio
#                 with VideoFileClip(video_path) as video:
#                     video.audio.write_audiofile(audio_path, fps=16000)
#                     print(f"Audio extracted successfully: {audio_path}")
#             except Exception as e:
#                 print(f"Video Path: {video_path}")
#                 print(f"Audio Path: {audio_path}")
#                 print(f"Error processing {video_path}: {e}")
#                 continue

#             metadata.append({
#                 "video_path": video_path,
#                 "audio_path": audio_path
#             })

#     return metadata

import os

def rename_files(video_dir):
    """
    Rename files starting with '._' to remove the prefix, avoiding overwriting existing files.

    Args:
        video_dir (str): Directory containing video files.
    """
    skipping_nb = 0
    for file in os.listdir(video_dir):
        if file.startswith("._") and file.endswith(".mp4"):
            original_path = os.path.join(video_dir, file)
            new_name = file[2:]  # Remove the `._` prefix
            new_path = os.path.join(video_dir, new_name)
            if os.path.exists(new_path):
                # print(f"File already exists: {new_path}. Skipping...")
                skipping_nb += 1
                continue

            # Rename the file
            os.rename(original_path, new_path)
            print(f"Renamed: {original_path} -> {new_path}")

    print(f"Nb files already exists: {skipping_nb}")
