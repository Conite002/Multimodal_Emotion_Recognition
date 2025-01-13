import re
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


def load_text_data(csv_file):
    """
    Load text data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing text metadata.

    Returns:
        pd.DataFrame: DataFrame containing text metadata.
    """
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading text data from {csv_file}: {e}")
        return pd.DataFrame()




def load_text_model(model_name="bert-base-uncased"):
    """
    Load the text model and tokenizer.

    Args:
        model_name (str): Hugging Face model name for text processing.

    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def preprocess_text_for_model(text, tokenizer, model, max_length=128):
    """
    Preprocess text and extract embeddings using a specified model.

    Args:
        text (str): Input text.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        np.ndarray: Extracted text embeddings.
    """
    try:
        inputs = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return embeddings
    except Exception as e:
        print(f"Error processing text: {e}")
        return None
