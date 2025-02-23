from sklearn.decomposition import PCA
import numpy as np


def reduce_dimensions(embeddings, target_dim, method="PCA"):
    """
    Reduce dimensions of embeddings using the specified method.

    Args:
        embeddings (np.ndarray): Input embeddings.
        target_dim (int): Target number of dimensions.
        method (str): Dimensionality reduction method. Default is "PCA".

    Returns:
        np.ndarray: Reduced dimension embeddings.
    """
    if embeddings is None:
        return None

    if method == "PCA":
        pca = PCA(n_components=target_dim)
        reduced_embeddings = pca.fit_transform(embeddings)
        return reduced_embeddings
    else:
        raise ValueError(f"Dimensionality reduction method {method} is not supported.")
