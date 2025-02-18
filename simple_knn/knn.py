import torch

from simple_knn.torch_convert import get_device
from typing import Tuple


def knn(
    x: torch.Tensor, query: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the k-nearest neighbors using MPS acceleration on Apple Silicon.

    Args:
        x (torch.Tensor): The dataset of shape (N, D), where N is the number of points, and D is the feature dimension.
        query (torch.Tensor): The query tensor of shape (Q, D), where Q is the number of query points.
        k (int): Number of nearest neighbors to return.

    Returns:
        torch.Tensor: Indices of the k nearest neighbors for each query point (Q, k).
        torch.Tensor: Distances of the k nearest neighbors (Q, k).
    """
    device = get_device()

    x = x.to(device)
    query = query.to(device)

    # Compute pairwise Euclidean distance using device acceleration
    distances = torch.cdist(query, x, p=2)  # (Q, N)

    # Get the indices of the k-nearest neighbors
    knn_distances, knn_indices = torch.topk(distances, k, largest=False, dim=1)

    return (
        knn_indices.cpu(),
        knn_distances.cpu(),
    )  # Move results back to CPU for compatibility
