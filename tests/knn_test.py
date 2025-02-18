import pytest
import torch

from simple_knn.knn import knn


@pytest.mark.parametrize("k", [1, 2, 3])  # type: ignore[misc]
def test_knn(k: int) -> None:
    """Test k-nearest neighbors function on CPU and MPS if available."""

    # Small dataset: 5 points in 2D space
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        dtype=torch.float32,
    )

    # Query point near (1.1, 1.1), expected to be close to (1.0, 1.0)
    query = torch.tensor([[1.1, 1.1]], dtype=torch.float32)

    # Run k-NN
    indices, distances = knn(x, query, k)

    # Ensure correct shapes
    assert indices.shape == (1, k), "Indices should have shape (1, k)"
    assert distances.shape == (1, k), "Distances should have shape (1, k)"

    # Ensure nearest neighbor is the closest expected point
    expected_first_nn = 1  # Index of (1.0, 1.0) in dataset
    assert (
        expected_first_nn in indices[0].tolist()
    ), f"Expected {expected_first_nn} as one of the nearest neighbors."

    # Ensure distances are sorted in ascending order
    sorted_distances = sorted(distances[0].tolist())
    assert (
        distances[0].tolist() == sorted_distances
    ), "Distances should be sorted from smallest to largest."
