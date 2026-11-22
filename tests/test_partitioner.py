"""
Unit tests for FederatedPartitioner in FedShield.
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.partitioner import FederatedPartitioner


def create_synthetic_dataset(num_samples: int = 500, num_classes: int = 10):
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.tensor(np.random.randint(0, num_classes, size=num_samples), dtype=torch.long)
    return TensorDataset(x, y)


def test_iid_partitioning():
    """Verify IID partitioning evenly distributes samples without loss."""
    dataset = create_synthetic_dataset(num_samples=500, num_classes=10)
    partitioner = FederatedPartitioner(dataset, num_clients=5, partition_type="iid", seed=42)

    total_allocated = sum(len(indices) for indices in partitioner.client_indices)
    assert total_allocated == 500, f"Expected 500 allocated samples, got {total_allocated}"
    for indices in partitioner.client_indices:
        assert len(indices) == 100, f"Expected 100 samples per client, got {len(indices)}"


def test_dirichlet_partitioning():
    """Verify Dirichlet partitioning maintains total samples and creates non-IID distributions."""
    dataset = create_synthetic_dataset(num_samples=600, num_classes=10)
    partitioner = FederatedPartitioner(
        dataset, num_clients=6, partition_type="non_iid_dirichlet", dirichlet_alpha=0.5, seed=42
    )

    total_allocated = sum(len(indices) for indices in partitioner.client_indices)
    assert total_allocated == 600, f"Expected 600 samples, got {total_allocated}"

    dist = partitioner.get_client_class_distribution()
    assert dist.shape == (6, 10), f"Expected shape (6, 10), got {dist.shape}"
    assert np.all(dist >= 0)


def test_client_dataloader():
    """Verify DataLoader yields properly shaped batches."""
    dataset = create_synthetic_dataset(num_samples=200, num_classes=10)
    partitioner = FederatedPartitioner(dataset, num_clients=4, partition_type="iid", seed=42)
    loader = partitioner.get_client_dataloader(client_id=0, batch_size=16)

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[0] <= 16
    assert batch_x.shape[1:] == (3, 32, 32)


if __name__ == "__main__":
    test_iid_partitioning()
    test_dirichlet_partitioning()
    test_client_dataloader()
    print("All partitioner tests passed successfully!")
