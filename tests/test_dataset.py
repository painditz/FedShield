"""
Unit tests for dataset loading and DataLoader construction in FedShield.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_cifar10, profile_dataset, get_dataloaders


def test_cifar10_loading_and_profile():
    """Verify CIFAR-10 loads cleanly and profile returns accurate dimensions and counts."""
    train_ds, test_ds = load_cifar10("./data", download=False)

    assert len(train_ds) == 50000, f"Expected 50,000 train samples, got {len(train_ds)}"
    assert len(test_ds) == 10000, f"Expected 10,000 test samples, got {len(test_ds)}"

    profile = profile_dataset(train_ds, test_ds)
    assert profile["num_classes"] == 10
    assert profile["image_dimensions"] == [3, 32, 32]
    assert profile["total_samples"] == 60000


def test_quick_mode_dataloaders():
    """Verify quick_mode subsampling creates correct batch shapes."""
    train_ds, test_ds = load_cifar10("./data", download=False)
    train_loader, test_loader = get_dataloaders(
        train_ds, test_ds, batch_size=16, quick_mode_samples=100
    )

    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (16, 3, 32, 32), f"Expected batch shape (16, 3, 32, 32), got {batch_x.shape}"
    assert batch_y.shape == (16,), f"Expected target shape (16,), got {batch_y.shape}"


if __name__ == "__main__":
    test_cifar10_loading_and_profile()
    test_quick_mode_dataloaders()
    print("All dataset tests passed successfully!")
