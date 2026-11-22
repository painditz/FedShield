"""
Dataset Loader and Profiler for FedShield.
Handles automated CIFAR-10 downloading (with robust User-Agent headers & extraction),
normalization, augmentation, dataset profiling, and PyTorch DataLoader construction.
"""

import os
import tarfile
import urllib.request
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


# CIFAR-10 Standard Class Names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Standard Normalization Constants for CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_NAME = "cifar-10-python.tar.gz"
CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"


def download_and_extract_cifar10(data_dir: str = "./data") -> None:
    """
    Downloads and extracts CIFAR-10 with a browser User-Agent to avoid
    HTTP 403 Forbidden or connection drop issues on university servers.
    """
    os.makedirs(data_dir, exist_ok=True)
    extracted_path = os.path.join(data_dir, CIFAR10_EXTRACTED_DIR)
    archive_path = os.path.join(data_dir, CIFAR10_ARCHIVE_NAME)

    if os.path.exists(extracted_path):
        return  # Already downloaded and extracted

    if not os.path.exists(archive_path):
        print(f"[FedShield Data] Downloading CIFAR-10 from {CIFAR10_URL}...")
        req = urllib.request.Request(
            CIFAR10_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req) as response, open(archive_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 1024 * 1024  # 1MB
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="CIFAR-10 Download") as pbar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    pbar.update(len(buffer))

    print(f"[FedShield Data] Extracting {archive_path} to {data_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("[FedShield Data] CIFAR-10 extraction completed.")


def get_cifar10_transforms(augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and test transforms.
    Augmentation includes random crop with padding and horizontal flip.
    """
    train_transforms_list = []
    if augment:
        train_transforms_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return transforms.Compose(train_transforms_list), test_transforms


def load_cifar10(
    data_dir: str = "./data",
    download: bool = True,
    augment: bool = True
) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """
    Ensures CIFAR-10 dataset is available and returns the train and test datasets.
    """
    if download:
        download_and_extract_cifar10(data_dir=data_dir)

    train_transform, test_transform = get_cifar10_transforms(augment=augment)

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )

    return train_dataset, test_dataset


def profile_dataset(train_dataset: Dataset, test_dataset: Dataset) -> Dict[str, Any]:
    """
    Computes and returns comprehensive dataset statistics:
    - total samples (train, test, total)
    - number of classes
    - class names
    - image dimensions
    - class distribution in train and test sets
    """
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)

    unique_classes, train_counts = np.unique(train_targets, return_counts=True)
    _, test_counts = np.unique(test_targets, return_counts=True)

    train_class_dist = {CIFAR10_CLASSES[c]: int(count) for c, count in zip(unique_classes, train_counts)}
    test_class_dist = {CIFAR10_CLASSES[c]: int(count) for c, count in zip(unique_classes, test_counts)}

    sample_img, _ = train_dataset[0]
    img_shape = list(sample_img.shape)  # [C, H, W]

    profile = {
        "dataset_name": "CIFAR-10",
        "num_classes": len(unique_classes),
        "class_names": CIFAR10_CLASSES,
        "image_dimensions": img_shape,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "total_samples": len(train_dataset) + len(test_dataset),
        "train_class_distribution": train_class_dist,
        "test_class_distribution": test_class_dist,
    }
    return profile


def print_dataset_profile(profile: Dict[str, Any]) -> None:
    """Prints a clean human-readable summary of the dataset profile."""
    print("=" * 60)
    print(f" DATASET PROFILE: {profile['dataset_name']}")
    print("=" * 60)
    print(f" Total Samples   : {profile['total_samples']:,} (Train: {profile['train_samples']:,} | Test: {profile['test_samples']:,})")
    print(f" Image Dimensions: {profile['image_dimensions']} (Channels x Height x Width)")
    print(f" Classes ({profile['num_classes']})     : {', '.join(profile['class_names'])}")
    print("-" * 60)
    print(" Train Class Distribution:")
    for cls_name, count in profile["train_class_distribution"].items():
        bar = "#" * int(count / 200)
        print(f"   {cls_name:<12}: {count:>5} samples  {bar}")
    print("=" * 60)


def get_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    quick_mode_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for centralized training and evaluation.
    If quick_mode_samples is provided, creates a stratified/balanced subset for rapid testing.
    """
    if quick_mode_samples is not None and quick_mode_samples < len(train_dataset):
        indices = np.random.RandomState(42).choice(len(train_dataset), quick_mode_samples, replace=False)
        train_sub = Subset(train_dataset, indices)
        test_sub_size = min(1000, len(test_dataset))
        test_indices = np.random.RandomState(42).choice(len(test_dataset), test_sub_size, replace=False)
        test_sub = Subset(test_dataset, test_indices)

        train_loader = DataLoader(
            train_sub, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_sub, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        return train_loader, test_loader

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, test_loader
