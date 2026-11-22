"""
Centralized Training Baseline for FedShield.
Trains a centralized Lightweight CNN on CIFAR-10 to establish the benchmark
for accuracy, convergence rate, loss, and training time.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Ensure FedShield root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_parser import load_config, set_seed, get_device
from data.loader import load_cifar10, profile_dataset, print_dataset_profile, get_dataloaders
from models.cnn import create_model
from evaluation.metrics import evaluate_model
from evaluation.logger import ExperimentLogger


def train_centralized(
    mode: str = "quick",
    config_path: str = "config/config.yaml",
    custom_epochs: int = None
):
    """Executes centralized training baseline."""
    cfg = load_config(config_path)
    set_seed(cfg.system.random_seed)
    device = get_device(cfg.system.device)

    print("=" * 60)
    print(f" FedShield Centralized Training Baseline [Mode: {mode.upper()}]")
    print(f" Device: {device} | Random Seed: {cfg.system.random_seed}")
    print("=" * 60)

    # Determine parameters based on mode
    if mode == "quick":
        epochs = custom_epochs if custom_epochs is not None else 2
        batch_size = 32
        quick_samples = cfg.centralized.quick_mode_samples
    else:  # full
        epochs = custom_epochs if custom_epochs is not None else cfg.centralized.epochs
        batch_size = cfg.centralized.batch_size
        quick_samples = None

    # Load CIFAR-10
    print("[1/4] Loading and profiling dataset...")
    train_dataset, test_dataset = load_cifar10(
        data_dir=cfg.dataset.data_dir,
        download=cfg.dataset.download,
        augment=True
    )
    profile = profile_dataset(train_dataset, test_dataset)
    print_dataset_profile(profile)

    train_loader, test_loader = get_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=cfg.system.num_workers,
        quick_mode_samples=quick_samples
    )

    print(f"[2/4] Initializing model ({cfg.model.name})...")
    model = create_model(
        model_name=cfg.model.name,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes
    ).to(device)

    param_count = model.count_parameters()
    model_size_kb = model.get_model_size_bytes() / 1024
    print(f" Total Trainable Parameters: {param_count:,}")
    print(f" Serialized Model Size     : {model_size_kb:.2f} KB")

    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.centralized.learning_rate,
        weight_decay=cfg.centralized.weight_decay
    )

    # Logger initialization
    exp_id = f"centralized_{mode}_{int(time.time())}"
    logger = ExperimentLogger(results_dir=cfg.logging.results_dir, experiment_name=exp_id)
    exp_metadata = {
        "algorithm": "centralized",
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": cfg.centralized.learning_rate,
        "device": str(device),
        "param_count": param_count,
        "model_size_bytes": model.get_model_size_bytes()
    }
    logger.set_config(exp_metadata)

    # Training Loop
    print(f"\n[3/4] Beginning Centralized Training for {epochs} epoch(s)...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        epoch_time = time.time() - epoch_start

        # Evaluate on test loader at each epoch
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        print(f" Epoch [{epoch}/{epochs}] ({epoch_time:.1f}s) | "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.2f}%")

        logger.log_round({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "train_acc": round(epoch_acc, 2),
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "epoch_duration_s": round(epoch_time, 2)
        })

    # Final Comprehensive Evaluation
    print("\n[4/4] Computing Final Comprehensive Evaluation Metrics...")
    final_metrics = evaluate_model(model, test_loader, criterion, device)
    print("=" * 60)
    print(" CENTRALIZED BASELINE RESULTS:")
    print("=" * 60)
    print(f" Final Test Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f" Final Test Loss    : {final_metrics['loss']:.4f}")
    print(f" Macro Precision    : {final_metrics['precision']:.4f}")
    print(f" Macro Recall       : {final_metrics['recall']:.4f}")
    print(f" Macro F1-Score     : {final_metrics['f1_score']:.4f}")
    print("=" * 60)

    # Save checkpoint
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("./checkpoints", f"centralized_{mode}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f" Checkpoint saved to: {checkpoint_path}")

    # Finalize logger
    logger.finalize(final_metrics)
    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedShield Centralized Baseline Training")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"],
                        help="Execution mode: quick (subsampled fast run) or full (complete dataset)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    train_centralized(mode=args.mode, config_path=args.config, custom_epochs=args.epochs)
