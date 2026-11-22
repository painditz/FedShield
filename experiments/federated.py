"""
Federated Experiment Runner for FedShield.
Supports custom CLI parameter overrides for algorithm, data partitioning,
Byzantine attack injection, robust aggregation, and differential privacy.
"""

import os
import sys
import argparse
import torch

# Ensure FedShield root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_parser import load_config, set_seed, get_device
from data.loader import load_cifar10
from server.server import FederatedServer
from torch.utils.data import Subset


def run_federated_experiment(
    config_path: str = "config/config.yaml",
    mode: str = "quick",
    algorithm: str = None,
    partition_type: str = None,
    dirichlet_alpha: float = None,
    rounds: int = None,
    num_clients: int = None,
    malicious_fraction: float = None,
    attack_type: str = None,
    aggregation: str = None,
    scaling_factor: float = None,
    dp_enabled: bool = None,
    noise_multiplier: float = None,
    experiment_name: str = None
):
    """Executes a federated learning experiment."""
    cfg = load_config(config_path)

    # CLI Overrides
    if algorithm:
        cfg.federated.algorithm = algorithm
    if partition_type:
        cfg.partition.type = partition_type
    if dirichlet_alpha is not None:
        cfg.partition.dirichlet_alpha = dirichlet_alpha
    if num_clients is not None:
        cfg.federated.num_clients = num_clients
    if rounds is not None:
        cfg.federated.communication_rounds = rounds
    if malicious_fraction is not None:
        cfg.security.malicious_client_fraction = malicious_fraction
    if attack_type:
        cfg.security.attack_type = attack_type
    if aggregation:
        cfg.security.aggregation_method = aggregation
    if scaling_factor is not None:
        cfg.security.scaling_factor = scaling_factor
    if dp_enabled is not None:
        cfg.privacy.dp_enabled = dp_enabled
    if noise_multiplier is not None:
        cfg.privacy.noise_multiplier = noise_multiplier

    # Quick mode adjustments for fast verification & live demos
    if mode == "quick":
        if rounds is None:
            cfg.federated.communication_rounds = 3
        cfg.federated.local_epochs = 1
        cfg.centralized.quick_mode_samples = 1500
    else:
        if rounds is None:
            cfg.federated.communication_rounds = cfg.federated.communication_rounds

    set_seed(cfg.system.random_seed)
    device = get_device(cfg.system.device)

    # Load CIFAR-10
    print(f"[FedShield] Loading CIFAR-10 Dataset (Mode: {mode.upper()})...")
    train_dataset, test_dataset = load_cifar10(
        data_dir=cfg.dataset.data_dir,
        download=cfg.dataset.download,
        augment=True
    )

    if mode == "quick":
        # Subsample for swift demo execution
        train_indices = list(range(min(len(train_dataset), cfg.centralized.quick_mode_samples)))
        test_indices = list(range(min(len(test_dataset), 500)))
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Instantiate and execute Federated Server
    server = FederatedServer(
        config=cfg,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device=device,
        experiment_id=experiment_name
    )

    results = server.train()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedShield Federated Training Runner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"], help="quick or full training")
    parser.add_argument("--algorithm", type=str, choices=["fedavg", "fedprox"], help="Optimization algorithm")
    parser.add_argument("--partition", type=str, choices=["iid", "non_iid_dirichlet"], help="Data partition type")
    parser.add_argument("--alpha", type=float, help="Dirichlet alpha (e.g. 0.1, 0.5, 1.0)")
    parser.add_argument("--rounds", type=int, help="Number of communication rounds")
    parser.add_argument("--clients", type=int, help="Total number of clients")
    parser.add_argument("--malicious-fraction", type=float, help="Fraction of Byzantine clients (0.0 - 0.5)")
    parser.add_argument("--attack", type=str, choices=["sign_flipping", "update_scaling", "random_noise", "zero_update"], help="Attack type")
    parser.add_argument("--aggregation", type=str, choices=["fedavg", "coordinate_median", "trimmed_mean", "krum"], help="Aggregation method")
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy")
    parser.add_argument("--noise-multiplier", type=float, help="DP Gaussian noise scale")
    parser.add_argument("--name", type=str, help="Custom experiment name")

    args = parser.parse_args()
    run_federated_experiment(
        config_path=args.config,
        mode=args.mode,
        algorithm=args.algorithm,
        partition_type=args.partition,
        dirichlet_alpha=args.alpha,
        rounds=args.rounds,
        num_clients=args.clients,
        malicious_fraction=args.malicious_fraction,
        attack_type=args.attack,
        aggregation=args.aggregation,
        dp_enabled=True if args.dp else None,
        noise_multiplier=args.noise_multiplier,
        experiment_name=args.name
    )
