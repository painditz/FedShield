"""
Configuration Parser for FedShield.
Loads, validates, and manages configuration settings with typed structures and seed reproducibility.
"""

import os
import random
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Ensure deterministic execution across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_setting: str = "auto") -> torch.device:
    """Resolves computational device (CUDA or CPU)."""
    if device_setting == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif device_setting == "cpu":
        return torch.device("cpu")
    else:  # "auto"
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SystemConfig:
    random_seed: int = 42
    device: str = "auto"
    num_workers: int = 0


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    data_dir: str = "./data"
    download: bool = True
    num_classes: int = 10
    image_size: List[int] = field(default_factory=lambda: [32, 32])
    in_channels: int = 3


@dataclass
class ModelConfig:
    name: str = "lightweight_cnn"
    in_channels: int = 3
    num_classes: int = 10


@dataclass
class CentralizedConfig:
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    quick_mode_samples: int = 2000


@dataclass
class FederatedConfig:
    algorithm: str = "fedavg"
    num_clients: int = 10
    client_fraction: float = 1.0
    communication_rounds: int = 10
    local_epochs: int = 2
    local_batch_size: int = 32
    local_learning_rate: float = 0.001
    weight_decay: float = 0.0001
    fedprox_mu: float = 0.01


@dataclass
class PartitionConfig:
    type: str = "iid"
    dirichlet_alpha: float = 0.5
    imbalance: bool = False


@dataclass
class SecurityConfig:
    malicious_client_fraction: float = 0.0
    attack_type: str = "sign_flipping"
    scaling_factor: float = 10.0
    aggregation_method: str = "fedavg"
    trimmed_mean_beta: float = 0.1


@dataclass
class PrivacyConfig:
    dp_enabled: bool = False
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.5
    clip_per_sample: bool = False


@dataclass
class LoggingConfig:
    results_dir: str = "./experiments/results"
    save_plots: bool = True
    log_level: str = "INFO"


@dataclass
class FedShieldConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    centralized: CentralizedConfig = field(default_factory=CentralizedConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str = "config/config.yaml") -> FedShieldConfig:
    """Loads and validates configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f) or {}

    config = FedShieldConfig(
        system=SystemConfig(**raw_cfg.get("system", {})),
        dataset=DatasetConfig(**raw_cfg.get("dataset", {})),
        model=ModelConfig(**raw_cfg.get("model", {})),
        centralized=CentralizedConfig(**raw_cfg.get("centralized", {})),
        federated=FederatedConfig(**raw_cfg.get("federated", {})),
        partition=PartitionConfig(**raw_cfg.get("partition", {})),
        security=SecurityConfig(**raw_cfg.get("security", {})),
        privacy=PrivacyConfig(**raw_cfg.get("privacy", {})),
        logging=LoggingConfig(**raw_cfg.get("logging", {})),
    )
    return config
