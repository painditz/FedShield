"""
Unit tests for configuration loader and environment reproducibility in FedShield.
"""

import sys
import os
import random
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config_parser import load_config, set_seed, get_device, FedShieldConfig


def test_load_default_config():
    """Verify that config/config.yaml loads into typed dataclasses."""
    cfg = load_config("config/config.yaml")
    assert isinstance(cfg, FedShieldConfig), "Config should be an instance of FedShieldConfig"
    assert cfg.dataset.name == "cifar10", f"Expected 'cifar10', got {cfg.dataset.name}"
    assert cfg.model.name == "lightweight_cnn", f"Expected 'lightweight_cnn', got {cfg.model.name}"
    assert cfg.centralized.epochs >= 1, "Epochs must be positive"
    assert cfg.federated.num_clients > 0, "num_clients must be positive"
    assert cfg.security.aggregation_method in ["fedavg", "coordinate_median", "trimmed_mean"]
    assert cfg.privacy.noise_multiplier >= 0.0


def test_reproducibility_seed():
    """Verify that set_seed ensures deterministic pseudo-random sequences."""
    set_seed(42)
    t1 = torch.randn(10)
    n1 = np.random.randn(10)
    r1 = [random.random() for _ in range(5)]

    set_seed(42)
    t2 = torch.randn(10)
    n2 = np.random.randn(10)
    r2 = [random.random() for _ in range(5)]

    assert torch.equal(t1, t2), "PyTorch random seed is not reproducible"
    assert np.array_equal(n1, n2), "NumPy random seed is not reproducible"
    assert r1 == r2, "Python random seed is not reproducible"


def test_device_selection():
    """Verify get_device correctly handles auto, cpu, and cuda."""
    cpu_dev = get_device("cpu")
    assert cpu_dev.type == "cpu"

    auto_dev = get_device("auto")
    assert auto_dev.type in ["cuda", "cpu"]


if __name__ == "__main__":
    test_load_default_config()
    test_reproducibility_seed()
    test_device_selection()
    print("All config tests passed successfully!")
