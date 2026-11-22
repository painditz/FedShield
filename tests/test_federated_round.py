"""
Integration tests for a complete Federated Learning Round in FedShield.
"""

import sys
import os
import torch
from torch.utils.data import TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config_parser import load_config
from server.server import FederatedServer


def create_synthetic_dataset(num_samples: int = 100, num_classes: int = 10):
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
    return TensorDataset(x, y)


def test_federated_server_single_round():
    """Verify FederatedServer successfully coordinates a single communication round."""
    cfg = load_config("config/config.yaml")
    cfg.federated.num_clients = 3
    cfg.federated.communication_rounds = 1
    cfg.federated.local_epochs = 1
    cfg.federated.local_batch_size = 16
    cfg.centralized.batch_size = 16

    train_ds = create_synthetic_dataset(90, 10)
    test_ds = create_synthetic_dataset(30, 10)

    device = torch.device("cpu")
    server = FederatedServer(
        config=cfg,
        train_dataset=train_ds,
        test_dataset=test_ds,
        device=device,
        experiment_id="test_round_exec"
    )

    round_record = server.run_round(round_num=1)
    assert round_record["round"] == 1
    assert "test_accuracy" in round_record
    assert "test_loss" in round_record
    assert round_record["participating_clients"] == 3


def test_federated_server_with_dp():
    """Verify Differential Privacy noise injection operates without crash."""
    cfg = load_config("config/config.yaml")
    cfg.federated.num_clients = 3
    cfg.privacy.dp_enabled = True
    cfg.privacy.noise_multiplier = 0.5
    cfg.privacy.max_grad_norm = 1.0

    train_ds = create_synthetic_dataset(60, 10)
    test_ds = create_synthetic_dataset(20, 10)

    device = torch.device("cpu")
    server = FederatedServer(
        config=cfg,
        train_dataset=train_ds,
        test_dataset=test_ds,
        device=device,
        experiment_id="test_dp_round"
    )

    round_record = server.run_round(round_num=1)
    assert round_record["privacy"]["dp_enabled"] is True
    assert round_record["privacy"]["epsilon"] is not None


if __name__ == "__main__":
    test_federated_server_single_round()
    test_federated_server_with_dp()
    print("All federated integration tests passed successfully!")
