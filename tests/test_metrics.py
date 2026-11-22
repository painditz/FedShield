"""
Unit tests for evaluation metrics and communication cost calculations in FedShield.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import evaluate_model, compute_communication_cost


class MockClassifier(nn.Module):
    """Simple linear mock model to test evaluation pipeline."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def test_evaluate_model_metrics():
    """Verify accuracy, precision, recall, f1, and confusion matrix calculation."""
    model = MockClassifier()
    model.eval()

    # Create synthetic dataset with known targets
    x = torch.randn(20, 4)
    y = torch.randint(0, 3, (20,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, loader, criterion, device=torch.device("cpu"))

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["sample_count"] == 20
    assert 0.0 <= metrics["accuracy"] <= 100.0
    assert len(metrics["confusion_matrix"]) == 3  # 3 classes


def test_communication_cost_estimation():
    """Verify analytical uplink and downlink communication cost estimates."""
    model_size_bytes = 1024 * 1024  # 1 MB
    num_clients = 10
    num_rounds = 5

    comm = compute_communication_cost(
        model_size_bytes=model_size_bytes,
        num_participating_clients=num_clients,
        num_rounds=num_rounds
    )

    # Downlink + Uplink per client = 2 * 1MB * 10 clients = 20 MB per round
    # 5 rounds = 100 MB
    assert comm["bytes_per_round"] == 2 * num_clients * model_size_bytes
    assert comm["total_bytes"] == 2 * num_clients * model_size_bytes * num_rounds
    assert comm["total_mb"] == 100.0
    assert comm["total_rounds"] == 5


if __name__ == "__main__":
    test_evaluate_model_metrics()
    test_communication_cost_estimation()
    print("All metrics tests passed successfully!")
