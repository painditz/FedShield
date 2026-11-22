"""
Evaluation Metrics & Communication Estimators for FedShield.
Calculates Accuracy, Precision, Recall, Macro F1, Confusion Matrix, Loss,
and computes realistic communication costs in Bytes, KB, and MB.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates a model over a DataLoader and computes:
    - Loss
    - Accuracy (%)
    - Precision (Macro)
    - Recall (Macro)
    - F1 Score (Macro)
    - Confusion Matrix (List of lists)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * data.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    total_samples = len(all_targets)
    if total_samples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": []
        }

    avg_loss = total_loss / total_samples
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = float(np.mean(all_preds == all_targets) * 100.0)
    precision = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    recall = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "sample_count": total_samples
    }


def compute_communication_cost(
    model_size_bytes: int,
    num_participating_clients: int,
    num_rounds: int = 1
) -> Dict[str, Any]:
    """
    Estimates communication cost in a federated round:
    - Downlink: Server transmits global model to K clients (K * model_size)
    - Uplink: K clients transmit local updates back to server (K * model_size)
    - Total per round: 2 * K * model_size
    """
    bytes_per_round = 2 * num_participating_clients * model_size_bytes
    total_bytes = bytes_per_round * num_rounds

    return {
        "model_size_bytes": model_size_bytes,
        "model_size_kb": round(model_size_bytes / 1024, 2),
        "model_size_mb": round(model_size_bytes / (1024 * 1024), 3),
        "clients_per_round": num_participating_clients,
        "bytes_per_round": bytes_per_round,
        "mb_per_round": round(bytes_per_round / (1024 * 1024), 2),
        "total_rounds": num_rounds,
        "total_bytes": total_bytes,
        "total_kb": round(total_bytes / 1024, 2),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "note": "Communication cost is an analytical estimate (2 * clients * model_size) for uplink & downlink."
    }
