"""
Server Aggregation Operators for Federated Learning in FedShield.
Supports standard and Byzantine-robust aggregation algorithms:
1. FedAvg: Standard sample-weighted average.
2. Coordinate Median: Coordinate-wise median, impervious to extreme value scaling.
3. Trimmed Mean: Trims the lowest and highest beta fraction of values per coordinate.
4. Krum / Multi-Krum: Byzantine-tolerant selection based on minimal Euclidean neighborhood distances.
"""

from typing import List, Dict, Optional, Any
import numpy as np
import torch


def aggregate_fedavg(
    updates: List[Dict[str, torch.Tensor]],
    sample_counts: Optional[List[int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Standard Federated Averaging (FedAvg):
    Weighted sum of client parameters proportional to local sample counts.
    w_global = sum_{k} (n_k / N) * w_k
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of updates.")

    num_clients = len(updates)
    if sample_counts is None:
        sample_counts = [1] * num_clients

    total_samples = sum(sample_counts)
    weights = [count / total_samples for count in sample_counts]

    aggregated: Dict[str, torch.Tensor] = {}
    keys = updates[0].keys()

    for k in keys:
        if updates[0][k].is_floating_point():
            weighted_sum = torch.zeros_like(updates[0][k], dtype=torch.float32)
            for i, update in enumerate(updates):
                weighted_sum += weights[i] * update[k].to(torch.float32)
            aggregated[k] = weighted_sum
        else:
            # For non-floating point buffers (e.g., num_batches_tracked in BatchNorm)
            aggregated[k] = updates[0][k].clone()

    return aggregated


def aggregate_coordinate_median(
    updates: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise Median aggregation:
    Calculates median along the client dimension for every single weight tensor coordinate.
    Extremely robust against single-client extreme value scaling / outlier attacks.
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of updates.")

    aggregated: Dict[str, torch.Tensor] = {}
    keys = updates[0].keys()

    for k in keys:
        if updates[0][k].is_floating_point():
            # Stack along client dimension: shape (num_clients, *param_shape)
            stacked = torch.stack([u[k].to(torch.float32) for u in updates], dim=0)
            median_val, _ = torch.median(stacked, dim=0)
            aggregated[k] = median_val
        else:
            aggregated[k] = updates[0][k].clone()

    return aggregated


def aggregate_trimmed_mean(
    updates: List[Dict[str, torch.Tensor]],
    beta: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise Trimmed Mean aggregation:
    For each parameter coordinate, trims the lowest and highest beta fraction of values
    before computing the arithmetic mean of remaining values.
    beta: fraction of clients to trim from each end (0.0 <= beta < 0.5)
    """
    if not updates:
        raise ValueError("Cannot aggregate an empty list of updates.")

    num_clients = len(updates)
    k_trim = max(1, int(np.round(beta * num_clients)))

    # If trimming leaves no or fewer than 1 client, fall back to simple mean
    if 2 * k_trim >= num_clients:
        k_trim = max(0, (num_clients - 1) // 2)

    aggregated: Dict[str, torch.Tensor] = {}
    keys = updates[0].keys()

    for k in keys:
        if updates[0][k].is_floating_point():
            # Stack along client dimension (num_clients, *param_shape)
            stacked = torch.stack([u[k].to(torch.float32) for u in updates], dim=0)
            if k_trim > 0:
                sorted_vals, _ = torch.sort(stacked, dim=0)
                # Trim lowest and highest k_trim values
                trimmed = sorted_vals[k_trim : num_clients - k_trim]
                aggregated[k] = torch.mean(trimmed, dim=0)
            else:
                aggregated[k] = torch.mean(stacked, dim=0)
        else:
            aggregated[k] = updates[0][k].clone()

    return aggregated


def aggregate_krum(
    updates: List[Dict[str, torch.Tensor]],
    num_malicious: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Krum Aggregation:
    Computes pairwise Euclidean distances between client update vectors.
    Selects the client vector that minimizes the sum of squared distances to its
    closest (n - f - 2) neighbors, where n is total clients and f is assumed malicious.
    """
    num_clients = len(updates)
    if num_clients <= 2:
        return aggregate_fedavg(updates)

    # Flatten each client update into a single 1D tensor
    vectors = []
    for u in updates:
        flat_parts = [u[k].flatten().to(torch.float32) for k in u.keys() if u[k].is_floating_point()]
        vectors.append(torch.cat(flat_parts))

    # Compute pairwise distance matrix
    dist_matrix = torch.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = torch.norm(vectors[i] - vectors[j]).item()
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Number of closest neighbors to consider: n - f - 2
    nb_neighbors = max(1, num_clients - num_malicious - 2)
    scores = []
    for i in range(num_clients):
        sorted_dists, _ = torch.sort(dist_matrix[i])
        # Skip distance to itself at index 0, sum closest nb_neighbors
        score = torch.sum(sorted_dists[1 : nb_neighbors + 1]).item()
        scores.append(score)

    best_idx = int(np.argmin(scores))
    return {k: v.clone() for k, v in updates[best_idx].items()}


AGGREGATION_REGISTRY = {
    "fedavg": aggregate_fedavg,
    "coordinate_median": aggregate_coordinate_median,
    "median": aggregate_coordinate_median,
    "trimmed_mean": aggregate_trimmed_mean,
    "krum": aggregate_krum
}


def aggregate(
    updates: List[Dict[str, torch.Tensor]],
    method: str = "fedavg",
    sample_counts: Optional[List[int]] = None,
    trimmed_mean_beta: float = 0.1,
    num_malicious: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Aggregation dispatcher for federated server.
    """
    norm_method = method.lower()
    if norm_method == "fedavg":
        return aggregate_fedavg(updates, sample_counts=sample_counts)
    elif norm_method in ["coordinate_median", "median"]:
        return aggregate_coordinate_median(updates)
    elif norm_method == "trimmed_mean":
        return aggregate_trimmed_mean(updates, beta=trimmed_mean_beta)
    elif norm_method == "krum":
        return aggregate_krum(updates, num_malicious=num_malicious)
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Available: {list(AGGREGATION_REGISTRY.keys())}")
