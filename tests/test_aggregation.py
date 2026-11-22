"""
Unit tests for Server Aggregation Operators in FedShield.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.aggregation import aggregate_fedavg, aggregate_coordinate_median, aggregate_trimmed_mean, aggregate_krum


def test_fedavg_aggregation():
    """Verify FedAvg calculates correct weighted average."""
    u1 = {"w": torch.tensor([1.0, 2.0])}
    u2 = {"w": torch.tensor([3.0, 4.0])}

    res_equal = aggregate_fedavg([u1, u2])
    assert torch.allclose(res_equal["w"], torch.tensor([2.0, 3.0]))

    # Weighted: u1 has 1 sample, u2 has 3 samples
    res_weighted = aggregate_fedavg([u1, u2], sample_counts=[1, 3])
    expected = (1.0 * torch.tensor([1.0, 2.0]) + 3.0 * torch.tensor([3.0, 4.0])) / 4.0
    assert torch.allclose(res_weighted["w"], expected)


def test_coordinate_median_defense():
    """Verify Coordinate Median rejects extreme outlier updates."""
    u1 = {"w": torch.tensor([10.0, 20.0])}
    u2 = {"w": torch.tensor([11.0, 21.0])}
    u3 = {"w": torch.tensor([10.5, 20.5])}
    # Outlier / malicious client scaling update by 100x
    u_malicious = {"w": torch.tensor([1000.0, 2000.0])}

    res_median = aggregate_coordinate_median([u1, u2, u3, u_malicious])
    # Median of [10.0, 10.5, 11.0, 1000.0] is (10.5 + 11.0)/2 = 10.75
    assert res_median["w"][0].item() < 15.0, "Median failed to reject extreme outlier"


def test_trimmed_mean_defense():
    """Verify Trimmed Mean trims extreme values."""
    u1 = {"w": torch.tensor([1.0])}
    u2 = {"w": torch.tensor([2.0])}
    u3 = {"w": torch.tensor([3.0])}
    u4 = {"w": torch.tensor([4.0])}
    u_mal = {"w": torch.tensor([100.0])}  # Outlier

    # 5 clients, beta=0.2 -> trims 1 lowest and 1 highest (1.0 and 100.0 trimmed), mean of [2.0, 3.0, 4.0] = 3.0
    res = aggregate_trimmed_mean([u1, u2, u3, u4, u_mal], beta=0.2)
    assert torch.isclose(res["w"], torch.tensor([3.0])), f"Expected 3.0, got {res['w'].item()}"


if __name__ == "__main__":
    test_fedavg_aggregation()
    test_coordinate_median_defense()
    test_trimmed_mean_defense()
    print("All aggregation tests passed successfully!")
