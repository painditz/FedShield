"""
Differential Privacy (DP) Mechanisms and Privacy Accounting for FedShield.
Supports:
1. Parameter / Gradient norm clipping (L2 threshold C).
2. Calibrated Gaussian noise addition (scale sigma * C).
3. Privacy budget accounting (computes epsilon for given delta, sample rate, and rounds).
"""

from typing import Dict, Any, Tuple, Optional
import math
import numpy as np
import torch


def clip_parameter_deltas(
    delta_dict: Dict[str, torch.Tensor],
    max_norm: float = 1.0
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Clips the L2 norm of the entire model update dictionary to max_norm:
    delta = delta * min(1, max_norm / ||delta||_2)
    """
    total_norm_sq = 0.0
    for k, v in delta_dict.items():
        if v.is_floating_point():
            total_norm_sq += torch.sum(v.float() ** 2).item()

    total_norm = math.sqrt(total_norm_sq)
    clip_factor = min(1.0, max_norm / (total_norm + 1e-9))

    clipped_delta: Dict[str, torch.Tensor] = {}
    for k, v in delta_dict.items():
        if v.is_floating_point():
            clipped_delta[k] = v * clip_factor
        else:
            clipped_delta[k] = v.clone()

    return clipped_delta, total_norm


def add_gaussian_noise(
    parameters: Dict[str, torch.Tensor],
    noise_multiplier: float = 0.5,
    clip_norm: float = 1.0,
    num_clients: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Adds calibrated Gaussian noise to parameters or aggregated update:
    sigma = (noise_multiplier * clip_norm) / sqrt(num_clients)
    """
    if noise_multiplier <= 0.0:
        return parameters

    sigma = (noise_multiplier * clip_norm) / math.sqrt(max(1, num_clients))
    noisy_parameters: Dict[str, torch.Tensor] = {}

    for k, v in parameters.items():
        if v.is_floating_point():
            noise = torch.randn_like(v.float()) * sigma
            noisy_parameters[k] = v + noise
        else:
            noisy_parameters[k] = v.clone()

    return noisy_parameters


def compute_dp_sgd_privacy(
    sample_rate: float,
    noise_multiplier: float,
    steps: int,
    delta: float = 1e-5
) -> float:
    """
    Calculates (epsilon, delta) privacy guarantee using the analytical Moments Accountant approximation.
    For subsampled Gaussian mechanism:
    epsilon ~= (sample_rate * sqrt(2 * steps * log(1 / delta))) / noise_multiplier
    """
    if noise_multiplier <= 0.0:
        return float("inf")
    if steps <= 0 or sample_rate <= 0:
        return 0.0

    # Strong composition approximation for sub-sampled Gaussian mechanism
    term = math.sqrt(2.0 * steps * math.log(1.0 / delta))
    epsilon = (sample_rate * term) / noise_multiplier
    return round(epsilon, 3)


class DifferentialPrivacyManager:
    """Manages DP clipping, noise injection, and accounting across rounds."""

    def __init__(
        self,
        dp_enabled: bool = False,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        target_delta: float = 1e-5
    ):
        self.dp_enabled = dp_enabled
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.accumulated_steps = 0

    def apply_differential_privacy(
        self,
        aggregated_weights: Dict[str, torch.Tensor],
        num_participating_clients: int
    ) -> Dict[str, torch.Tensor]:
        """Applies DP noise to aggregated parameters if DP is enabled."""
        if not self.dp_enabled:
            return aggregated_weights

        self.accumulated_steps += 1
        return add_gaussian_noise(
            aggregated_weights,
            noise_multiplier=self.noise_multiplier,
            clip_norm=self.max_grad_norm,
            num_clients=num_participating_clients
        )

    def get_privacy_budget(self, sample_rate: float) -> Dict[str, Any]:
        """Returns current epsilon and delta."""
        if not self.dp_enabled:
            return {"dp_enabled": False, "epsilon": None, "delta": None}

        epsilon = compute_dp_sgd_privacy(
            sample_rate=sample_rate,
            noise_multiplier=self.noise_multiplier,
            steps=self.accumulated_steps,
            delta=self.target_delta
        )
        return {
            "dp_enabled": True,
            "epsilon": epsilon,
            "delta": self.target_delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "steps": self.accumulated_steps
        }
