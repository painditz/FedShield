"""
Unit tests for Byzantine Adversarial Attacks in FedShield.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.poisoning import apply_attack, SignFlippingAttack, UpdateScalingAttack, ZeroUpdateAttack


def test_sign_flipping():
    """Verify sign flipping inverts parameter delta."""
    global_params = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    local_params = {"weight": torch.tensor([1.5, 2.5, 3.5])}  # delta = [0.5, 0.5, 0.5]

    poisoned = apply_attack(
        local_parameters=local_params,
        global_parameters=global_params,
        attack_type="sign_flipping",
        scaling_factor=1.0
    )
    # expected: global - 1.0 * delta = [0.5, 1.5, 2.5]
    expected = torch.tensor([0.5, 1.5, 2.5])
    assert torch.allclose(poisoned["weight"], expected), f"Expected {expected}, got {poisoned['weight']}"


def test_update_scaling():
    """Verify update scaling drastically amplifies delta."""
    global_params = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    local_params = {"weight": torch.tensor([1.1, 2.1, 3.1])}  # delta = [0.1, 0.1, 0.1]

    poisoned = apply_attack(
        local_parameters=local_params,
        global_parameters=global_params,
        attack_type="update_scaling",
        scaling_factor=10.0
    )
    # expected: global + 10.0 * delta = [2.0, 3.0, 4.0]
    expected = torch.tensor([2.0, 3.0, 4.0])
    assert torch.allclose(poisoned["weight"], expected), f"Expected {expected}, got {poisoned['weight']}"


def test_zero_update():
    """Verify zero update keeps parameters identical to global."""
    global_params = {"weight": torch.tensor([5.0, 10.0])}
    local_params = {"weight": torch.tensor([6.0, 12.0])}

    poisoned = apply_attack(
        local_parameters=local_params,
        global_parameters=global_params,
        attack_type="zero_update"
    )
    assert torch.allclose(poisoned["weight"], global_params["weight"])


if __name__ == "__main__":
    test_sign_flipping()
    test_update_scaling()
    test_zero_update()
    print("All attack tests passed successfully!")
