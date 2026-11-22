"""
Byzantine Adversarial Attack Simulation for FedShield.
Supports model poisoning strategies designed to degrade global convergence:
1. Sign Flipping Attack: Inverts parameter delta direction (-s * delta).
2. Update Scaling Attack: Drastically amplifies update magnitudes (s * delta).
3. Random Noise Attack: Injects high-variance Gaussian noise into updates.
4. Zero Gradient / Freeze Attack: Sends zero update or cancels updates.
"""

from typing import Dict, Any, Optional
import torch


class AdversarialAttack:
    """Base class for all federated attacks."""
    def attack(
        self,
        local_parameters: Dict[str, torch.Tensor],
        global_parameters: Dict[str, torch.Tensor],
        scaling_factor: float = 10.0
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class SignFlippingAttack(AdversarialAttack):
    """
    Inverts parameter update direction with optional scaling:
    w_poisoned = w_global - scaling * (w_local - w_global)
    """
    def attack(
        self,
        local_parameters: Dict[str, torch.Tensor],
        global_parameters: Dict[str, torch.Tensor],
        scaling_factor: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        poisoned: Dict[str, torch.Tensor] = {}
        for k in global_parameters.keys():
            delta = local_parameters[k] - global_parameters[k].cpu()
            poisoned[k] = global_parameters[k].cpu() - (scaling_factor * delta)
        return poisoned


class UpdateScalingAttack(AdversarialAttack):
    """
    Amplifies update magnitude by a large scaling factor to dominate aggregation:
    w_poisoned = w_global + scaling * (w_local - w_global)
    """
    def attack(
        self,
        local_parameters: Dict[str, torch.Tensor],
        global_parameters: Dict[str, torch.Tensor],
        scaling_factor: float = 10.0
    ) -> Dict[str, torch.Tensor]:
        poisoned: Dict[str, torch.Tensor] = {}
        for k in global_parameters.keys():
            delta = local_parameters[k] - global_parameters[k].cpu()
            poisoned[k] = global_parameters[k].cpu() + (scaling_factor * delta)
        return poisoned


class RandomNoiseAttack(AdversarialAttack):
    """
    Replaces client update with large-variance Gaussian noise.
    """
    def attack(
        self,
        local_parameters: Dict[str, torch.Tensor],
        global_parameters: Dict[str, torch.Tensor],
        scaling_factor: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        poisoned: Dict[str, torch.Tensor] = {}
        for k in global_parameters.keys():
            param_std = local_parameters[k].float().std().item() if local_parameters[k].is_floating_point() else 1.0
            noise = torch.randn_like(local_parameters[k].float()) * (param_std * scaling_factor)
            if local_parameters[k].is_floating_point():
                poisoned[k] = global_parameters[k].cpu() + noise
            else:
                poisoned[k] = local_parameters[k].clone()
        return poisoned


class ZeroUpdateAttack(AdversarialAttack):
    """
    Sends back the unchanged global model, simulating straggler or drop-out poisoning.
    """
    def attack(
        self,
        local_parameters: Dict[str, torch.Tensor],
        global_parameters: Dict[str, torch.Tensor],
        scaling_factor: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        return {k: v.clone().cpu() for k, v in global_parameters.items()}


ATTACK_REGISTRY = {
    "sign_flipping": SignFlippingAttack(),
    "update_scaling": UpdateScalingAttack(),
    "random_noise": RandomNoiseAttack(),
    "zero_update": ZeroUpdateAttack()
}


def apply_attack(
    local_parameters: Dict[str, torch.Tensor],
    global_parameters: Dict[str, torch.Tensor],
    attack_type: str = "sign_flipping",
    scaling_factor: float = 10.0
) -> Dict[str, torch.Tensor]:
    """
    Applies the specified attack type to the client's local parameters.
    """
    attack_obj = ATTACK_REGISTRY.get(attack_type.lower())
    if attack_obj is None:
        raise ValueError(f"Unknown attack type: {attack_type}. Available: {list(ATTACK_REGISTRY.keys())}")
    return attack_obj.attack(local_parameters, global_parameters, scaling_factor)
