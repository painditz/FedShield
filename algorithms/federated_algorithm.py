"""
Federated Learning Algorithms for FedShield.
Encapsulates algorithm-specific client optimization and server coordination:
1. FedAvg: Standard Federated Averaging (McMahan et al., 2017).
2. FedProx: Federated Optimization in Heterogeneous Networks (Li et al., 2020) with proximal term.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from clients.client import Client


class FederatedAlgorithm:
    """Base class for Federated Learning algorithms."""
    name: str = "base"

    def __init__(self, **kwargs):
        self.params = kwargs

    def train_client(
        self,
        client: Client,
        global_parameters: Dict[str, torch.Tensor],
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
        dp_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError


class FedAvg(FederatedAlgorithm):
    """
    FedAvg: Standard federated averaging with local SGD/Adam.
    """
    name: str = "fedavg"

    def train_client(
        self,
        client: Client,
        global_parameters: Dict[str, torch.Tensor],
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
        dp_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        return client.train(
            global_parameters=global_parameters,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            algorithm="fedavg",
            fedprox_mu=0.0,
            dp_config=dp_config
        )


class FedProx(FederatedAlgorithm):
    """
    FedProx: Tackles system and statistical heterogeneity by adding a proximal term
    to the client objective: (mu / 2) * ||w - w_global||^2
    """
    name: str = "fedprox"

    def __init__(self, mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu

    def train_client(
        self,
        client: Client,
        global_parameters: Dict[str, torch.Tensor],
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
        dp_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        return client.train(
            global_parameters=global_parameters,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            algorithm="fedprox",
            fedprox_mu=self.mu,
            dp_config=dp_config
        )


def get_algorithm(name: str, **kwargs) -> FederatedAlgorithm:
    """Factory function for FL algorithms."""
    name = name.lower()
    if name == "fedavg":
        return FedAvg(**kwargs)
    elif name == "fedprox":
        mu = kwargs.get("fedprox_mu", 0.01)
        return FedProx(mu=mu, **kwargs)
    else:
        raise ValueError(f"Unknown federated algorithm: {name}. Available: ['fedavg', 'fedprox']")
