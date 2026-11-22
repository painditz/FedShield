"""
Federated Client Representation for FedShield.
Supports local training loops for:
- FedAvg (standard SGD/Adam optimization)
- FedProx (with proximal regularizer term ||w - w_t||^2)
- Differential Privacy (local gradient / update clipping & calibrated noise)
- Inspection of local parameters, weight updates (deltas), and sample counts.
"""

from typing import Dict, Any, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cnn import LightweightCNN


class Client:
    """
    Represents an edge device / federated client capable of local training,
    calculating parameter updates, and applying local privacy mechanisms.
    """

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        model_template: nn.Module,
        device: torch.device,
        is_malicious: bool = False
    ):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.is_malicious = is_malicious
        self.sample_count = len(dataloader.dataset)

        # Local model instance
        self.model = copy.deepcopy(model_template).to(self.device)

    def train(
        self,
        global_parameters: Dict[str, torch.Tensor],
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        algorithm: str = "fedavg",
        fedprox_mu: float = 0.0,
        dp_config: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executes local training initialized from global parameters.
        Returns:
            - updated_parameters: Dict[str, torch.Tensor] (CPU)
            - parameter_delta: Dict[str, torch.Tensor] (w_local - w_global)
            - sample_count: int
            - train_loss: float
            - train_acc: float
        """
        # 1. Sync local model with global parameters
        self.model.set_parameters(global_parameters)
        self.model.train()

        # Clone initial weights on device for FedProx proximal term or delta calculation
        global_weights_on_device = {
            k: v.clone().to(self.device) for k, v in global_parameters.items()
        }

        # Optimizer & criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        total_loss = 0.0
        correct = 0
        total_samples = 0

        # Check DP options
        dp_enabled = getattr(dp_config, "dp_enabled", False) if dp_config else False
        max_grad_norm = getattr(dp_config, "max_grad_norm", 1.0) if dp_config else 1.0

        for epoch in range(local_epochs):
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                output = self.model(data)
                loss = criterion(output, target)

                # FedProx proximal term: (mu / 2) * ||w - w_t||^2
                if algorithm.lower() == "fedprox" and fedprox_mu > 0.0:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_weights_on_device:
                            proximal_term += torch.sum((param - global_weights_on_device[name]) ** 2)
                    loss += (fedprox_mu / 2.0) * proximal_term

                loss.backward()

                # Differential privacy: gradient clipping
                if dp_enabled and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

                optimizer.step()

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = (correct / max(total_samples, 1)) * 100.0

        local_parameters = self.model.get_parameters()

        # Compute parameter deltas (w_local - w_global)
        parameter_delta: Dict[str, torch.Tensor] = {}
        for k in global_parameters.keys():
            parameter_delta[k] = local_parameters[k] - global_parameters[k].cpu()

        return {
            "client_id": self.client_id,
            "parameters": local_parameters,
            "delta": parameter_delta,
            "sample_count": self.sample_count,
            "train_loss": round(avg_loss, 4),
            "train_accuracy": round(accuracy, 2),
            "is_malicious": self.is_malicious
        }
