"""
Lightweight Modular CNN for CIFAR-10.
Designed for Federated Learning research: includes serialization, parameter vectorization,
size estimation, and gradient inspection methods.
"""

from typing import Dict, List, OrderedDict, Any
import io
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNN(nn.Module):
    """
    Lightweight 3-stage Convolutional Neural Network:
    Input (3, 32, 32)
    -> Conv2D (3 -> 32) -> BatchNorm -> ReLU -> MaxPool (16x16)
    -> Conv2D (32 -> 64) -> BatchNorm -> ReLU -> MaxPool (8x8)
    -> Conv2D (64 -> 128) -> ReLU -> AdaptiveAvgPool2d (4x4)
    -> Flatten -> FC (128*4*4 -> 128) -> ReLU -> FC (128 -> num_classes)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(LightweightCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        # Flatten
        x = torch.flatten(x, 1)
        # Linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns deep copied state dictionary detached on CPU."""
        return {k: v.clone().detach().cpu() for k, v in self.state_dict().items()}

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Loads state dictionary into model, keeping target device."""
        device = next(self.parameters()).device
        state = {k: v.to(device) for k, v in parameters.items()}
        self.load_state_dict(state)

    def get_parameter_vector(self) -> torch.Tensor:
        """Returns all trainable parameters flattened into a 1D vector."""
        return torch.cat([p.detach().cpu().flatten() for p in self.parameters() if p.requires_grad])

    def set_parameter_vector(self, vector: torch.Tensor) -> None:
        """Sets trainable parameters from a 1D flattened vector."""
        device = next(self.parameters()).device
        vector = vector.to(device)
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                if p.requires_grad:
                    numel = p.numel()
                    p.copy_(vector[offset:offset + numel].view_as(p))
                    offset += numel

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_bytes(self) -> int:
        """
        Calculates exact serialization size of model parameters in bytes.
        Used for realistic communication cost computation.
        """
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.getbuffer().nbytes


def create_model(model_name: str = "lightweight_cnn", in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """Factory function for model creation."""
    if model_name == "lightweight_cnn":
        return LightweightCNN(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
