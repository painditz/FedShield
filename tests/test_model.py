"""
Unit tests for Lightweight CNN model in FedShield.
Validates forward pass, output shape, parameter extraction/loading,
vector flattening/reconstruction, and size estimation.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn import LightweightCNN, create_model


def test_model_forward():
    """Verify forward pass with CIFAR-10 tensor dimensions."""
    model = create_model("lightweight_cnn", in_channels=3, num_classes=10)
    model.eval()

    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)

    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"


def test_parameter_get_and_set():
    """Verify get_parameters() returns clean tensors and set_parameters() updates weights."""
    model1 = create_model("lightweight_cnn")
    model2 = create_model("lightweight_cnn")

    # Change weights in model1
    with torch.no_grad():
        for p in model1.parameters():
            p.add_(1.0)

    params = model1.get_parameters()
    model2.set_parameters(params)

    # Check that model2 weights match model1 weights exactly
    p1 = model1.get_parameters()
    p2 = model2.get_parameters()

    for k in p1:
        assert torch.equal(p1[k], p2[k]), f"Mismatch in layer {k}"


def test_parameter_vectorization():
    """Verify vector flattening and reconstruction."""
    model = create_model("lightweight_cnn")
    vec = model.get_parameter_vector()

    assert vec.ndim == 1, "Parameter vector must be 1-dimensional"
    assert vec.numel() == model.count_parameters(), "Vector length must equal trainable parameter count"

    # Perturb vector and reload
    new_vec = vec + 0.5
    model.set_parameter_vector(new_vec)
    reloaded_vec = model.get_parameter_vector()

    assert torch.allclose(new_vec, reloaded_vec, atol=1e-6), "Vector reloading did not preserve values"


def test_model_size_computation():
    """Verify parameter count and serialized size computations."""
    model = create_model("lightweight_cnn")
    count = model.count_parameters()
    size_bytes = model.get_model_size_bytes()

    assert count > 0, "Parameter count must be positive"
    assert size_bytes > 0, "Model size in bytes must be positive"
    print(f"Verified CNN: {count} params, {size_bytes / 1024:.2f} KB")


if __name__ == "__main__":
    test_model_forward()
    test_parameter_get_and_set()
    test_parameter_vectorization()
    test_model_size_computation()
    print("All model tests passed successfully!")
