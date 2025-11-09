#!/usr/bin/env python3
"""
Test script to validate the new Macro-F1 loss functions.

This script tests:
1. SoftMacroF1Loss computation
2. CombinedMacroF1CELoss behavior
3. AdaptiveMacroF1CELoss scheduling
4. Comparison with standard CrossEntropyLoss
"""

import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Make sure the ml module is in the python path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.losses import SoftMacroF1Loss, CombinedMacroF1CELoss, AdaptiveMacroF1CELoss

def create_imbalanced_dataset(num_samples=1000, num_classes=11):
    """
    Create a synthetic imbalanced dataset similar to our traffic classification problem.
    Classes 5 and 7 are minority classes (like in the real data).
    """
    torch.manual_seed(42)  # For reproducibility

    # Create imbalanced class distribution
    class_sizes = []
    for i in range(num_classes):
        if i in [5, 7]:  # Minority classes
            class_sizes.append(np.random.randint(20, 100))  # 20-100 samples
        else:  # Majority classes
            class_sizes.append(np.random.randint(500, 1500))  # 500-1500 samples

    # Normalize to achieve roughly num_samples total
    total_samples = sum(class_sizes)
    class_sizes = [int(size * num_samples / total_samples) for size in class_sizes]

    # Create data
    data = []
    labels = []

    for class_idx, size in enumerate(class_sizes):
        # Create random features
        class_data = torch.randn(size, 50)  # 50-dimensional features

        # Add class-specific bias to make it somewhat learnable
        class_bias = torch.randn(50) * 2
        class_data = class_data + class_bias

        data.append(class_data)
        labels.extend([class_idx] * size)

    data = torch.cat(data, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    # Create a simple classifier
    model = nn.Sequential(
        nn.Linear(50, 100),
        nn.ReLU(),
        nn.Linear(100, num_classes)
    )

    return model, data, labels, class_sizes

def test_soft_macro_f1_loss():
    """Test SoftMacroF1Loss with synthetic data."""
    print("\n=== Testing SoftMacroF1Loss ===")

    num_classes = 11
    batch_size = 32

    # Create random logits and labels
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test loss computation
    loss_fn = SoftMacroF1Loss(num_classes)
    loss = loss_fn(logits, labels)

    print(f"SoftMacroF1Loss computed: {loss.item():.4f}")

    # Test with perfect predictions (should be 0)
    perfect_logits = torch.zeros(batch_size, num_classes)
    for i, label in enumerate(labels):
        perfect_logits[i, label] = 10.0  # High confidence for correct class

    perfect_loss = loss_fn(perfect_logits, labels)
    print(f"SoftMacroF1Loss with perfect predictions: {perfect_loss.item():.6f}")

    # Test with random predictions (should be higher)
    random_logits = torch.randn(batch_size, num_classes)
    random_loss = loss_fn(random_logits, labels)
    print(f"SoftMacroF1Loss with random predictions: {random_loss.item():.4f}")

def test_combined_loss():
    """Test CombinedMacroF1CELoss."""
    print("\n=== Testing CombinedMacroF1CELoss ===")

    num_classes = 11
    batch_size = 32

    # Create random logits and labels
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Test different lambda values
    for lambda_macro in [0.0, 0.3, 0.5, 0.7, 1.0]:
        loss_fn = CombinedMacroF1CELoss(num_classes, lambda_macro=lambda_macro)
        loss_dict = loss_fn(logits, labels)

        print(f"λ={lambda_macro}: Total={loss_dict['total_loss'].item():.4f}, "
              f"CE={loss_dict['ce_loss'].item():.4f}, "
              f"Macro-F1={loss_dict['macro_f1_loss'].item():.4f}")

def test_adaptive_loss():
    """Test AdaptiveMacroF1CELoss scheduling."""
    print("\n=== Testing AdaptiveMacroF1CELoss ===")

    num_classes = 11
    total_epochs = 10
    batch_size = 32

    # Create random logits and labels
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    loss_fn = AdaptiveMacroF1CELoss(
        num_classes=num_classes,
        initial_lambda=0.1,
        final_lambda=0.7,
        total_epochs=total_epochs
    )

    print("Epoch progression:")
    for epoch in range(total_epochs):
        loss_dict = loss_fn(logits, labels, current_epoch=epoch)
        print(f"Epoch {epoch+1:2d}: λ={loss_dict['current_lambda']:.3f}, "
              f"Total={loss_dict['total_loss'].item():.4f}")

def compare_with_ce_loss():
    """Compare training behavior with and without Macro-F1 loss."""
    print("\n=== Training Comparison: CE vs Combined Loss ===")

    # Create imbalanced dataset
    model_ce, data_ce, labels_ce, class_sizes = create_imbalanced_dataset(num_samples=800)
    model_combined, data_combined, labels_combined, _ = create_imbalanced_dataset(num_samples=800)

    print(f"Class distribution: {class_sizes}")
    print(f"Minority classes (5,7): {class_sizes[5]}, {class_sizes[7]}")

    # Create data loaders
    dataset_ce = TensorDataset(data_ce, labels_ce)
    dataset_combined = TensorDataset(data_combined, labels_combined)

    loader_ce = DataLoader(dataset_ce, batch_size=32, shuffle=True)
    loader_combined = DataLoader(dataset_combined, batch_size=32, shuffle=True)

    # Train with standard CE loss
    print("\nTraining with standard CE loss...")
    optimizer_ce = torch.optim.Adam(model_ce.parameters(), lr=0.01)
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in loader_ce:
            optimizer_ce.zero_grad()
            outputs = model_ce(batch_x)
            loss = ce_loss_fn(outputs, batch_y)
            loss.backward()
            optimizer_ce.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader_ce):.4f}, Acc={correct/total:.4f}")

    # Train with combined loss
    print("\nTraining with Combined CE + Macro-F1 loss (λ=0.5)...")
    optimizer_combined = torch.optim.Adam(model_combined.parameters(), lr=0.01)
    combined_loss_fn = CombinedMacroF1CELoss(num_classes=11, lambda_macro=0.5)

    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in loader_combined:
            optimizer_combined.zero_grad()
            outputs = model_combined(batch_x)
            loss_dict = combined_loss_fn(outputs, batch_y)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer_combined.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader_combined):.4f}, Acc={correct/total:.4f}")

def main():
    """Run all tests."""
    print("Testing Macro-F1 Loss Implementation")
    print("=" * 50)

    test_soft_macro_f1_loss()
    test_combined_loss()
    test_adaptive_loss()
    compare_with_ce_loss()

    print("\n" + "=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    main()