#!/usr/bin/env python3
"""
Utility functions for GEE model analysis
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

# Add parent directory to path to import ml modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.model import ResNet, CNN, GatingNetwork


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_resnet_model(model_path: str, device: str = "cpu") -> ResNet:
    """
    Load a ResNet model from checkpoint

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded ResNet model in eval mode
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ResNet.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_cnn_model(model_path: str, device: str = "cpu") -> CNN:
    """
    Load a CNN model from checkpoint

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded CNN model in eval mode
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = CNN.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_gating_network(model_path: str,
                        num_classes: int,
                        use_garbage_class: bool = False,
                        device: str = "cpu") -> GatingNetwork:
    """
    Load a GatingNetwork model from state dict

    Args:
        model_path: Path to model state dict
        num_classes: Number of output classes
        use_garbage_class: Whether the model uses garbage class
        device: Device to load model on

    Returns:
        Loaded GatingNetwork in eval mode
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Gating network not found: {model_path}")

    gating_network = GatingNetwork(
        num_classes=num_classes,
        use_garbage_class=use_garbage_class
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    gating_network.load_state_dict(state_dict)
    gating_network.eval()

    # Freeze parameters
    for param in gating_network.parameters():
        param.requires_grad = False

    return gating_network


def load_test_data(data_path: str,
                   batch_size: int = 64,
                   num_workers: int = 0,
                   shuffle: bool = False) -> torch.utils.data.DataLoader:
    """
    Load test data from parquet file

    Args:
        data_path: Path to test data parquet file
        batch_size: Batch size for dataloader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data

    Returns:
        DataLoader for test data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found: {data_path}")

    import pyarrow.parquet as pq
    from torch.utils.data import DataLoader, TensorDataset

    # Load data
    table = pq.read_table(data_path)
    df = table.to_pandas()

    # Extract features and labels
    features = torch.from_numpy(np.array(df['feature'].tolist(), dtype=np.float32))
    labels = torch.from_numpy(np.array(df['label'].tolist(), dtype=np.int64))

    # Add channel dimension if needed
    if len(features.shape) == 2:
        features = features.unsqueeze(1)

    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader


def compute_softmax_entropy(probs: np.ndarray,
                           eps: float = 1e-10) -> np.ndarray:
    """
    Compute Softmax entropy for probability distributions

    Args:
        probs: Probability array of shape (n_samples, n_classes)
        eps: Small value to avoid log(0)

    Returns:
        Entropy values of shape (n_samples,)
    """
    return -np.sum(probs * np.log(probs + eps), axis=1)


def compute_confidence(probs: np.ndarray) -> np.ndarray:
    """
    Compute confidence (max probability) for probability distributions

    Args:
        probs: Probability array of shape (n_samples, n_classes)

    Returns:
        Confidence values of shape (n_samples,)
    """
    return np.max(probs, axis=1)


def ensure_dir(directory: str) -> str:
    """
    Ensure directory exists, create if it doesn't

    Args:
        directory: Directory path

    Returns:
        Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Dict, file_path: str) -> None:
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)


def load_json(file_path: str) -> Dict:
    """
    Load dictionary from JSON file

    Args:
        file_path: Input file path

    Returns:
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_npz(data: Dict[str, np.ndarray], file_path: str) -> None:
    """
    Save numpy arrays to NPZ file

    Args:
        data: Dictionary of numpy arrays
        file_path: Output file path
    """
    ensure_dir(os.path.dirname(file_path))
    np.savez_compressed(file_path, **data)


def load_npz(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load numpy arrays from NPZ file

    Args:
        file_path: Input file path

    Returns:
        Dictionary of numpy arrays
    """
    return dict(np.load(file_path))


class FeatureExtractor:
    """
    Utility class for extracting features from models
    """

    def __init__(self, model, device: str = "cpu"):
        """
        Initialize feature extractor

        Args:
            model: PyTorch model
            device: Device to run model on
        """
        self.model = model
        self.device = device
        self.features = None
        self.hook_handle = None

    def register_hook(self, layer_name: str = "fc3"):
        """
        Register forward hook to capture intermediate layer output

        Args:
            layer_name: Name of the layer to extract features from
        """
        self.features = []

        def hook_fn(module, input, output):
            # Handle different output formats
            if isinstance(output, torch.Tensor):
                self.features.append(output.cpu().detach())
            else:
                self.features.append(output[0].cpu().detach())

        # Find the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if layer_name in name:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")

        self.hook_handle = target_layer.register_forward_hook(hook_fn)

    def extract(self, dataloader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from all samples in dataloader

        Args:
            dataloader: Data loader

        Returns:
            Tuple of (features, labels)
        """
        if self.hook_handle is None:
            raise RuntimeError("Hook not registered. Call register_hook() first.")

        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(dataloader, desc="Extracting features"):
                features = features.to(self.device)
                _ = self.model(features)
                all_labels.append(labels.cpu())

        # Remove hook
        self.hook_handle.remove()
        self.hook_handle = None

        # Concatenate all features
        extracted_features = torch.cat(self.features, dim=0).numpy()
        extracted_labels = torch.cat(all_labels, dim=0).numpy()

        self.features = None  # Clear memory

        return extracted_features, extracted_labels


class ProgressPrinter:
    """
    Utility class for printing progress information
    """

    @staticmethod
    def print_section(title: str):
        """Print a section header"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")

    @staticmethod
    def print_step(step: int, total: int, description: str):
        """Print a step indicator"""
        print(f"[{step}/{total}] {description}")

    @staticmethod
    def print_success(message: str):
        """Print a success message"""
        print(f"✓ {message}")

    @staticmethod
    def print_error(message: str):
        """Print an error message"""
        print(f"✗ {message}")

    @staticmethod
    def print_info(message: str):
        """Print an info message"""
        print(f"  {message}")


def get_class_name(class_id: int,
                   traffic_type_dict: Dict = None) -> str:
    """
    Get traffic type name from class ID

    Args:
        class_id: Class ID
        traffic_type_dict: Optional dictionary mapping class IDs to names

    Returns:
        Class name string
    """
    if traffic_type_dict is None:
        traffic_type_dict = {
            0: "Chat",
            1: "Email",
            2: "File Transfer",
            3: "Streaming",
            4: "VoIP",
            5: "VPN: Chat",
            6: "VPN: File Transfer",
            7: "VPN: Email",
            8: "VPN: Streaming",
            9: "VPN: Torrent",
            10: "VPN: VoIP",
        }

    return traffic_type_dict.get(class_id, f"Unknown({class_id})")
