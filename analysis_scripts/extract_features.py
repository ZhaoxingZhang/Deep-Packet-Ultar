#!/usr/bin/env python3
"""
Extract features and model parameters from trained GEE models

This script extracts:
1. Penultimate layer features (for PCA visualization)
2. Gating network weights
3. Softmax probability outputs
4. Output contributions from baseline and expert models
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_config,
    load_resnet_model,
    load_gating_network,
    load_test_data,
    FeatureExtractor,
    compute_softmax_entropy,
    compute_confidence,
    save_json,
    save_npz,
    ensure_dir,
    ProgressPrinter
)


class GEEModelExtractor:
    """
    Extract features and parameters from GEE models
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Initialize extractor

        Args:
            config: Configuration dictionary
            device: Device to run models on
        """
        self.config = config
        self.device = device
        self.models = {}
        self.data = {}

        # Output directories
        self.output_dir = config['output']['base_dir']
        ensure_dir(self.output_dir)

        pp = ProgressPrinter()
        pp.print_section("Initializing GEE Model Extractor")
        pp.print_info(f"Device: {device}")
        pp.print_info(f"Output directory: {self.output_dir}")

    def load_models(self, experiment: str, scenario: str):
        """
        Load all models for a specific experiment

        Args:
            experiment: Experiment name (e.g., 'traffic')
            scenario: Scenario name (e.g., 'incremental')
        """
        pp = ProgressPrinter()
        pp.print_step(1, 4, f"Loading models for {experiment}/{scenario}")

        exp_config = self.config['experiments'][experiment][scenario]

        # Load baseline model
        baseline_path = exp_config['baseline_model']
        pp.print_info(f"  Baseline: {baseline_path}")
        self.models['baseline'] = load_resnet_model(baseline_path, self.device)

        # Load expert model
        expert_path = exp_config['expert_model']
        pp.print_info(f"  Expert: {expert_path}")
        self.models['expert'] = load_resnet_model(expert_path, self.device)

        # Load gating network
        gating_path = exp_config['gating_network']
        has_garbage = exp_config.get('has_garbage_class', False)
        num_classes = self.models['baseline'].out.out_features

        pp.print_info(f"  Gating: {gating_path}")
        pp.print_info(f"  Has garbage class: {has_garbage}")
        self.models['gating'] = load_gating_network(
            gating_path,
            num_classes=num_classes,
            use_garbage_class=has_garbage,
            device=self.device
        )

        self.data['minority_classes'] = exp_config['minority_classes']
        self.data['num_classes'] = num_classes
        self.data['has_garbage_class'] = has_garbage

        pp.print_success("Models loaded successfully")

    def load_test_data(self, experiment: str, scenario: str):
        """
        Load test data

        Args:
            experiment: Experiment name
            scenario: Scenario name
        """
        pp = ProgressPrinter()
        pp.print_step(2, 4, f"Loading test data for {experiment}/{scenario}")

        exp_config = self.config['experiments'][experiment][scenario]
        data_path = exp_config['test_data']

        if not os.path.exists(data_path):
            pp.print_error(f"Test data not found: {data_path}")
            pp.print_info("Please download test data first using:")
            pp.print_info("  bash download_test_data.sh")
            raise FileNotFoundError(f"Test data not found: {data_path}")

        batch_size = self.config['analysis']['batch_size']
        num_workers = self.config['analysis']['num_workers']

        self.data['test_loader'] = load_test_data(
            data_path,
            batch_size=batch_size,
            num_workers=num_workers
        )

        pp.print_success(f"Test data loaded: {data_path}")

    def extract_penultimate_features(self):
        """
        Extract penultimate layer features from baseline and expert models
        Note: Gating network is skipped as it only outputs probability distributions
        """
        pp = ProgressPrinter()
        pp.print_step(3, 4, "Extracting penultimate layer features")

        features_dir = os.path.join(
            self.output_dir,
            self.config['output']['subdirs']['features']
        )
        ensure_dir(features_dir)

        # Extract features from baseline and expert only (gating is just a probability combiner)
        for model_name in ['baseline', 'expert']:
            pp.print_info(f"  Extracting from {model_name}...")

            # Create feature extractor
            extractor = FeatureExtractor(self.models[model_name], self.device)
            extractor.register_hook(layer_name="fc3")  # Penultimate layer

            # Extract features
            features, labels = extractor.extract(self.data['test_loader'])

            # Save features
            output_path = os.path.join(features_dir, f"{model_name}_features.npz")
            save_npz({
                'features': features,
                'labels': labels
            }, output_path)

            pp.print_info(f"    Shape: {features.shape}")
            pp.print_success(f"    Saved to: {output_path}")

    def extract_softmax_probs(self):
        """
        Extract Softmax probability outputs from all models
        """
        pp = ProgressPrinter()
        pp.print_step(4, 4, "Extracting Softmax probabilities")

        probs_dir = os.path.join(
            self.output_dir,
            self.config['output']['subdirs']['features']
        )
        ensure_dir(probs_dir)

        test_loader = self.data['test_loader']
        minority_classes = self.data['minority_classes']

        # Extract probabilities from baseline and expert
        for model_name in ['baseline', 'expert']:
            pp.print_info(f"  Extracting from {model_name}...")

            all_probs = []
            all_labels = []

            model = self.models[model_name]
            model.eval()

            with torch.no_grad():
                for features, labels in tqdm(test_loader, desc=f"  {model_name}"):
                    features = features.to(self.device)
                    output = model(features)
                    probs = torch.softmax(output, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(labels.numpy())

            probs = np.concatenate(all_probs, axis=0)
            labels = np.concatenate(all_labels, axis=0)

            # Compute entropy and confidence
            entropy = compute_softmax_entropy(probs)
            confidence = compute_confidence(probs)

            # Save results
            output_path = os.path.join(probs_dir, f"{model_name}_probs.npz")
            save_npz({
                'probs': probs,
                'labels': labels,
                'entropy': entropy,
                'confidence': confidence
            }, output_path)

            pp.print_info(f"    Probs shape: {probs.shape}")
            pp.print_info(f"    Mean entropy: {entropy.mean():.4f}")
            pp.print_info(f"    Mean confidence: {confidence.mean():.4f}")
            pp.print_success(f"    Saved to: {output_path}")

        # Extract probabilities from GEE model (gating network combining baseline and expert)
        pp.print_info(f"  Extracting from GEE (gating network)...")

        all_gating_probs = []
        all_labels = []

        baseline_model = self.models['baseline']
        expert_model = self.models['expert']
        gating_model = self.models['gating']

        baseline_model.eval()
        expert_model.eval()
        gating_model.eval()

        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="  gating"):
                features = features.to(self.device)

                # Get baseline and expert probabilities
                baseline_output = baseline_model(features)
                baseline_probs = torch.softmax(baseline_output, dim=1)

                expert_output = expert_model(features)
                expert_probs_small = torch.softmax(expert_output, dim=1)

                # Align expert probabilities to full class space
                expert_probs_full = torch.zeros_like(baseline_probs)
                for expert_idx, original_label in enumerate(minority_classes):
                    if expert_idx < expert_probs_small.shape[1]:
                        expert_probs_full[:, original_label] = expert_probs_small[:, expert_idx]

                # Get gating network output
                gating_output = gating_model(baseline_probs, expert_probs_full)
                gating_probs = torch.softmax(gating_output, dim=1)

                all_gating_probs.append(gating_probs.cpu().numpy())
                all_labels.append(labels.numpy())

        gating_probs = np.concatenate(all_gating_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Compute entropy and confidence
        entropy = compute_softmax_entropy(gating_probs)
        confidence = compute_confidence(gating_probs)

        # Save results
        output_path = os.path.join(probs_dir, "gating_probs.npz")
        save_npz({
            'probs': gating_probs,
            'labels': labels,
            'entropy': entropy,
            'confidence': confidence
        }, output_path)

        pp.print_info(f"    Probs shape: {gating_probs.shape}")
        pp.print_info(f"    Mean entropy: {entropy.mean():.4f}")
        pp.print_info(f"    Mean confidence: {confidence.mean():.4f}")
        pp.print_success(f"    Saved to: {output_path}")

    def extract_gating_weights(self):
        """
        Extract gating network weights
        """
        pp = ProgressPrinter()
        pp.print_section("Extracting Gating Network Weights")

        weights_dir = os.path.join(
            self.output_dir,
            self.config['output']['subdirs']['weights']
        )
        ensure_dir(weights_dir)

        gating_model = self.models['gating']

        # Extract first layer weights
        # The gating network structure: network[0] is the first Linear layer
        first_layer = gating_model.network[0]
        weight_matrix = first_layer.weight.data.cpu().numpy()  # (hidden_dim, input_dim)

        input_dim = weight_matrix.shape[1] // 2
        baseline_weights = weight_matrix[:, :input_dim]
        expert_weights = weight_matrix[:, input_dim:]

        # Compute statistics
        analysis = {
            'weight_matrix_shape': weight_matrix.shape,
            'baseline': {
                'mean_abs': float(np.abs(baseline_weights).mean()),
                'std_abs': float(np.abs(baseline_weights).std()),
                'per_dim_mean_abs': np.abs(baseline_weights).mean(axis=0).tolist()
            },
            'expert': {
                'mean_abs': float(np.abs(expert_weights).mean()),
                'std_abs': float(np.abs(expert_weights).std()),
                'per_dim_mean_abs': np.abs(expert_weights).mean(axis=0).tolist()
            },
            'minority_classes': {}
        }

        # Analyze minority class dimensions
        minority_classes = self.data['minority_classes']
        for minority_class in minority_classes:
            if minority_class < input_dim:
                baseline_weight = np.abs(baseline_weights[:, minority_class]).mean()
                expert_weight = np.abs(expert_weights[:, minority_class]).mean()
                weight_ratio = expert_weight / (baseline_weight + 1e-10)

                analysis['minority_classes'][str(minority_class)] = {
                    'baseline_weight': float(baseline_weight),
                    'expert_weight': float(expert_weight),
                    'weight_ratio': float(weight_ratio)
                }

        # Save analysis
        json_path = os.path.join(weights_dir, 'gating_network_weights.json')
        save_json(analysis, json_path)
        pp.print_success(f"Weight analysis saved to: {json_path}")

        # Save weight matrix
        npz_path = os.path.join(weights_dir, 'weight_matrix.npz')
        save_npz({'weight_matrix': weight_matrix}, npz_path)
        pp.print_success(f"Weight matrix saved to: {npz_path}")

        # Print summary
        pp.print_info("Summary:")
        pp.print_info(f"  Baseline mean abs weight: {analysis['baseline']['mean_abs']:.4f}")
        pp.print_info(f"  Expert mean abs weight: {analysis['expert']['mean_abs']:.4f}")

        for mc, stats in analysis['minority_classes'].items():
            pp.print_info(f"  Class {mc} weight ratio (expert/baseline): {stats['weight_ratio']:.2f}")

        return analysis

    def compute_output_contribution(self):
        """
        Compute output contribution from baseline and expert models
        """
        pp = ProgressPrinter()
        pp.print_section("Computing Output Contribution")

        weights_dir = os.path.join(
            self.output_dir,
            self.config['output']['subdirs']['weights']
        )
        ensure_dir(weights_dir)

        test_loader = self.data['test_loader']
        minority_classes = self.data['minority_classes']

        contributions = {
            'baseline': [],
            'expert': [],
            'labels': []
        }

        baseline_model = self.models['baseline']
        expert_model = self.models['expert']
        gating_model = self.models['gating']

        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="Computing contributions"):
                features = features.to(self.device)

                # Get predictions
                baseline_output = baseline_model(features)
                baseline_probs = torch.softmax(baseline_output, dim=1)

                expert_output = expert_model(features)
                expert_probs_small = torch.softmax(expert_output, dim=1)

                # Align expert probabilities to full class space
                expert_probs_full = torch.zeros_like(baseline_probs)
                for expert_idx, original_label in enumerate(minority_classes):
                    if expert_idx < expert_probs_small.shape[1]:
                        expert_probs_full[:, original_label] = expert_probs_small[:, expert_idx]

                # Get gating network output
                gating_output = gating_model(baseline_probs, expert_probs_full)
                gating_probs = torch.softmax(gating_output, dim=1)

                # Estimate contribution using correlation
                for i in range(features.size(0)):
                    b = baseline_probs[i].cpu().numpy()
                    e = expert_probs_full[i].cpu().numpy()
                    g = gating_probs[i].cpu().numpy()

                    # Compute correlation-based contribution
                    baseline_contrib = abs(np.corrcoef(b, g)[0, 1]) if len(b) > 1 else 0.5
                    expert_contrib = abs(np.corrcoef(e, g)[0, 1]) if len(e) > 1 else 0.5

                    # Normalize
                    total = abs(baseline_contrib) + abs(expert_contrib) + 1e-10
                    baseline_contrib = abs(baseline_contrib) / total
                    expert_contrib = abs(expert_contrib) / total

                    contributions['baseline'].append(baseline_contrib)
                    contributions['expert'].append(expert_contrib)
                    contributions['labels'].append(labels[i].item())

        # Compute per-class statistics
        labels = np.array(contributions['labels'])
        baseline_contrib = np.array(contributions['baseline'])
        expert_contrib = np.array(contributions['expert'])

        per_class_stats = {}
        for c in np.unique(labels):
            mask = labels == c
            per_class_stats[str(c)] = {
                'baseline_mean': float(baseline_contrib[mask].mean()),
                'expert_mean': float(expert_contrib[mask].mean()),
                'baseline_std': float(baseline_contrib[mask].std()),
                'expert_std': float(expert_contrib[mask].std()),
                'n_samples': int(mask.sum())
            }

        analysis = {
            'per_class': per_class_stats,
            'baseline_contrib': baseline_contrib.tolist(),
            'expert_contrib': expert_contrib.tolist(),
            'labels': labels.tolist()
        }

        # Save results
        json_path = os.path.join(weights_dir, 'output_contribution.json')
        save_json(analysis, json_path)
        pp.print_success(f"Output contribution saved to: {json_path}")

        # Print summary
        pp.print_info("Per-class contribution summary:")
        for c, stats in sorted(per_class_stats.items()):
            pp.print_info(f"  Class {c}: Baseline={stats['baseline_mean']:.3f}, Expert={stats['expert_mean']:.3f}")

        return analysis

    def run_full_extraction(self, experiment: str, scenario: str):
        """
        Run full extraction pipeline

        Args:
            experiment: Experiment name
            scenario: Scenario name
        """
        pp = ProgressPrinter()
        pp.print_section(f"Running Full Extraction: {experiment}/{scenario}")

        # Load models
        self.load_models(experiment, scenario)

        # Load test data
        self.load_test_data(experiment, scenario)

        # Extract features
        self.extract_penultimate_features()

        # Extract probabilities
        self.extract_softmax_probs()

        # Extract gating weights
        self.extract_gating_weights()

        # Compute output contribution
        self.compute_output_contribution()

        pp.print_section("Extraction Complete!")
        pp.print_success(f"All results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract features from GEE models")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--experiment", type=str, default="traffic",
                       help="Experiment name (traffic, traffic_v2, traffic_v3)")
    parser.add_argument("--scenario", type=str, default="incremental",
                       help="Scenario name (incremental, openset)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu, cuda, mps)")

    args = parser.parse_args()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Load config
    config = load_config(args.config)

    # Run extraction
    extractor = GEEModelExtractor(config, device=args.device)
    extractor.run_full_extraction(args.experiment, args.scenario)


if __name__ == "__main__":
    main()
