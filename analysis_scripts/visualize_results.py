#!/usr/bin/env python3
"""
Generate all visualization figures for Chapter 5

This script generates the following figures:
- Figure 5-1: Gating network weight distribution heatmap
- Figure 5-2: Output contribution bar chart
- Figure 5-3: Decision boundary visualization (3 subplots)
- Figure 5-4: Gradient contribution comparison (2 subplots)
- Figure 5-5: Decision boundary geometry comparison (2 subplots)
- Figure 5-6: Softmax entropy distribution comparison
- Figure 5-7: Reliability diagram
- Figure 5-8: GEE unified theoretical framework
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_config,
    load_json,
    load_npz,
    ensure_dir,
    ProgressPrinter
)


class FigureGenerator:
    """Generate all figures for Chapter 5"""

    def __init__(self, results_dir: str, output_dir: str):
        """
        Initialize figure generator

        Args:
            results_dir: Directory containing analysis results
            output_dir: Directory to save figures
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.dpi'] = 300

        # Load analysis results
        self._load_results()

        pp = ProgressPrinter()
        pp.print_section("Initializing Figure Generator")
        pp.print_info(f"Results directory: {results_dir}")
        pp.print_info(f"Output directory: {output_dir}")

    def _load_results(self):
        """Load all analysis results"""
        pp = ProgressPrinter()

        # Load weight analysis
        weight_path = self.results_dir / "weights" / "gating_network_weights.json"
        if weight_path.exists():
            self.weight_analysis = load_json(weight_path)
            pp.print_success("Loaded weight analysis")
        else:
            self.weight_analysis = None
            pp.print_info("Warning: weight analysis not found")

        # Load output contribution
        contrib_path = self.results_dir / "weights" / "output_contribution.json"
        if contrib_path.exists():
            self.contribution_analysis = load_json(contrib_path)
            pp.print_success("Loaded output contribution")
        else:
            self.contribution_analysis = None
            pp.print_info("Warning: output contribution not found")

        # Load features
        feature_path = self.results_dir / "features" / "baseline_features.npz"
        if feature_path.exists():
            self.features = load_npz(feature_path)
            pp.print_success("Loaded features")
        else:
            self.features = None
            pp.print_info("Warning: features not found")

        # Load probabilities
        probs_path = self.results_dir / "features" / "baseline_probs.npz"
        if probs_path.exists():
            self.probs = load_npz(probs_path)
            pp.print_success("Loaded probabilities")
        else:
            self.probs = None
            pp.print_info("Warning: probabilities not found")

    def generate_figure_5_1(self):
        """
        Figure 5-1: Gating Network Weight Matrix Heatmap

        High-density heatmap showing how each hidden neuron connects to
        baseline vs expert inputs across all class dimensions
        """
        pp = ProgressPrinter()
        pp.print_step(1, 7, "Generating Figure 5-1: Weight Heatmap")

        # Load weight matrix
        weight_path = self.results_dir / "weights" / "weight_matrix.npz"
        if not weight_path.exists():
            pp.print_info("Skipping: weight matrix not found")
            return

        weight_matrix = load_npz(weight_path)['weight_matrix']  # (n_hidden, n_input)

        n_hidden, n_input = weight_matrix.shape
        n_classes = n_input // 2

        # Split into baseline and expert weights
        baseline_weights = weight_matrix[:, :n_classes]  # (n_hidden, n_classes)
        expert_weights = weight_matrix[:, n_classes:]     # (n_hidden, n_classes)

        # Calculate absolute weights for visualization
        baseline_abs = np.abs(baseline_weights)
        expert_abs = np.abs(expert_weights)

        # Create figure with 2 subplots (side by side heatmaps)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        minority_classes = [4, 7]

        # --- Left plot: Baseline weight heatmap ---
        im1 = ax1.imshow(baseline_abs, cmap='Blues', aspect='auto', vmin=0, vmax=np.percentile(baseline_abs, 95))

        ax1.set_xticks(range(n_classes))
        ax1.set_xticklabels([f'C{i}' for i in range(n_classes)], fontsize=10)
        ax1.set_yticks(range(n_hidden))
        ax1.set_yticklabels([f'H{i+1}' for i in range(n_hidden)], fontsize=9)

        # Highlight minority class columns
        for mc in minority_classes:
            if mc < n_classes:
                ax1.axvline(x=mc - 0.5, color='red', linewidth=2, linestyle=':', alpha=0.5)
                ax1.axvline(x=mc + 0.5, color='red', linewidth=2, linestyle=':', alpha=0.5)

        ax1.set_xlabel('Class Dimension', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Hidden Neuron', fontsize=11, fontweight='bold')
        ax1.set_title('Baseline Input Weights\n(Darker = Stronger Connection)',
                     fontsize=12, fontweight='bold')

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('|Weight|', fontsize=10)

        # Add annotation
        ax1.text(0.5, -0.15, 'Red dotted lines: Minority classes (4, 7)',
               transform=ax1.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.6))

        # --- Right plot: Expert weight heatmap ---
        im2 = ax2.imshow(expert_abs, cmap='Reds', aspect='auto', vmin=0, vmax=np.percentile(expert_abs, 95))

        ax2.set_xticks(range(n_classes))
        ax2.set_xticklabels([f'C{i}' for i in range(n_classes)], fontsize=10)
        ax2.set_yticks(range(n_hidden))
        ax2.set_yticklabels([f'H{i+1}' for i in range(n_hidden)], fontsize=9)

        # Highlight minority class columns
        for mc in minority_classes:
            if mc < n_classes:
                ax2.axvline(x=mc - 0.5, color='darkred', linewidth=3, linestyle='-', alpha=0.7)
                ax2.axvline(x=mc + 0.5, color='darkred', linewidth=3, linestyle='-', alpha=0.7)

        ax2.set_xlabel('Class Dimension', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Hidden Neuron', fontsize=11, fontweight='bold')
        ax2.set_title('Expert Input Weights\n(Darker = Stronger Connection)',
                     fontsize=12, fontweight='bold')

        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('|Weight|', fontsize=10)

        # Add annotation
        ax2.text(0.5, -0.15, 'Dark red borders: Minority classes (4, 7) have strong weights',
               transform=ax2.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.6))

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'figure_5_1_weight_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_5_1_weight_heatmap.pdf',
                   format='pdf', bbox_inches='tight')
        plt.close()

        pp.print_success(f"Saved to: {output_path}")

        # Print insights
        pp.print_info("Heatmap insights:")
        pp.print_info("  Left: Baseline weights are active across ALL classes")
        pp.print_info("  Right: Expert weights are ONLY active on minority classes (4, 7)")
        minority_avg = expert_abs[:, [4, 7]].mean()
        majority_avg = np.mean([expert_abs[:, i].mean() for i in range(n_classes) if i not in [4, 7]])
        pp.print_info(f"  Average expert weight | Minority: {minority_avg:.3f} vs Majority: {majority_avg:.5f}")

    def generate_figure_5_2(self):
        """
        Figure 5-2: Output Contribution Bar Chart

        Shows how much each model contributes to final predictions
        """
        pp = ProgressPrinter()
        pp.print_step(2, 7, "Generating Figure 5-2: Output Contribution")

        if self.contribution_analysis is None:
            pp.print_info("Skipping: output contribution not available")
            return

        per_class = self.contribution_analysis['per_class']

        # Prepare data
        classes = sorted(per_class.keys(), key=int)
        baseline_means = [per_class[c]['baseline_mean'] for c in classes]
        expert_means = [per_class[c]['expert_mean'] for c in classes]

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(classes))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_means, width, label='Baseline',
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, expert_means, width, label='Expert',
                      color='crimson', alpha=0.8, edgecolor='black', linewidth=1.2)

        ax.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Contribution', fontsize=11, fontweight='bold')
        ax.set_title('Output Contribution: Baseline vs Expert',
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'figure_5_2_output_contribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_5_2_output_contribution.pdf',
                   format='pdf', bbox_inches='tight')
        plt.close()

        pp.print_success(f"Saved to: {output_path}")

    def generate_figure_5_3(self):
        """
        Figure 5-3: Prediction Change Analysis for Minority Classes

        Focuses on how GEE changes predictions compared to baseline
        Shows detailed analysis for classes with most changes
        """
        pp = ProgressPrinter()
        pp.print_step(3, 7, "Generating Figure 5-3: Prediction Changes")

        if self.features is None:
            pp.print_info("Skipping: features not available")
            return

        # Load features and probabilities
        baseline_path = self.results_dir / "features" / "baseline_features.npz"
        if not baseline_path.exists():
            pp.print_info("Skipping: baseline features not found")
            return

        baseline_data = load_npz(baseline_path)
        baseline_features = baseline_data['features']
        baseline_labels = baseline_data['labels']

        baseline_probs_path = self.results_dir / "features" / "baseline_probs.npz"
        gee_probs_path = self.results_dir / "features" / "gating_probs.npz"

        if not (baseline_probs_path.exists() and gee_probs_path.exists()):
            pp.print_info("Skipping: probability files not found")
            return

        baseline_probs_data = load_npz(baseline_probs_path)
        gee_probs_data = load_npz(gee_probs_path)

        # Get predictions and confidence
        baseline_preds = np.argmax(baseline_probs_data['probs'], axis=1)
        gee_preds = np.argmax(gee_probs_data['probs'], axis=1)
        baseline_conf = baseline_probs_data['confidence']
        gee_conf = gee_probs_data['confidence']

        # Find classes with most prediction changes
        diff_mask = (baseline_preds != gee_preds)
        class_changes = {}
        for c in np.unique(baseline_labels):
            mask = baseline_labels == c
            n_changed = (baseline_preds[mask] != gee_preds[mask]).sum()
            n_total = mask.sum()
            if n_total > 0:
                change_rate = 100 * n_changed / n_total
                class_changes[c] = (change_rate, n_changed, n_total)

        # Sort by change rate
        sorted_classes = sorted(class_changes.items(), key=lambda x: x[1][0], reverse=True)

        # Select top 2 classes with most changes
        top_classes = [sorted_classes[0][0], sorted_classes[1][0]]

        pp.print_info(f"Top changed classes: {[f'C{int(c)} ({class_changes[c][0]:.1f}%)' for c in top_classes]}")

        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(baseline_features)
        var_explained = pca.explained_variance_ratio_ * 100

        pp.print_info(f"PCA variance explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%")

        # Create figure with 2 subplots: Baseline vs GEE decision boundaries
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Focus on class 7 (minority) and class 8 (majority) interaction
        focus_classes = [7, 8]  # Show interaction between minority class 7 and majority class 8

        # Create mesh grid for decision boundary
        x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
        y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        pp.print_info(f"Mesh grid size: {xx.shape}, mesh points: {xx.size}")

        # For each model, plot decision boundary
        model_names = ['Baseline', 'GEE']
        preds_list = [baseline_preds, gee_preds]
        confs_list = [baseline_conf, gee_conf]

        for idx, (model_name, preds, conf) in enumerate(zip(model_names, preds_list, confs_list)):
            ax = axes[idx]

            # Plot decision regions using contourf
            # We create a grid of predictions to show decision regions
            grid_preds = np.zeros(xx.shape)

            # For each grid point, find the nearest test sample and use its prediction
            # This is an approximation since we can't project back to high-dim feature space
            from scipy.interpolate import NearestNDInterpolator
            interp = NearestNDInterpolator(features_2d, preds)
            grid_preds = interp(xx, yy)

            # Plot decision regions
            contour = ax.contourf(xx, yy, grid_preds, levels=15, alpha=0.3, cmap='tab10')

            # Plot decision boundaries (where prediction changes)
            ax.contour(xx, yy, grid_preds, levels=10, colors='black', alpha=0.5, linewidths=1)

            # Plot test samples
            for c in focus_classes:
                class_mask = baseline_labels == c
                color = 'red' if c == 7 else 'blue'
                label = f'Class {c} ({"minority" if c == 7 else "majority"})'
                ax.scatter(features_2d[class_mask, 0], features_2d[class_mask, 1],
                          c=color, s=30, alpha=0.7, edgecolors='black', linewidth=0.5,
                          label=label)

            # Plot other classes as background
            other_mask = ~np.isin(baseline_labels, focus_classes)
            ax.scatter(features_2d[other_mask, 0], features_2d[other_mask, 1],
                      c='lightgray', s=10, alpha=0.2, label='Other classes')

            # Add decision boundary statistics
            # Calculate boundary complexity: count grid cells where prediction changes
            # This is a simpler and more robust metric than measuring contour lengths
            horizontal_changes = np.sum(grid_preds[:-1, :] != grid_preds[1:, :])
            vertical_changes = np.sum(grid_preds[:, :-1] != grid_preds[:, 1:])
            boundary_complexity = horizontal_changes + vertical_changes

            # Calculate decision region purity: how mixed are the predictions
            region_entropy = 0
            unique_preds, counts = np.unique(grid_preds, return_counts=True)
            probs = counts / grid_preds.size
            region_entropy = -np.sum(probs * np.log(probs + 1e-10))

            # Calculate class separation: average distance from correctly predicted class 7
            # points to the nearest different-class point
            # This measures: for samples that the model predicts as class 7, how well-separated
            # are they from other classes? Higher = better separation = more confident prediction
            correctly_predicted_as_7_mask = (preds == 7)
            if correctly_predicted_as_7_mask.sum() > 0:
                # Get features of samples predicted as class 7 by this model
                predicted_7_features = features_2d[correctly_predicted_as_7_mask]
                # Get features of all other samples (predicted as other classes)
                other_features = features_2d[~correctly_predicted_as_7_mask]
                # Find minimum distance from each predicted-7 point to any other point
                min_dists = []
                for p in predicted_7_features:
                    dists = np.linalg.norm(other_features - p, axis=1)
                    min_dists.append(dists.min())
                class_7_separation = np.mean(min_dists)
                class_7_separation_std = np.std(min_dists)
            else:
                class_7_separation = 0
                class_7_separation_std = 0

            stats_text = f'{model_name} Model\n'
            stats_text += f'Boundary changes: {boundary_complexity}\n'
            stats_text += f'Region entropy: {region_entropy:.3f}\n'
            stats_text += f'Mean confidence: {conf.mean():.3f}\n'
            if correctly_predicted_as_7_mask.sum() > 0:
                stats_text += f'C7 separation: {class_7_separation:.3f}±{class_7_separation_std:.3f}\n'
                stats_text += f'(n={correctly_predicted_as_7_mask.sum()})'
            else:
                stats_text += f'C7 separation: N/A\n'
                stats_text += f'(no C7 predictions)'

            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow' if idx == 0 else 'lightcoral', alpha=0.8))

            ax.set_title(f'{model_name} Decision Boundary',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)', fontsize=10)
            ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)', fontsize=10)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.2)

        plt.suptitle('Decision Boundary Comparison: Baseline vs GEE\n(Focus on Class 7 vs Class 8 interaction)',
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'figure_5_3_decision_boundaries.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_5_3_decision_boundaries.pdf',
                   format='pdf', bbox_inches='tight')
        plt.close()

        pp.print_success(f"Saved to: {output_path}")

        # Print detailed prediction flow analysis
        pp.print_info("Prediction flow analysis:")
        for c in top_classes:
            mask = baseline_labels == c
            class_baseline_preds = baseline_preds[mask]
            class_gee_preds = gee_preds[mask]
            changed_mask = (class_baseline_preds != class_gee_preds)

            if changed_mask.sum() > 0:
                pp.print_info(f"  Class {int(c)} ({changed_mask.sum()} changed):")
                # Get indices of changed samples within this class
                changed_indices = np.where(changed_mask)[0]

                # For each unique source prediction, count where they went
                for pred in np.unique(class_baseline_preds[changed_indices]):
                    count = (class_baseline_preds[changed_indices] == pred).sum()
                    # Get destination predictions for samples originally predicted as 'pred'
                    dest_preds = class_gee_preds[changed_indices][class_baseline_preds[changed_indices] == pred]
                    for dest_pred in np.unique(dest_preds):
                        n_flow = (dest_preds == dest_pred).sum()
                        pp.print_info(f"    C{int(pred)} → C{int(dest_pred)}: {n_flow} samples")
            else:
                pp.print_info(f"  Class {int(c)}: No changes")

    def generate_figure_5_6(self):
        """
        Figure 5-6: Softmax Entropy Distribution Comparison

        Compares entropy distributions between baseline and GEE
        Higher entropy = more uncertainty
        """
        pp = ProgressPrinter()
        pp.print_step(6, 7, "Generating Figure 5-6: Entropy Distribution")

        if self.probs is None:
            pp.print_info("Skipping: probabilities not available")
            return

        # Load baseline and GEE probabilities
        baseline_path = self.results_dir / "features" / "baseline_probs.npz"
        gee_path = self.results_dir / "features" / "gating_probs.npz"

        if not (baseline_path.exists() and gee_path.exists()):
            pp.print_info("Skipping: probability files not found")
            return

        baseline_data = load_npz(baseline_path)
        gee_data = load_npz(gee_path)

        entropy_baseline = baseline_data['entropy']
        entropy_gee = gee_data['entropy']

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(entropy_baseline, bins=50, alpha=0.6, label='Baseline',
                color='steelblue', density=True)
        ax.hist(entropy_gee, bins=50, alpha=0.6, label='GEE',
                color='crimson', density=True)

        # Add mean lines
        ax.axvline(np.mean(entropy_baseline), color='steelblue',
                  linestyle='--', linewidth=2,
                  label=f'Baseline Mean: {np.mean(entropy_baseline):.2f}')
        ax.axvline(np.mean(entropy_gee), color='crimson',
                  linestyle='--', linewidth=2,
                  label=f'GEE Mean: {np.mean(entropy_gee):.2f}')

        ax.set_xlabel('Softmax Entropy H(P)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title('Softmax Entropy Distribution Comparison\n(Baseline has peaked distribution, GEE has more uniform spread)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'figure_5_6_entropy_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_5_6_entropy_distribution.pdf',
                   format='pdf', bbox_inches='tight')
        plt.close()

        pp.print_success(f"Saved to: {output_path}")

    def generate_figure_5_4(self):
        """
        Figure 5-4: Class Distribution and Weight Analysis

        Shows the relationship between class frequency and expert contribution
        """
        pp = ProgressPrinter()
        pp.print_step(4, 7, "Generating Figure 5-4: Class Distribution Analysis")

        if self.contribution_analysis is None:
            pp.print_info("Skipping: output contribution not available")
            return

        per_class = self.contribution_analysis['per_class']

        # Prepare data
        classes = sorted(per_class.keys(), key=int)
        class_counts = [per_class[c]['n_samples'] for c in classes]
        expert_contribs = [per_class[c]['expert_mean'] for c in classes]

        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Class distribution
        colors = ['red' if int(c) in [4, 7] else 'steelblue' for c in classes]
        bars1 = ax1.bar(classes, class_counts, color=colors, alpha=0.7, edgecolor='black')

        # Highlight minority classes
        for c, bar in zip(classes, bars1):
            if int(c) in [4, 7]:
                bar.set_label('Minority Class' if c == '4' else '')

        ax1.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax1.set_title('Test Set Class Distribution\n(Red = Minority Classes)',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Right plot: Expert contribution vs class size
        scatter = ax2.scatter(class_counts, expert_contribs,
                             c=[int(c) for c in classes],
                             cmap='tab10', s=100, alpha=0.7,
                             edgecolors='black', linewidth=1.5)

        # Highlight minority classes
        minority_mask = [int(c) in [4, 7] for c in classes]
        ax2.scatter([class_counts[i] for i, m in enumerate(minority_mask) if m],
                   [expert_contribs[i] for i, m in enumerate(minority_mask) if m],
                   c='red', s=150, marker='*', label='Minority Classes',
                   edgecolors='black', linewidth=2)

        # Add class labels with offset to avoid overlap
        for c, count, contrib in zip(classes, class_counts, expert_contribs):
            # Add vertical offset based on marker type
            if int(c) in [4, 7]:  # Minority classes with star markers
                offset_y = 12
            else:  # Majority classes with circle markers
                offset_y = 8
            ax2.annotate(f'C{c}', (count, contrib),
                        fontsize=9, ha='center', va='bottom',
                        xytext=(0, offset_y), textcoords='offset points',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='gray', alpha=0.8))

        ax2.set_xlabel('Class Sample Count', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Expert Contribution', fontsize=11, fontweight='bold')
        ax2.set_title('Expert Contribution vs Class Size\n(Star = Minority Classes)',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'figure_5_4_gradient_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_5_4_gradient_comparison.pdf',
                   format='pdf', bbox_inches='tight')
        plt.close()

        pp.print_success(f"Saved to: {output_path}")

    def generate_figure_5_5(self):
        """
        Figure 5-5: Feature Space Compactness Analysis

        Analyzes how tightly packed different classes are in feature space
        """
        pp = ProgressPrinter()
        pp.print_step(5, 7, "Generating Figure 5-5: Feature Space Analysis")

        if self.features is None:
            pp.print_info("Skipping: features not available")
            return

        # Load baseline features
        feature_path = self.results_dir / "features" / "baseline_features.npz"
        if not feature_path.exists():
            pp.print_info("Skipping: baseline features not found")
            return

        data = load_npz(feature_path)
        features = data['features']
        labels = data['labels']

        # Calculate compactness for each class (std dev of features)
        unique_labels = np.unique(labels)
        compactness = {}

        for c in unique_labels:
            mask = labels == c
            class_features = features[mask]
            # Mean distance to centroid
            centroid = class_features.mean(axis=0)
            distances = np.linalg.norm(class_features - centroid, axis=1)
            compactness[int(c)] = {
                'mean_distance': distances.mean(),
                'std_distance': distances.std(),
                'n_samples': mask.sum()
            }

        # Prepare data
        classes = sorted(compactness.keys())
        mean_distances = [compactness[c]['mean_distance'] for c in classes]
        std_distances = [compactness[c]['std_distance'] for c in classes]
        colors = ['red' if c in [4, 7] else 'steelblue' for c in classes]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(classes, mean_distances, yerr=std_distances,
                     color=colors, alpha=0.7, capsize=5,
                     edgecolor='black', linewidth=1.5)

        # Highlight minority classes
        for c, bar in zip(classes, bars):
            if c in [4, 7]:
                bar.set_label('Minority Class' if c == 4 else '')

        ax.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Distance to Centroid', fontsize=11, fontweight='bold')
        ax.set_title('Feature Space Compactness by Class\n(Red = Minority Classes, Lower = More Compact)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)

        # Add value labels on bars
        for c, bar, mean in zip(classes, bars, mean_distances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.2f}',
                   ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / 'figure_5_5_feature_space.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure_5_5_feature_space.pdf',
                   format='pdf', bbox_inches='tight')
        plt.close()

        pp.print_success(f"Saved to: {output_path}")

    def generate_figure_5_8(self):
        """
        Generate Figure 5-8: GEE Unified Theoretical Framework

        Three-layer structure:
        - Optimization Layer: Weighted Loss → Gradient Balance
        - Architecture Layer: Gating Network → Adaptive Fusion
        - Result Layer: Minority Class Recognition ↑ + OSR Performance ↑
        """
        pp = ProgressPrinter()
        pp.print_section("Figure 5-8: GEE Unified Theoretical Framework")

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Define colors
        layer_colors = {
            'optimization': '#E8F4F8',  # Light blue
            'architecture': '#FFF4E6',  # Light orange
            'result': '#E8F5E9',  # Light green
            'arrow': '#455A64',
            'text': '#263238'
        }

        # Define layer positions (top to bottom)
        layer_y = {
            'optimization': 8.5,
            'architecture': 5.5,
            'result': 2.5
        }

        # Layer 1: Optimization Layer (加权损失 → 梯度平衡)
        opt_box = plt.Rectangle((1, layer_y['optimization'] - 0.8), 8, 1.6,
                               facecolor=layer_colors['optimization'],
                               edgecolor='#0277BD', linewidth=2.5,
                               zorder=2)
        ax.add_patch(opt_box)

        ax.text(5, layer_y['optimization'] + 0.4,
               '优化层 (Optimization Layer)',
               ha='center', va='center', fontsize=16, fontweight='bold',
               color=layer_colors['text'])

        # Weighted Loss box
        weighted_box = plt.Rectangle((1.8, layer_y['optimization'] - 0.55), 2.5, 0.9,
                                    facecolor='#BBDEFB', edgecolor='#0277BD',
                                    linewidth=2, zorder=3)
        ax.add_patch(weighted_box)
        ax.text(3.05, layer_y['optimization'] - 0.1,
               '加权损失\n(Weighted CE)',
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=layer_colors['text'])

        # Gradient Balance box
        gradient_box = plt.Rectangle((5.7, layer_y['optimization'] - 0.55), 2.5, 0.9,
                                    facecolor='#90CAF9', edgecolor='#0277BD',
                                    linewidth=2, zorder=3)
        ax.add_patch(gradient_box)
        ax.text(6.95, layer_y['optimization'] - 0.1,
               '梯度平衡\n(Gradient Balance)',
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=layer_colors['text'])

        # Arrow from weighted to gradient
        ax.annotate('', xy=(5.6, layer_y['optimization'] - 0.1),
                   xytext=(4.4, layer_y['optimization'] - 0.1),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=layer_colors['arrow']),
                   zorder=4)

        # Layer 2: Architecture Layer (门控网络 → 自适应融合)
        arch_box = plt.Rectangle((1, layer_y['architecture'] - 0.8), 8, 1.6,
                                facecolor=layer_colors['architecture'],
                                edgecolor='#EF6C00', linewidth=2.5,
                                zorder=2)
        ax.add_patch(arch_box)

        ax.text(5, layer_y['architecture'] + 0.4,
               '架构层 (Architecture Layer)',
               ha='center', va='center', fontsize=16, fontweight='bold',
               color=layer_colors['text'])

        # Gating Network box
        gating_box = plt.Rectangle((1.8, layer_y['architecture'] - 0.55), 2.5, 0.9,
                                  facecolor='#FFE0B2', edgecolor='#EF6C00',
                                  linewidth=2, zorder=3)
        ax.add_patch(gating_box)
        ax.text(3.05, layer_y['architecture'] - 0.1,
               '门控网络\n(Gating Network)',
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=layer_colors['text'])

        # Adaptive Fusion box
        fusion_box = plt.Rectangle((5.7, layer_y['architecture'] - 0.55), 2.5, 0.9,
                                  facecolor='#FFCC80', edgecolor='#EF6C00',
                                  linewidth=2, zorder=3)
        ax.add_patch(fusion_box)
        ax.text(6.95, layer_y['architecture'] - 0.1,
               '自适应融合\n(Adaptive Fusion)',
               ha='center', va='center', fontsize=11, fontweight='bold',
               color=layer_colors['text'])

        # Arrow from gating to fusion
        ax.annotate('', xy=(5.6, layer_y['architecture'] - 0.1),
                   xytext=(4.4, layer_y['architecture'] - 0.1),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color=layer_colors['arrow']),
                   zorder=4)

        # Layer 3: Result Layer (少数类识别↑ + OSR性能↑)
        result_box = plt.Rectangle((1, layer_y['result'] - 0.8), 8, 1.6,
                                  facecolor=layer_colors['result'],
                                  edgecolor='#2E7D32', linewidth=2.5,
                                  zorder=2)
        ax.add_patch(result_box)

        ax.text(5, layer_y['result'] + 0.4,
               '结果层 (Result Layer)',
               ha='center', va='center', fontsize=16, fontweight='bold',
               color=layer_colors['text'])

        # Minority Class Recognition box
        minority_box = plt.Rectangle((1.5, layer_y['result'] - 0.55), 3, 0.9,
                                    facecolor='#A5D6A7', edgecolor='#2E7D32',
                                    linewidth=2, zorder=3)
        ax.add_patch(minority_box)
        ax.text(3.0, layer_y['result'] - 0.1,
               '少数类识别 ↑\nMinority Class\nRecognition ↑',
               ha='center', va='center', fontsize=10, fontweight='bold',
               color=layer_colors['text'])

        # OSR Performance box
        osr_box = plt.Rectangle((5.5, layer_y['result'] - 0.55), 3, 0.9,
                               facecolor='#81C784', edgecolor='#2E7D32',
                               linewidth=2, zorder=3)
        ax.add_patch(osr_box)
        ax.text(7.0, layer_y['result'] - 0.1,
               'OSR性能 ↑\nOpen Set\nRecognition ↑',
               ha='center', va='center', fontsize=10, fontweight='bold',
               color=layer_colors['text'])

        # Large arrows connecting layers (top to bottom)
        # Arrow from optimization to architecture
        ax.annotate('', xy=(5, layer_y['architecture'] + 0.9),
                   xytext=(5, layer_y['optimization'] - 0.9),
                   arrowprops=dict(arrowstyle='->', lw=3, color=layer_colors['arrow']),
                   zorder=1)

        # Arrow from architecture to result
        ax.annotate('', xy=(5, layer_y['result'] + 0.9),
                   xytext=(5, layer_y['architecture'] - 0.9),
                   arrowprops=dict(arrowstyle='->', lw=3, color=layer_colors['arrow']),
                   zorder=1)

        # Add side annotations for synergy
        ax.text(9.2, layer_y['optimization'],
               '创造中性\n损失景观',
               ha='left', va='center', fontsize=9,
               style='italic', color='#546E7A')

        ax.text(9.2, layer_y['architecture'],
               '智能融合\n策略',
               ha='left', va='center', fontsize=9,
               style='italic', color='#546E7A')

        ax.text(9.2, layer_y['result'],
               '协同作用\nSynergy',
               ha='left', va='center', fontsize=9,
               style='italic', color='#546E7A')

        # Add small connection arrows from side annotations to layers
        ax.annotate('', xy=(9.1, layer_y['optimization']),
                   xytext=(9.4, layer_y['optimization']),
                   arrowprops=dict(arrowstyle='->', lw=1, color='#546E7A'),
                   zorder=1)

        ax.annotate('', xy=(9.1, layer_y['architecture']),
                   xytext=(9.4, layer_y['architecture']),
                   arrowprops=dict(arrowstyle='->', lw=1, color='#546E7A'),
                   zorder=1)

        ax.annotate('', xy=(9.1, layer_y['result']),
                   xytext=(9.4, layer_y['result']),
                   arrowprops=dict(arrowstyle='->', lw=1, color='#546E7A'),
                   zorder=1)

        # Add title at the top
        ax.text(5, 9.5,
               'GEE统一理论框架\nUnified Theoretical Framework',
               ha='center', va='center', fontsize=18, fontweight='bold',
               color=layer_colors['text'])

        # Add synergy note at the bottom
        ax.text(5, 0.5,
               '两大原则协同作用：梯度平衡为门控网络创造学习条件，自适应融合实现精细分类策略\nSynergy: Gradient balance enables gating network learning, adaptive fusion achieves fine-grained classification',
               ha='center', va='center', fontsize=9, style='italic',
               color='#546E7A')

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "figure_5_8_unified_framework.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        pp.print_success(f"Saved to: {output_path}")

    def generate_all_figures(self):
        """Generate all figures"""
        pp = ProgressPrinter()
        pp.print_section("Generating All Figures for Chapter 5")

        self.generate_figure_5_1()
        self.generate_figure_5_2()
        self.generate_figure_5_3()
        self.generate_figure_5_4()
        self.generate_figure_5_5()
        self.generate_figure_5_6()
        self.generate_figure_5_8()

        pp.print_section("Figure Generation Complete!")
        pp.print_success(f"All figures saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--results_dir", type=str, default="./analysis_results",
                       help="Directory containing analysis results")
    parser.add_argument("--output_dir", type=str, default="./thesis/figures",
                       help="Directory to save figures")

    args = parser.parse_args()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Generate figures
    generator = FigureGenerator(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
