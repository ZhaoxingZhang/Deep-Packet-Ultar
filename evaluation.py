import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import os
import click
import torch
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

# Make sure the ml module is in the python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import ResNet, GatingNetwork

def load_data(data_path):
    table = pq.read_table(data_path)
    df = table.to_pandas()
    
    features = torch.from_numpy(np.array(df['feature'].tolist(), dtype=np.float32))
    labels = torch.from_numpy(np.array(df['label'].tolist(), dtype=np.int64))
    
    # The model expects a channel dimension, so we add it.
    if len(features.shape) == 2:
        features = features.unsqueeze(1)
        
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    return dataloader

def get_output_dir_from_model_path(model_path):
    """从模型路径提取输出目录"""
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join("evaluation_results", model_name)
    return output_dir

@click.command()
@click.option("-d", "--data_path", required=True, help="Path to the test data parquet file.")
@click.option("-o", "--output_dir", required=True, help="Directory to save evaluation results.")
@click.option("--eval-mode", type=click.Choice(['standard', 'ensemble', 'gating_ensemble']), default='standard', help="Evaluation mode.")
# Options for 'standard' mode
@click.option("--model_path", help="Path to the single model checkpoint for standard evaluation.")
@click.option("--model_type", type=click.Choice(['resnet']), help="Type of model for standard evaluation.")
# Options for 'ensemble' mode
@click.option("--baseline_model_path", help="Path to the baseline model for ensemble evaluation.")
@click.option("--minority_model_path", help="Path to the minority expert model for ensemble evaluation.")
@click.option("--minority_classes", type=int, multiple=True, help="The original labels of the minority classes (e.g., 5 7).")
@click.option("--baseline_weight", type=float, default=0.5, help="Weight for the baseline model.")
@click.option("--minority_weight", type=float, default=0.5, help="Weight for the minority expert model.")
# Options for 'gating_ensemble' mode
@click.option("--gating_network_path", help="Path to the pre-trained Gating Network.")
def evaluate(data_path, output_dir, eval_mode, model_path, model_type, 
             baseline_model_path, minority_model_path, minority_classes, 
             baseline_weight, minority_weight,
             gating_network_path):
    """Evaluates a trained model or an ensemble of models on the given test set."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_preds = []
    all_labels = []

    print(f"Loading data from {data_path}...")
    test_dataloader = load_data(data_path)

    if eval_mode == 'standard':
        if not model_path or not model_type:
            raise ValueError("For 'standard' mode, --model_path and --model_type are required.")
        
        print(f"--- Running in STANDARD mode ---")
        print(f"Loading {model_type} model from {model_path}...")
        if model_type == 'resnet':
            model = ResNet.load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unknown model type for standard evaluation: {model_type}")
        
        model.eval()
        with torch.no_grad():
            for features, labels in test_dataloader:
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    elif eval_mode == 'ensemble':
        if not baseline_model_path or not minority_model_path or not minority_classes:
            raise ValueError("For 'ensemble' mode, --baseline_model_path, --minority_model_path, and --minority_classes are required.")
        
        print(f"--- Running in ENSEMBLE mode ---")
        print(f"Loading baseline model from {baseline_model_path}...")
        baseline_model = ResNet.load_from_checkpoint(baseline_model_path)
        baseline_model.eval()

        print(f"Loading minority expert model from {minority_model_path}...")
        minority_model = ResNet.load_from_checkpoint(minority_model_path)
        minority_model.eval()

        # Create the mapping from the expert's output index (0, 1, ...) to the original class label
        expert_idx_to_original_label = {i: label for i, label in enumerate(minority_classes)}
        print(f"Minority expert mapping: {expert_idx_to_original_label}\n")

        with torch.no_grad():
            for features, labels in test_dataloader:
                # 1. Get probabilities from baseline model
                base_outputs = baseline_model(features)
                base_probs = torch.nn.functional.softmax(base_outputs, dim=1)

                # 2. Get probabilities from minority expert
                expert_outputs = minority_model(features)
                expert_probs_small = torch.nn.functional.softmax(expert_outputs, dim=1)

                # 3. Align expert probabilities to the full class space
                num_total_classes = base_probs.shape[1]
                expert_probs_full = torch.zeros_like(base_probs)
                for expert_local_idx, original_label_idx in expert_idx_to_original_label.items():
                    # Ensure expert_local_idx is within the bounds of expert_probs_small
                    if expert_local_idx < expert_probs_small.shape[1]:
                        expert_probs_full[:, original_label_idx] = expert_probs_small[:, expert_local_idx]
                    else:
                        # This case should ideally not happen if the model was trained correctly,
                        # but we handle it for robustness as per user's request.
                        print(f"Warning: Minority expert output dimension ({expert_probs_small.shape[1]}) is smaller than expected for local index {expert_local_idx}. Skipping.")

                # 4. Combine probabilities with weights
                final_probs = (baseline_weight * base_probs) + (minority_weight * expert_probs_full)
                
                # 5. Get final prediction
                _, predicted = torch.max(final_probs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    elif eval_mode == 'gating_ensemble':
        if not baseline_model_path or not minority_model_path or not minority_classes or not gating_network_path:
            raise ValueError("For 'gating_ensemble' mode, --baseline_model_path, --minority_model_path, --minority_classes, and --gating_network_path are required.")
        
        print(f"--- Running in GATING ENSEMBLE mode ---\n")

        # --- Load Models ---
        print(f"Loading baseline model from {baseline_model_path}...")
        baseline_model = ResNet.load_from_checkpoint(baseline_model_path)
        baseline_model.eval()

        print(f"Loading minority expert model from {minority_model_path}...")
        minority_model = ResNet.load_from_checkpoint(minority_model_path)
        minority_model.eval()

        num_total_classes = baseline_model.out.out_features
        print(f"Detected {num_total_classes} total classes.")

        print(f"Loading pre-trained Gating Network from {gating_network_path}...")
        gating_network = GatingNetwork(num_total_classes)
        checkpoint = torch.load(gating_network_path)
        gating_network.load_state_dict(checkpoint['model_state_dict'])
        gating_network.eval()

        # --- Create Mappings and Evaluate ---
        expert_idx_to_original_label = {i: label for i, label in enumerate(minority_classes)}
        print(f"Minority expert mapping: {expert_idx_to_original_label}\n")

        with torch.no_grad():
            for features, labels in test_dataloader:
                # 1. Get probabilities from baseline model
                base_outputs = baseline_model(features)
                base_probs = torch.nn.functional.softmax(base_outputs, dim=1)

                # 2. Get probabilities from minority expert
                expert_outputs = minority_model(features)
                expert_probs_small = torch.nn.functional.softmax(expert_outputs, dim=1)

                # 3. Align expert probabilities to the full class space
                expert_probs_full = torch.zeros_like(base_probs)
                for expert_local_idx, original_label_idx in expert_idx_to_original_label.items():
                    if expert_local_idx < expert_probs_small.shape[1]:
                        expert_probs_full[:, original_label_idx] = expert_probs_small[:, expert_local_idx]
                
                # 4. Combine probabilities using the Gating Network
                final_probs = gating_network(base_probs, expert_probs_full)
                
                # 5. Get final prediction
                _, predicted = torch.max(final_probs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


    # --- Common evaluation logic ---
    print("\nCalculating metrics...")
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)
    
    results_file = os.path.join(output_dir, "evaluation_summary.txt")
    with open(results_file, 'w') as f:
        f.write(f"--- Evaluation Mode: {eval_mode} ---")
        if eval_mode == 'ensemble':
            f.write(f"Baseline Model: {baseline_model_path}\n")
            f.write(f"Minority Model: {minority_model_path}\n")
            f.write(f"Weights (Base/Minority): {baseline_weight}/{minority_weight}\n\n")
        elif eval_mode == 'gating_ensemble':
            f.write(f"Baseline Model: {baseline_model_path}\n")
            f.write(f"Minority Model: {minority_model_path}\n")
            f.write(f"Gating Network Path: {gating_network_path}\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    print(f"\nResults saved to {results_file}")

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({eval_mode.capitalize()} Mode)')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    evaluate()