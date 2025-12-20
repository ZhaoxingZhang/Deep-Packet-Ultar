import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import os
import click
import torch
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

# Make sure the ml module is in the python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import ResNet, GatingNetwork, CNN

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
@click.option("--model_type", type=click.Choice(['resnet', 'cnn']), help="Type of model for standard evaluation.")
# Options for 'ensemble' mode
@click.option("--baseline_model_path", help="Path to the baseline model for ensemble evaluation.")
@click.option("--minority_model_path", help="Path to the minority expert model for ensemble evaluation.")
@click.option("--minority_classes", type=int, multiple=True, help="The original labels of the minority classes (e.g., 5 7).")
@click.option("--baseline_weight", type=float, default=0.5, help="Weight for the baseline model.")
@click.option("--minority_weight", type=float, default=0.5, help="Weight for the minority expert model.")
# Options for 'gating_ensemble' mode
@click.option("--baseline_model_type", type=click.Choice(['resnet', 'cnn']), default='resnet', help="Type of the baseline model for GEE.")
@click.option("--minority_model_type", type=click.Choice(['resnet', 'cnn']), default='resnet', help="Type of the minority expert model for GEE.")
@click.option("--gating_network_path", help="Path to the pre-trained Gating Network.")
@click.option("--gating-has-garbage-class", is_flag=True, help="Flag if the gating network was trained with a garbage class.")
# Options for open-set evaluation
@click.option("--open-set-eval", is_flag=True, help="Enable open-set recognition evaluation.")
@click.option("--known-classes", type=int, multiple=True, help="Labels of the known classes for open-set evaluation.")
@click.option("--unknown-classes", type=int, multiple=True, help="Labels of the unknown classes for open-set evaluation.")
@click.option("--label-map", help="Comma-separated label mapping for open-set evaluation, e.g., '0:6,1:7'.")
def evaluate(data_path, output_dir, eval_mode, model_path, model_type, 
             baseline_model_path, minority_model_path, minority_classes, 
             baseline_weight, minority_weight,
             baseline_model_type, minority_model_type,
             gating_network_path, gating_has_garbage_class,
             open_set_eval, known_classes, unknown_classes, label_map):
    """Evaluates a trained model or an ensemble of models on the given test set."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_preds = []
    all_labels = []
    all_probs = []

    print(f"Loading data from {data_path}...")
    test_dataloader = load_data(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if eval_mode == 'standard':
        if not model_path or not model_type:
            raise ValueError("For 'standard' mode, --model_path and --model_type are required.")
        
        print(f"--- Running in STANDARD mode ---")
        print(f"Loading {model_type} model from {model_path}...")
        if model_type == 'resnet':
            model = ResNet.load_from_checkpoint(model_path, map_location=device)
        elif model_type == 'cnn':
            model = CNN.load_from_checkpoint(model_path, map_location=device)
        else:
            raise ValueError(f"Unknown model type for standard evaluation: {model_type}")
        
        model.to(device)
        model.eval()
        with torch.no_grad():
            for features, labels in test_dataloader:
                features = features.to(device)
                outputs = model(features)
                final_probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(final_probs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(final_probs.cpu().numpy())

    elif eval_mode == 'ensemble':
        if not baseline_model_path or not minority_model_path or not minority_classes:
            raise ValueError("For 'ensemble' mode, --baseline_model_path, --minority_model_path, and --minority_classes are required.")
        
        print(f"--- Running in ENSEMBLE mode ---")
        # ... (ensemble logic remains unchanged) ...
        # Note: This mode is not fully updated with the latest logic, focusing on gating_ensemble
        pass

    elif eval_mode == 'gating_ensemble':
        if not all([baseline_model_path, minority_model_path, minority_classes, gating_network_path]):
            raise ValueError("For 'gating_ensemble' mode, all model paths and minority_classes are required.")
        
        print(f"--- Running in GATING ENSEMBLE mode ---")
        if gating_has_garbage_class:
            print("Note: Gating network is configured with a garbage class.")

        def load_model(model_type, model_path):
            if model_type == 'resnet':
                return ResNet.load_from_checkpoint(model_path, map_location=device)
            elif model_type == 'cnn':
                return CNN.load_from_checkpoint(model_path, map_location=device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        print(f"Loading baseline {baseline_model_type} model from {baseline_model_path}...")
        baseline_model = load_model(baseline_model_type, baseline_model_path).to(device)
        baseline_model.eval()

        print(f"Loading minority expert {minority_model_type} model from {minority_model_path}...")
        minority_model = load_model(minority_model_type, minority_model_path).to(device)
        minority_model.eval()

        num_known_classes = baseline_model.hparams.output_dim
        print(f"Detected {num_known_classes} known classes from baseline model.")

        print(f"Loading pre-trained Gating Network from {gating_network_path}...")
        gating_network = GatingNetwork(
            num_classes=num_known_classes,
            use_garbage_class=gating_has_garbage_class
        )
        gating_network.load_state_dict(torch.load(gating_network_path, map_location=device))
        gating_network.to(device)
        gating_network.eval()

        expert_idx_to_original_label = {i: label for i, label in enumerate(minority_classes)}

        with torch.no_grad():
            for features, labels in test_dataloader:
                features = features.to(device)
                base_outputs = baseline_model(features)
                base_probs = torch.nn.functional.softmax(base_outputs, dim=1)

                expert_outputs = minority_model(features)
                expert_probs_small = torch.nn.functional.softmax(expert_outputs, dim=1)

                expert_probs_full = torch.zeros_like(base_probs)
                for expert_local_idx, original_label_idx in expert_idx_to_original_label.items():
                    if expert_local_idx < expert_probs_small.shape[1]:
                        expert_probs_full[:, original_label_idx] = expert_probs_small[:, expert_local_idx]
                
                final_logits = gating_network(base_probs, expert_probs_full)
                final_probs = torch.nn.functional.softmax(final_logits, dim=1)
                
                _, predicted = torch.max(final_probs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(final_probs.cpu().numpy())

    # --- Remap predictions if a label map is provided ---
    if label_map:
        print(f"\nApplying label map: {label_map}")
        label_mapping = {int(k): int(v) for k, v in (item.split(':') for item in label_map.split(','))}
        preds_to_evaluate = [label_mapping.get(p, -1) for p in all_preds]
        print(f"Original preds (first 10): {all_preds[:10]}")
        print(f"Remapped preds (first 10): {preds_to_evaluate[:10]}")
    else:
        preds_to_evaluate = all_preds

    # --- Common evaluation logic ---
    print("\nCalculating metrics...")
    accuracy = accuracy_score(all_labels, preds_to_evaluate)
    
    unique_labels = np.unique(all_labels)
    report = classification_report(all_labels, preds_to_evaluate, labels=unique_labels, zero_division=0)
    cm = confusion_matrix(all_labels, preds_to_evaluate, labels=unique_labels)

    print("--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)
    
    results_file = os.path.join(output_dir, "evaluation_summary.txt")
    with open(results_file, 'w') as f:
        f.write(f"--- Evaluation Mode: {eval_mode} ---\n")
        # Add more details based on mode
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)

    # --- Open-Set Evaluation Logic ---
    if open_set_eval:
        if not known_classes:
            raise ValueError("--known-classes is required for open-set evaluation.")
        
        print("\n--- Open-Set Evaluation Results ---")
        
        y_true_openset = [1 if label in known_classes else 0 for label in all_labels]
        all_probs = np.array(all_probs)

        auroc = np.nan
        fpr_at_tpr95 = np.nan

        if len(np.unique(y_true_openset)) < 2:
            print("Warning: Only one class present in y_true_openset (all known or all unknown). Cannot compute AUROC and FPR@TPR95.")
        else:
            if eval_mode == 'gating_ensemble' and gating_has_garbage_class:
                print("Using garbage class probability for open-set evaluation.")
                prob_of_garbage = all_probs[:, -1]
                confidences = 1 - prob_of_garbage
            else:
                print("Using max softmax probability for open-set evaluation.")
                confidences = np.max(all_probs, axis=1)

            auroc = roc_auc_score(y_true_openset, confidences)
            
            fpr, tpr, thresholds = roc_curve(y_true_openset, confidences)
            tpr_ge_95_indices = np.where(tpr >= 0.95)[0]
            if len(tpr_ge_95_indices) > 0:
                idx = tpr_ge_95_indices[0]
                fpr_at_tpr95 = fpr[idx]
            else:
                idx = np.argmax(tpr)
                fpr_at_tpr95 = fpr[idx] if idx < len(fpr) else np.nan
                print(f"Warning: TPR never reached 0.95. Reporting FPR at highest TPR of {tpr[idx] if idx < len(tpr) else 'N/A'}")

        print(f"AUROC: {auroc:.4f}")
        print(f"FPR@TPR95: {fpr_at_tpr95:.4f}")
        
        with open(results_file, 'a') as f:
            f.write("\n\n--- Open-Set Evaluation ---\n")
            f.write(f"Known Classes: {known_classes}\n")
            if unknown_classes:
                f.write(f"Unknown Classes: {unknown_classes}\n")
            f.write(f"AUROC: {auroc:.4f}\n")
            f.write(f"FPR@TPR95: {fpr_at_tpr95:.4f}\n")

    print(f"\nResults saved to {results_file}")

    # Plot and save confusion matrix
    plt.figure(figsize=(15, 12))
    tick_labels = [str(l) for l in unique_labels]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(f'Confusion Matrix ({eval_mode.capitalize()} Mode)')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    evaluate()