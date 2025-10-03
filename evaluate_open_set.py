import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import os
import click
import torch
import pyarrow.parquet as pq
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from ruamel.yaml import YAML
import json

# Make sure the ml module is in the python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import MixtureOfExperts, ResNet

def load_data(data_path):
    """Loads test data from a parquet file."""
    table = pq.read_table(data_path)
    df = table.to_pandas()
    
    features = torch.from_numpy(np.array(df['feature'].tolist(), dtype=np.float32))
    labels = torch.from_numpy(np.array(df['label'].tolist(), dtype=np.int64))
    
    if len(features.shape) == 2:
        features = features.unsqueeze(1)
        
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    return dataloader

def get_predictions(model, dataloader, model_type):
    """Gets predictions, probabilities, and gate outputs from a model."""
    model.eval()
    all_labels = []
    all_softmax_probs = []
    all_gate_outputs = []

    with torch.no_grad():
        for features, labels in dataloader:
            if model_type == 'moe':
                # MoE model's forward pass returns a dict
                result_dict = model(features, return_gate_outputs=True)
                outputs = result_dict['final_output']
                gate_outputs = result_dict['gate_outputs']
                all_gate_outputs.append(gate_outputs.cpu())
            else: # ResNet
                outputs = model(features)

            softmax_probs = F.softmax(outputs, dim=1)
            all_softmax_probs.append(softmax_probs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    all_labels = torch.cat(all_labels)
    all_softmax_probs = torch.cat(all_softmax_probs)
    if model_type == 'moe':
        all_gate_outputs = torch.cat(all_gate_outputs)

    return {
        "labels": all_labels,
        "softmax_probs": all_softmax_probs,
        "gate_outputs": all_gate_outputs if model_type == 'moe' else None
    }

def calculate_rejection_curve(labels, confidence_scores, predictions, known_classes):
    """Calculates the accuracy-rejection curve."""
    # Ensure tensors are on CPU
    labels = labels.cpu()
    confidence_scores = confidence_scores.cpu()
    predictions = predictions.cpu()

    curve_points = []
    # Define a range of thresholds to test
    thresholds = np.linspace(0, 1, 101)

    for tau in thresholds:
        # Samples with confidence below threshold are rejected
        max_confidence, _ = torch.max(confidence_scores, dim=1)
        rejected_mask = max_confidence < tau
        accepted_mask = ~rejected_mask

        num_total = len(labels)
        num_rejected = rejected_mask.sum().item()
        num_accepted = num_total - num_rejected

        rejection_rate = num_rejected / num_total if num_total > 0 else 0

        if num_accepted == 0:
            accuracy_on_accepted = 0.0
        else:
            accepted_labels = labels[accepted_mask]
            accepted_preds = predictions[accepted_mask]
            
            # Accuracy is calculated only on KNOWN classes that were accepted
            is_known_accepted_mask = torch.isin(accepted_labels, torch.tensor(known_classes, dtype=labels.dtype))
            
            if is_known_accepted_mask.sum().item() == 0:
                accuracy_on_accepted = 0.0
            else:
                correct_predictions = (accepted_preds[is_known_accepted_mask] == accepted_labels[is_known_accepted_mask]).sum().item()
                accuracy_on_accepted = correct_predictions / is_known_accepted_mask.sum().item()

        curve_points.append({"threshold": tau, "rejection_rate": rejection_rate, "accuracy": accuracy_on_accepted})
        
    return curve_points

@click.command()
@click.option('--baseline_model_path', required=True, type=click.Path(exists=True))
@click.option('--moe_model_path', required=True, type=click.Path(exists=True))
@click.option('--moe_config_path', required=True, type=click.Path(exists=True))
@click.option('--data_path', required=True, type=click.Path(exists=True))
@click.option('--output_file', required=True, type=click.Path())
def main(baseline_model_path, moe_model_path, moe_config_path, data_path, output_file):
    """
    Evaluates and compares Baseline and MoE models for open-set recognition
    by generating accuracy-rejection curves for different strategies.
    """
    # --- 1. Load Config and Data ---
    yaml = YAML(typ='safe')
    with open(moe_config_path, 'r') as f:
        config = yaml.load(f)

    known_classes_original = config['known_classes']
    # The model was trained on remapped labels 0..N-1
    known_classes_remapped = list(range(len(known_classes_original)))

    print(f"Loading test data from: {data_path}")
    test_dataloader = load_data(data_path)

    # --- 2. Load Models ---
    print(f"Loading Baseline ResNet model from: {baseline_model_path}")
    baseline_model = ResNet.load_from_checkpoint(baseline_model_path)

    print(f"Loading MoE model from: {moe_model_path}")
    generalist_expert = ResNet.load_from_checkpoint(config['generalist_expert_path'])
    minority_expert = ResNet.load_from_checkpoint(config['minority_expert_path'])
    moe_model = MixtureOfExperts.load_from_checkpoint(
        moe_model_path,
        generalist_expert=generalist_expert,
        minority_expert=minority_expert
    )

    # --- 3. Get Predictions ---
    print("Getting predictions from Baseline model...")
    baseline_preds_data = get_predictions(baseline_model, test_dataloader, 'resnet')
    
    print("Getting predictions from MoE model...")
    moe_preds_data = get_predictions(moe_model, test_dataloader, 'moe')

    # --- 4. Calculate Rejection Curves ---
    results = {}
    baseline_predictions = torch.argmax(baseline_preds_data['softmax_probs'], dim=1)
    moe_predictions = torch.argmax(moe_preds_data['softmax_probs'], dim=1)

    print("Calculating curve for Strategy 1: Baseline-Softmax")
    results['baseline_softmax'] = calculate_rejection_curve(
        baseline_preds_data['labels'], 
        baseline_preds_data['softmax_probs'],
        baseline_predictions,
        known_classes_remapped 
    )

    print("Calculating curve for Strategy 2: MoE-Softmax")
    results['moe_softmax'] = calculate_rejection_curve(
        moe_preds_data['labels'], 
        moe_preds_data['softmax_probs'],
        moe_predictions,
        known_classes_remapped
    )

    print("Calculating curve for Strategy 3: MoE-Gate")
    # For the gate, the prediction is which expert to choose, not a class.
    # We can't directly calculate accuracy here in the same way.
    # The confidence score is the gate output, and we measure how well rejecting by it improves accuracy of the main model.
    results['moe_gate'] = calculate_rejection_curve(
        moe_preds_data['labels'], 
        moe_preds_data['gate_outputs'],
        moe_predictions, # We still use the final predictions for accuracy calculation
        known_classes_remapped
    )

    # --- 5. Save Results ---
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nEvaluation complete. Results saved to {output_file}")


if __name__ == '__main__':
    main()