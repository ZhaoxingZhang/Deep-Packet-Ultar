import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import os
import click
import torch
import pyarrow.parquet as pq
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

# Make sure the ml module is in the python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import MixtureOfExperts, ResNet

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
@click.option("-m", "--model_path", required=True, help="Path to the trained model checkpoint.")
@click.option("-d", "--data_path", required=True, help="Path to the test data parquet file.")
@click.option("-o", "--output_dir", help="Directory to save evaluation results. If not provided, will be derived from model path.")
@click.option("--model_type", default="cnn", type=click.Choice(['cnn', 'resnet', 'moe']), help="Type of model to load.")
@click.option("--moe_config_path", default="moe_config.yaml", help="Path to the MoE YAML config file (required for model_type='moe').")
def evaluate(model_path, data_path, output_dir, model_type, moe_config_path):
    """Evaluates a trained model on the given test set."""
    if output_dir is None:
        output_dir = get_output_dir_from_model_path(model_path)
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {model_type} model from {model_path}...")
    if model_type == 'cnn':
        raise NotImplementedError("CNN loading not implemented in this version.")
    elif model_type == 'resnet':
        model = ResNet.load_from_checkpoint(model_path)
    elif model_type == 'moe':
        yaml = YAML(typ='safe')
        with open(moe_config_path, 'r') as f:
            config = yaml.load(f)
        
        print(f"Loading Generalist expert from: {config['generalist_expert_path']}")
        generalist_expert = ResNet.load_from_checkpoint(config['generalist_expert_path'])
        print(f"Loading Minority expert from: {config['minority_expert_path']}")
        minority_expert = ResNet.load_from_checkpoint(config['minority_expert_path'])

        model = MixtureOfExperts.load_from_checkpoint(
            model_path,
            generalist_expert=generalist_expert,
            minority_expert=minority_expert
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()

    print(f"Loading data from {data_path}...")
    test_dataloader = load_data(data_path)

    print("Running predictions...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Calculating metrics...")
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    results_file = os.path.join(output_dir, "evaluation_summary.txt")
    with open(results_file, 'w') as f:
        f.write("--- Evaluation Results ---")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    print(f"\nResults saved to {results_file}")

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    evaluate()
