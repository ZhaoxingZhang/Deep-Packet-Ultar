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

from ml.utils import (
    load_application_classification_cnn_model,
    load_application_classification_resnet_model,
)

def load_data(data_path):
    table = pq.read_table(data_path)
    df = table.to_pandas()
    
    features = torch.from_numpy(np.array(df['feature'].tolist(), dtype=np.float32))
    labels = torch.from_numpy(np.array(df['label'].tolist(), dtype=np.int64))
    
    if len(features.shape) == 2:
        features = features.unsqueeze(1)
        
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    return dataloader

def get_output_dir_from_model_path(model_path):
    """从模型路径提取输出目录"""
    # 获取模型文件名（不含扩展名）
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    # 输出目录为model/{model_name}
    output_dir = os.path.join("evaluation_results", model_name)
    return output_dir

@click.command()
@click.option("-m", "--model_path", required=True, help="Path to the trained model checkpoint.")
@click.option("-d", "--data_path", required=True, help="Path to the test data parquet directory.")
@click.option("-o", "--output_dir", help="Directory to save evaluation results. If not provided, will be derived from model path.")
@click.option("--model_type", default="cnn", help="Type of model to load: 'cnn' or 'resnet'.")
def evaluate(model_path, data_path, output_dir, model_type):
    """Evaluates a trained model on the given test set."""
    # 如果没有指定输出目录，从模型路径自动生成
    if output_dir is None:
        output_dir = get_output_dir_from_model_path(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading {model_type} model from {model_path}...")
    if model_type == 'cnn':
        model = load_application_classification_cnn_model(model_path)
    elif model_type == 'resnet':
        model = load_application_classification_resnet_model(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.eval()

    # Load data
    print(f"Loading data from {data_path}...")
    test_dataloader = load_data(data_path)

    # Run predictions
    print("Running predictions...")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    print("Calculating metrics...")
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save results to file
    results_file = os.path.join(output_dir, "evaluation_summary.txt")
    with open(results_file, 'w') as f:
        f.write("--- Evaluation Results ---\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
    print(f"\nResults saved to {results_file}")

    # Plot and save confusion matrix
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
