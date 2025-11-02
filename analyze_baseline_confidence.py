
import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import os
import click
import torch
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from ruamel.yaml import YAML

# Make sure the ml module is in the python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import ResNet

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

@click.command()
@click.option("-m", "--model_path", required=True, help="Path to the trained ResNet model checkpoint.")
@click.option("-d", "--data_path", required=True, help="Path to the test data parquet file.")
@click.option("-o", "--output_csv", default="minority_analysis.csv", help="Path to save the output CSV file.")
@click.option(
    "--minority-classes",
    type=int,
    multiple=True,
    required=True,
    help="List of minority class labels to analyze (e.g., --minority-classes 5 --minority-classes 7)."
)
def analyze(model_path, data_path, output_csv, minority_classes):
    """
    Analyzes the baseline model's confidence on specified minority class samples.
    """
    # Convert tuple from click to a list for easy 'in' check
    minority_classes = list(minority_classes)
    print(f"Analyzing for minority classes: {minority_classes}")

    print(f"Loading ResNet model from {model_path}...")
    model = ResNet.load_from_checkpoint(model_path)
    model.eval()

    print(f"Loading data from {data_path}...")
    test_dataloader = load_data(data_path)

    print(f"Running predictions and analyzing minority classes...")
    results = []
    with torch.no_grad():
        for features, labels in test_dataloader:
            outputs = model(features)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get the predicted label (the one with the highest probability)
            max_probs, predicted_labels = torch.max(probabilities, 1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                
                # Check if the true label is one of the specified minority classes
                if true_label in minority_classes:
                    predicted_label = predicted_labels[i].item()
                    prob_vector = probabilities[i].cpu().numpy()
                    
                    result_row = {
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                    }
                    # Add all probabilities to the row
                    for j, prob in enumerate(prob_vector):
                        result_row[f'prob_class_{j}'] = prob
                    
                    results.append(result_row)

    if not results:
        print(f"No samples with minority classes {minority_classes} found in the dataset.")
        return

    # Create a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    # Reorder columns to have labels first
    cols = ['true_label', 'predicted_label'] + [col for col in results_df.columns if col not in ['true_label', 'predicted_label']]
    results_df = results_df[cols]
    
    results_df.to_csv(output_csv, index=False)
    print(f"Analysis complete. Results for minority classes saved to {output_csv}")

if __name__ == "__main__":
    analyze()
