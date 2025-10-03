
import click
import torch
import pyarrow.parquet as pq
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from ruamel.yaml import YAML
import os
import sys

# Make sure the ml module is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import ResNet, MixtureOfExperts

@click.command()
@click.option("--model_path", required=True, type=click.Path(exists=True))
@click.option("--config_path", required=True, type=click.Path(exists=True), help="Path to the YAML config file used for training the model.")
@click.option("--model_type", required=True, type=click.Choice(['resnet', 'moe']))
def get_accuracy(model_path, config_path, model_type):
    """Calculates the accuracy of a model on the known classes from its corresponding test set."""

    # 1. Load Config and Data
    yaml = YAML(typ='safe')
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    data_path = config['data_path']
    test_data_path = os.path.join(data_path, 'test.parquet')
    known_classes_original = config['known_classes']
    label_mapping = {original: i for i, original in enumerate(sorted(known_classes_original))}

    print(f"Loading test data from: {test_data_path}")
    table = pq.read_table(test_data_path)
    df = table.to_pandas()

    # Filter for known classes only
    df_known = df[df['label'].isin(known_classes_original)]
    print(f"Found {len(df_known)} samples belonging to the {len(known_classes_original)} known classes.")

    features = torch.from_numpy(np.array(df_known['feature'].tolist(), dtype=np.float32))
    labels_original = torch.from_numpy(np.array(df_known['label'].tolist(), dtype=np.int64))

    # Remap labels to the 0-(N-1) space the model was trained on
    labels_remapped = torch.tensor([label_mapping.get(l.item(), -1) for l in labels_original])

    if len(features.shape) == 2:
        features = features.unsqueeze(1)
        
    dataset = TensorDataset(features, labels_remapped)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    # 2. Load Model
    print(f"Loading {model_type} model from {model_path}...")
    if model_type == 'resnet':
        model = ResNet.load_from_checkpoint(model_path)
    elif model_type == 'moe':
        generalist_expert = ResNet.load_from_checkpoint(config['generalist_expert_path'])
        minority_expert = ResNet.load_from_checkpoint(config['minority_expert_path'])
        model = MixtureOfExperts.load_from_checkpoint(
            model_path,
            generalist_expert=generalist_expert,
            minority_expert=minority_expert
        )
    else:
        raise ValueError("Invalid model type")

    model.eval()

    # 3. Calculate Accuracy
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features_batch, labels_batch in dataloader:
            outputs = model(features_batch)
            if isinstance(outputs, dict):
                outputs = outputs['final_output']
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    correct_predictions = (all_preds == all_labels).sum().item()
    total_samples = len(all_labels)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    print("\n--- Initial Performance on Known Classes ---")
    print(f"Model: {model_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print("-------------------------------------------")

if __name__ == '__main__':
    get_accuracy()
