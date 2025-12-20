import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import os
import click
import torch
import pyarrow.parquet as pq
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Make sure the ml module is in the python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import ResNet, GatingNetwork, CNN
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_feature_data(data_path, validation_split=0.0, seed=9876, train_on_val=False):
    """Loads the raw feature data and optionally splits it."""
    table = pq.read_table(data_path)
    df = table.to_pandas()
    
    features = torch.from_numpy(np.array(df['feature'].tolist(), dtype=np.float32))
    labels = torch.from_numpy(np.array(df['label'].tolist(), dtype=np.int64))
    
    # The model expects a channel dimension, so we add it.
    if len(features.shape) == 2:
        features = features.unsqueeze(1)
        
    full_dataset = TensorDataset(features, labels)
    
    if validation_split > 0:
        seed_everything(seed)
        val_size = int(len(full_dataset) * validation_split)
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        dataset = val_ds if train_on_val else train_ds
        print(f"  -> Split data: total={len(full_dataset)}, used={'val' if train_on_val else 'train'}={len(dataset)}")
    else:
        dataset = full_dataset
        
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    return dataloader

def generate_gating_inputs(feature_dataloader, baseline_model, minority_model, minority_classes, device):
    """
    Processes raw features through baseline and expert models to generate inputs
    for the Gating Network.
    Returns:
        - A tensor of concatenated probability vectors.
        - A tensor of original labels.
    """
    expert_idx_to_original_label = {i: label for i, label in enumerate(minority_classes)}
    
    gating_inputs = []
    original_labels = []
    
    print("  -> Generating model predictions for gating input...")
    with torch.no_grad():
        for features, labels in feature_dataloader:
            features = features.to(device)
            # Get probabilities from baseline model
            base_outputs = baseline_model(features)
            base_probs = torch.nn.functional.softmax(base_outputs, dim=1)

            # Get probabilities from minority expert
            expert_outputs = minority_model(features)
            expert_probs_small = torch.nn.functional.softmax(expert_outputs, dim=1)

            # Align expert probabilities to the full class space
            expert_probs_full = torch.zeros_like(base_probs)
            for expert_local_idx, original_label_idx in expert_idx_to_original_label.items():
                if expert_local_idx < expert_probs_small.shape[1]:
                    expert_probs_full[:, original_label_idx] = expert_probs_small[:, expert_local_idx]
            
            # The input to the gating network is the concatenated probabilities
            gating_inputs.append(torch.cat((base_probs, expert_probs_full), dim=1).cpu())
            original_labels.append(labels.cpu())
    
    gating_inputs = torch.cat(gating_inputs, dim=0)
    original_labels = torch.cat(original_labels, dim=0)
    
    return gating_inputs, original_labels

@click.command()
@click.option("--train_data_path", required=True, help="Path to the training data parquet file for KNOWN classes.")
@click.option("--baseline_model_path", required=True, help="Path to the pre-trained baseline model.")
@click.option("--minority_model_path", required=True, help="Path to the pre-trained minority expert model.")
@click.option("--baseline_model_type", type=click.Choice(['resnet', 'cnn']), default='resnet', help="Type of the baseline model.")
@click.option("--minority_model_type", type=click.Choice(['resnet', 'cnn']), default='resnet', help="Type of the minority expert model.")
@click.option("--minority_classes", type=int, multiple=True, required=True, help="The original labels of the minority classes.")
@click.option("--output_path", required=True, help="Path to save the trained Gating Network.")
@click.option("--epochs", type=int, default=10, help="Number of epochs to train the Gating Network.")
@click.option("--lr", type=float, default=0.001, help="Learning rate for the Gating Network.")
@click.option("--use-garbage-class", is_flag=True, help="Enable training with a garbage class for open-set.")
@click.option("--unknown-class-data-path", default=None, help="Path to the data for the UNKNOWN/GARBAGE class. Required if --use-garbage-class is set.")
@click.option("--validation_split", type=float, default=0.1, help="Validation split used for the models.")
@click.option("--seed", type=int, default=9876, help="Random seed for splitting.")
@click.option("--train_on_val", is_flag=True, help="Train the gating network only on the validation split of the training data.")
def train_gating_network(
    train_data_path,
    baseline_model_path,
    minority_model_path,
    baseline_model_type,
    minority_model_type,
    minority_classes,
    output_path,
    epochs,
    lr,
    use_garbage_class,
    unknown_class_data_path,
    validation_split,
    seed,
    train_on_val
):
    """
    Trains a Gating Network to combine the outputs of a baseline and a minority expert model.
    Can optionally include a 'garbage' class for open-set recognition training.
    """
    title = "Gating Network Training (Garbage Class)" if use_garbage_class else "Gating Network Training (Weighted Cross-Entropy)"
    print(f"--- Starting {title} ---")
    
    # --- 1. Load Models (Frozen) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def load_model(model_type, model_path):
        if model_type == 'resnet':
            return ResNet.load_from_checkpoint(model_path)
        elif model_type == 'cnn':
            return CNN.load_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    print(f"Loading baseline {baseline_model_type} model from {baseline_model_path}...")
    baseline_model = load_model(baseline_model_type, baseline_model_path).to(device)
    baseline_model.eval()
    for param in baseline_model.parameters():
        param.requires_grad = False

    print(f"Loading minority expert {minority_model_type} model from {minority_model_path}...")
    minority_model = load_model(minority_model_type, minority_model_path).to(device)
    minority_model.eval()
    for param in minority_model.parameters():
        param.requires_grad = False

    num_known_classes = baseline_model.out.out_features

    # --- 2. Prepare Data for Gating Network Training ---
    print(f"Loading KNOWN class data from {train_data_path}...")
    known_feature_dataloader = load_feature_data(train_data_path, validation_split=validation_split, seed=seed, train_on_val=train_on_val)
    gating_known_inputs, gating_known_labels = generate_gating_inputs(
        known_feature_dataloader, baseline_model, minority_model, minority_classes, device
    )

    if use_garbage_class:
        if not unknown_class_data_path:
            raise ValueError("--unknown-class-data-path must be provided when --use-garbage-class is enabled.")
        
        print(f"Loading UNKNOWN class data from {unknown_class_data_path}...")
        # For garbage data, we typically use the whole set if it's already a separate "unknown" set
        unknown_feature_dataloader = load_feature_data(unknown_class_data_path)
        gating_garbage_inputs, _ = generate_gating_inputs(
            unknown_feature_dataloader, baseline_model, minority_model, minority_classes, device
        )
        
        garbage_label = num_known_classes  # New label for the garbage class
        gating_garbage_labels = torch.full((gating_garbage_inputs.shape[0],), garbage_label, dtype=torch.int64)
        
        print(f"Combining {gating_known_inputs.shape[0]} known samples with {gating_garbage_inputs.shape[0]} unknown (garbage) samples.")
        
        gating_train_inputs = torch.cat((gating_known_inputs, gating_garbage_inputs), dim=0)
        gating_train_labels = torch.cat((gating_known_labels, gating_garbage_labels), dim=0)
    else:
        gating_train_inputs = gating_known_inputs
        gating_train_labels = gating_known_labels

    full_gating_dataset = TensorDataset(gating_train_inputs, gating_train_labels)
    
    # --- 3. Create Training and Validation Sets for the Gating Network itself ---
    # We still split the gating dataset to monitor its own training
    gating_val_split = 0.1
    gating_val_size = int(len(full_gating_dataset) * gating_val_split)
    gating_train_size = len(full_gating_dataset) - gating_val_size
    gating_train_dataset, gating_val_dataset = torch.utils.data.random_split(full_gating_dataset, [gating_train_size, gating_val_size])

    train_dataloader = DataLoader(gating_train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(gating_val_dataset, batch_size=256, shuffle=False)
    print(f"Created gating training dataset with {len(gating_train_dataset)} samples and validation dataset with {len(gating_val_dataset)} samples.")

    # --- 4. Train Gating Network ---
    num_output_classes = num_known_classes + 1 if use_garbage_class else num_known_classes
    
    train_labels = gating_train_dataset.dataset.tensors[1][gating_train_dataset.indices]
    class_counts_full = torch.bincount(train_labels, minlength=num_output_classes)
    
    # Improved class weights: only weight classes that are present
    active_classes = (class_counts_full > 0).sum()
    class_weights = torch.zeros_like(class_counts_full, dtype=torch.float)
    if active_classes > 0:
        class_weights[class_counts_full > 0] = len(train_labels) / (active_classes * class_counts_full[class_counts_full > 0].float())
    
    class_weights = class_weights.to(device)
    print(f"Calculated class weights: {class_weights}")

    gating_network = GatingNetwork(
        num_classes=num_known_classes, 
        use_garbage_class=use_garbage_class
    ).to(device)
    optimizer = torch.optim.Adam(gating_network.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float('inf')
    print(f"Training Gating Network for {epochs} epochs...")

    for epoch in range(epochs):
        gating_network.train()
        total_train_loss = 0
        train_correct_preds = 0
        train_total_samples = 0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            current_base_probs = inputs[:, :num_known_classes]
            current_expert_probs_full = inputs[:, num_known_classes:]

            outputs = gating_network(current_base_probs, current_expert_probs_full)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct_preds += (predicted == labels).sum().item()
            train_total_samples += labels.size(0)

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = train_correct_preds / train_total_samples

        gating_network.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                current_base_probs = inputs[:, :num_known_classes]
                current_expert_probs_full = inputs[:, num_known_classes:]
                
                outputs = gating_network(current_base_probs, current_expert_probs_full)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torch.save(gating_network.state_dict(), output_path)
            print(f"  -> New best model saved to {output_path} (Val Loss: {best_val_loss:.4f})")

    print(f"\n Gating Network training complete. Best model (Val Loss: {best_val_loss:.4f}) is saved at {output_path}")

if __name__ == "__main__":
    train_gating_network()
