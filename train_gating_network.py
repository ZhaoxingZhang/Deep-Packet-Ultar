import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import os
import click
import torch
import pyarrow.parquet as pq
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from ruamel.yaml import YAML

# Make sure the ml module is in the python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import ResNet, GatingNetwork
from ml.losses import CombinedMacroF1CELoss, AdaptiveMacroF1CELoss

def load_feature_data(data_path):
    """Loads the raw feature data (for the main models)."""
    table = pq.read_table(data_path)
    df = table.to_pandas()
    
    features = torch.from_numpy(np.array(df['feature'].tolist(), dtype=np.float32))
    labels = torch.from_numpy(np.array(df['label'].tolist(), dtype=np.int64))
    
    # The model expects a channel dimension, so we add it.
    if len(features.shape) == 2:
        features = features.unsqueeze(1)
        
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    return dataloader

@click.command()
@click.option("--train_data_path", required=True, help="Path to the training data parquet file.")
@click.option("--baseline_model_path", required=True, help="Path to the pre-trained baseline model.")
@click.option("--minority_model_path", required=True, help="Path to the pre-trained minority expert model.")
@click.option("--minority_classes", type=int, multiple=True, required=True, help="The original labels of the minority classes.")
@click.option("--output_path", required=True, help="Path to save the trained Gating Network.")
@click.option("--epochs", type=int, default=10, help="Number of epochs to train the Gating Network.")
@click.option("--lr", type=float, default=0.001, help="Learning rate for the Gating Network.")
@click.option("--lambda_macro", type=float, default=0.5, help="Weight for Macro-F1 loss (0-1).")
@click.option("--use_adaptive", is_flag=True, default=False, help="Use adaptive Macro-F1 weighting.")
@click.option("--initial_lambda", type=float, default=0.1, help="Initial lambda for adaptive training.")
@click.option("--final_lambda", type=float, default=0.7, help="Final lambda for adaptive training.")
def train_gating_network(train_data_path, baseline_model_path, minority_model_path, minority_classes, output_path, epochs, lr, lambda_macro, use_adaptive, initial_lambda, final_lambda):
    """
    Trains a Gating Network to combine the outputs of a baseline and a minority expert model.
    """
    print("--- Starting Gating Network Training with Macro-F1 Optimization ---")

    # Validate lambda_macro
    if not 0 <= lambda_macro <= 1:
        print("Error: lambda_macro must be between 0 and 1")
        return
    
    # --- 1. Load Models (Frozen) ---
    print(f"Loading baseline model from {baseline_model_path}...")
    baseline_model = ResNet.load_from_checkpoint(baseline_model_path)
    baseline_model.eval()
    for param in baseline_model.parameters():
        param.requires_grad = False

    print(f"Loading minority expert model from {minority_model_path}...")
    minority_model = ResNet.load_from_checkpoint(minority_model_path)
    minority_model.eval()
    for param in minority_model.parameters():
        param.requires_grad = False

    # --- 2. Load Data ---
    print(f"Loading training data from {train_data_path}...")
    feature_dataloader = load_feature_data(train_data_path)

    # --- 3. Prepare Data for Gating Network Training ---
    num_total_classes = baseline_model.out.out_features
    expert_idx_to_original_label = {i: label for i, label in enumerate(minority_classes)}
    
    gating_train_inputs = []
    gating_train_labels = []
    
    print("Generating Gating Network training data by processing with base models...")
    with torch.no_grad():
        for features, labels in feature_dataloader:
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
            gating_train_inputs.append(torch.cat((base_probs, expert_probs_full), dim=1).cpu())
            gating_train_labels.append(labels.cpu())
    
    gating_train_inputs = torch.cat(gating_train_inputs, dim=0)
    gating_train_labels = torch.cat(gating_train_labels, dim=0)

    gating_dataset = TensorDataset(gating_train_inputs, gating_train_labels)
    gating_dataloader = DataLoader(gating_dataset, batch_size=64, shuffle=True)
    print(f"Created training dataset for Gating Network with {len(gating_dataset)} samples.")

    # --- 4. Train Gating Network ---
    gating_network = GatingNetwork(num_total_classes)
    optimizer = torch.optim.Adam(gating_network.parameters(), lr=lr)

    # Choose the appropriate loss function
    if use_adaptive:
        print(f"Using adaptive Macro-F1 + CE loss (λ: {initial_lambda} → {final_lambda})")
        criterion = AdaptiveMacroF1CELoss(
            num_classes=num_total_classes,
            initial_lambda=initial_lambda,
            final_lambda=final_lambda,
            total_epochs=epochs
        )
    else:
        print(f"Using combined Macro-F1 + CE loss (λ_macro = {lambda_macro})")
        criterion = CombinedMacroF1CELoss(
            num_classes=num_total_classes,
            lambda_macro=lambda_macro
        )

    print(f"Training Gating Network for {epochs} epochs...")
    for epoch in range(epochs):
        gating_network.train()
        total_loss = 0
        total_ce_loss = 0
        total_macro_f1_loss = 0
        correct_preds = 0
        total_samples = 0

        for inputs, labels in gating_dataloader:
            optimizer.zero_grad()

            # Split concatenated inputs back into two probability vectors
            current_base_probs = inputs[:, :num_total_classes]
            current_expert_probs_full = inputs[:, num_total_classes:]

            outputs = gating_network(current_base_probs, current_expert_probs_full)

            # Calculate loss
            if use_adaptive:
                loss_dict = criterion(outputs, labels, current_epoch=epoch)
                current_lambda = loss_dict['current_lambda']
            else:
                loss_dict = criterion(outputs, labels)
                current_lambda = lambda_macro

            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_ce_loss += loss_dict['ce_loss'].item()
            total_macro_f1_loss += loss_dict['macro_f1_loss'].item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate averages
        avg_loss = total_loss / len(gating_dataloader)
        avg_ce_loss = total_ce_loss / len(gating_dataloader)
        avg_macro_f1_loss = total_macro_f1_loss / len(gating_dataloader)
        accuracy = correct_preds / total_samples

        if use_adaptive:
            print(f"Epoch {epoch+1}/{epochs}, Total: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, "
                  f"Macro-F1 Loss: {avg_macro_f1_loss:.4f}, Acc: {accuracy:.4f}, λ: {current_lambda:.3f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Total: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, "
                  f"Macro-F1 Loss: {avg_macro_f1_loss:.4f}, Acc: {accuracy:.4f}, λ: {lambda_macro:.3f}")
    
    # --- 5. Save Gating Network ---
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    torch.save(gating_network.state_dict(), output_path)
    print(f"Gating Network training complete. Model saved to {output_path}")

if __name__ == "__main__":
    train_gating_network()
