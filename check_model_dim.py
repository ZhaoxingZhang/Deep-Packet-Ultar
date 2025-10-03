
import click
import torch
import sys
import os

# Make sure the ml module is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.model import ResNet

@click.command()
@click.option("--model_path", required=True, type=click.Path(exists=True), help="Path to the model checkpoint file.")
def check_dim(model_path):
    """Loads a ResNet model and prints its output dimension."""
    try:
        # Load the model from the checkpoint.
        # The `hparams` attribute will contain the arguments passed to the model's __init__ method.
        model = ResNet.load_from_checkpoint(model_path)
        
        # Access the output_dim from the saved hyperparameters.
        output_dim = model.hparams.output_dim
        
        print(f"--- Model Inspection Result ---")
        print(f"Model Path: {model_path}")
        print(f"Detected Output Dimension: {output_dim}")
        print(f"-----------------------------")

    except Exception as e:
        print(f"Failed to load or inspect model: {model_path}")
        print(f"Error: {e}")

if __name__ == "__main__":
    check_dim()
