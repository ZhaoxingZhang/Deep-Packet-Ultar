import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import click
import os
import pyarrow.parquet as pq
from pathlib import Path

from ml.utils import (
    train_application_classification_resnet_model,
    train_traffic_classification_resnet_model,
)


@click.command()
@click.option(
    "-d",
    "--data_path",
    help="training data dir path containing parquet files",
    required=True,
)
@click.option("-m", "--model_path", help="output model path", required=True)
@click.option(
    "-t",
    "--task",
    help='classification task. Option: "app" or "traffic"',
    required=True,
)
@click.option(
    "--use_attention",
    is_flag=True,
    default=False,
    help="Enable attention mechanism in the ResNet model."
)
@click.option(
    "--validation_split",
    default=0.1,
    help="Fraction of the training data to use for validation.",
    type=float,
)
@click.option(
    "--loss_type",
    default='cross_entropy',
    help="Type of loss function to use.",
    type=click.Choice(['cross_entropy', 'focal_loss']),
)
def main(data_path, model_path, task, use_attention, validation_split, loss_type):
    if task == "app":
        # Calculate output_dim from the training data
        train_parquet_path = os.path.join(data_path, 'train.parquet')
        table = pq.read_table(train_parquet_path)
        output_dim = table['label'].to_pandas().max() + 1
        print(f"Dynamically determined output_dim: {output_dim}")

        train_application_classification_resnet_model(data_path, model_path, output_dim=output_dim, use_attention=use_attention, validation_split=validation_split, loss_type=loss_type)
    elif task == "traffic":
        # Assuming similar logic for traffic classification if needed in the future
        train_traffic_classification_resnet_model(data_path, model_path, validation_split=validation_split)
    else:
        exit("Not Support")


if __name__ == "__main__":
    main()