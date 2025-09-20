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
    "--validation_split",
    default=0.1,
    help="Fraction of the training data to use for validation.",
    type=float,
)
@click.option(
    "--sampling_strategy",
    default='random',
    help="The sampling strategy to use.",
    type=click.Choice(['random', 'class_aware']),
)
def main(data_path, model_path, task, validation_split, sampling_strategy):
    if task == "app":
        # Calculate output_dim from the training data
        train_parquet_path = os.path.join(data_path, 'train.parquet')
        table = pq.read_table(train_parquet_path)
        output_dim = table['label'].to_pandas().max() + 1
        print(f"Dynamically determined output_dim: {output_dim}")

        train_application_classification_resnet_model(data_path, model_path, output_dim=output_dim, validation_split=validation_split, sampling_strategy=sampling_strategy)
    elif task == "traffic":
        # Assuming similar logic for traffic classification if needed in the future
        train_traffic_classification_resnet_model(data_path, model_path, validation_split=validation_split)
    else:
        exit("Not Support")


if __name__ == "__main__":
    main()