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
def main(data_path, model_path, task):
    if task == "app":
        # Calculate output_dim from the training data
        train_parquet_path = os.path.join(data_path, 'train.parquet')
        table = pq.read_table(train_parquet_path)
        output_dim = len(table['label'].unique())
        print(f"Dynamically determined output_dim: {output_dim}")

        train_application_classification_resnet_model(data_path, model_path, output_dim=output_dim)
    elif task == "traffic":
        # Assuming similar logic for traffic classification if needed in the future
        train_traffic_classification_resnet_model(data_path, model_path)
    else:
        exit("Not Support")


if __name__ == "__main__":
    main()