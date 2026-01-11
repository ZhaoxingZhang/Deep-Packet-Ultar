import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")
import click
import os
import pyarrow.parquet as pq

from ml.utils import (
    train_application_classification_cnn_model,
    train_traffic_classification_cnn_model,
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
@click.option(
    "--epochs",
    default=20,
    help="Number of epochs to train.",
    type=int,
)
def main(data_path, model_path, task, validation_split, sampling_strategy, epochs):
    data_path = os.path.abspath(data_path)
    model_path = os.path.abspath(model_path)
    if task == "app":
        # Calculate output_dim from the training data
        train_parquet_path = os.path.join(data_path, 'train.parquet')
        table = pq.read_table(train_parquet_path)
        output_dim = table['label'].to_pandas().max() + 1
        print(f"Dynamically determined output_dim: {output_dim}")

        train_application_classification_cnn_model(
            data_path, 
            model_path, 
            output_dim=output_dim, 
            validation_split=validation_split, 
            sampling_strategy=sampling_strategy,
            max_epochs=epochs
        )
    elif task == "traffic":
        # Dynamically determine output_dim from the training data
        table = pq.read_table(data_path)
        output_dim = table['label'].to_pandas().max() + 1
        print(f"Dynamically determined output_dim for traffic task: {output_dim}")

        train_traffic_classification_cnn_model(
            data_path, 
            model_path, 
            validation_split=validation_split, 
            output_dim=output_dim,
            max_epochs=epochs
        )
    else:
        exit("Not Support")


if __name__ == "__main__":
    main()
