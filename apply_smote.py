
import click
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from imblearn.over_sampling import SMOTE

@click.command()
@click.option(
    "--input_train_path",
    help="Path to the imbalanced training data parquet directory.",
    required=True,
)
@click.option(
    "--output_dir",
    help="Directory to save the new balanced dataset.",
    required=True,
)
def main(input_train_path, output_dir):
    """
    Applies SMOTE to a training set to balance the classes.
    """
    print(f"Loading data from {input_train_path}...")
    input_path = Path(input_train_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read the Parquet file into a pandas DataFrame
    df = pq.read_table(input_path).to_pandas()
    print("Original dataset shape: %s" % str(df.shape))
    print("Original dataset samples per class: \n%s" % str(df['label'].value_counts()))

    # Separate features (X) and labels (y)
    # The 'feature' column needs to be converted from a list of floats to a 2D numpy array
    X = np.array(df['feature'].tolist())
    y = df['label']

    print("\nApplying SMOTE...")
    
    # Dynamically set k_neighbors based on the smallest class size
    min_samples = y.value_counts().min()
    if min_samples < 2:
        raise ValueError(
            f"SMOTE cannot be applied. The smallest class has {min_samples} samples, "
            f"which is less than the minimum required (2)."
        )
    
    # k_neighbors must be less than the number of samples in the smallest class.
    k_neighbors = min_samples - 1
    print(f"Smallest class has {min_samples} samples. Setting k_neighbors to {k_neighbors}.")

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("Resampled dataset shape: %s" % str(X_resampled.shape))
    
    # Create a new DataFrame from the resampled data
    # The feature column must be converted back to a list of lists for pyarrow
    resampled_df = pd.DataFrame({'feature': list(X_resampled), 'label': y_resampled})
    print("Resampled dataset samples per class: \n%s" % str(resampled_df['label'].value_counts()))

    # Save the new balanced DataFrame to a parquet file
    output_file = output_path / "train.parquet"
    resampled_df.to_parquet(output_file)
    print(f"\nSaved balanced training set to: {output_file}")

if __name__ == "__main__":
    main()
