import os
import random
import sys
from pathlib import Path
import shutil

# -- experiment_type Parameter Guide --
# This script uses the --experiment_type flag to define how the dataset should be processed.
# Each type corresponds to a different data splitting and sampling strategy for various experimental needs.
#
# - `imbalanced` (Recommended for baseline):
#   - What it does: Performs a standard, stratified train/test split based on `--test_size`.
#   - Key feature: The original imbalanced data distribution is preserved in both the training and testing sets.
#   - Use case: Training a model on the real-world data distribution.
#
# - `exp2`:
#   - What it does: Splits into train/test sets, then performs under-sampling on the **training set**.
#   - Key feature: The final training set is balanced (all classes have the same number of samples as the smallest class).
#     The test set remains imbalanced.
#   - Use case: Training a model on a balanced dataset to avoid bias towards majority classes, while testing on a realistic distribution.
#
# - `exp3`:
#   - What it does: Performs under-sampling on the **entire dataset** before splitting.
#   - Key feature: Both the training and testing sets are balanced.
#   - Use case: When you need balanced data for both training and testing, but be aware that this discards a significant amount of data from majority classes.
#
# - `exp1` / `exp_open_set`:
#   - What it does: Simulates an open-set recognition scenario. It splits classes into 'known' and 'unknown' groups.
#   - Key feature: The training set contains only 'known' classes. The test set contains a mix of 'known' and 'unknown' classes.
#     `exp1` uses ratios (`--known_ratio`) to define the groups, while `exp_open_set` uses a hard-coded list of classes.
#   - Use case: Evaluating a model's ability to handle classes it has never seen during training.
#
# - `exp8_majority` / `exp8_minority`:
#   - What it does: Filters the dataset to include only a specific, hard-coded list of majority or minority classes.
#   - Key feature: Creates a smaller, focused dataset for training expert models in a Mixture of Experts (MoE) setup.
#   - Use case: Training specialized models that only need to learn a subset of classes.
#
# - `exp_open_set_majority` / `exp_open_set_minority` / `exp_incremental_new`:
#   - What it does: Similar to `exp8`, these filter the dataset for specific, hard-coded lists of classes for various open-set and incremental learning experiments.
#   - Use case: Highly specific experimental setups.

import click
import psutil
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col,
    monotonically_increasing_id,
    lit,
    row_number,
    rand,
    floor,
    greatest,
)
from pyspark.sql.types import StructType, StructField, ArrayType, LongType, DoubleType


def top_n_per_group(spark_df, groupby, topn):
    spark_df = spark_df.withColumn("rand", rand(seed=9876))
    window = Window.partitionBy(col(groupby)).orderBy(col("rand"))

    return (
        spark_df.select(col("*"), row_number().over(window).alias("row_number"))
        .where(col("row_number") <= topn)
        .drop("row_number", "rand")
    )


def split_train_test(df, test_size, under_sampling_train=True):
    # Create a window partitioned by label, ordered randomly
    window = Window.partitionBy("label").orderBy(rand())

    # Assign a row number to each row within each label's partition
    df_with_row_number = df.withColumn("row_number", row_number().over(window))

    # Get the total count for each label
    label_counts = df.groupby("label").count()

    # Log labels that have only one sample, as they will only appear in the test set
    single_sample_labels_df = label_counts.filter(col("count") == 1)
    if single_sample_labels_df.count() > 0:
        single_sample_labels = [row.label for row in single_sample_labels_df.select("label").collect()]
        print(f"INFO: The following labels have only one sample and will be moved to the test set entirely: {sorted(single_sample_labels)}")

    # Join the row number and count information
    df_with_counts = df_with_row_number.join(label_counts, "label")

    # Determine the number of test samples for each label
    num_test_samples_expr = greatest(lit(1), floor(col("count") * test_size))
    
    # Split into train and test
    test_df = df_with_counts.where(col("row_number") <= num_test_samples_expr).select("feature", "label")
    train_df = df_with_counts.where(col("row_number") > num_test_samples_expr).select("feature", "label")

    # under sampling for training set
    if under_sampling_train:
        # get label list with count of each label
        label_count_df = train_df.groupby("label").count().toPandas()

        # if train_df is empty, label_count_df will be empty, skip under sampling
        if not label_count_df.empty:
            # get min label count in train set for under sampling
            min_label_count = int(label_count_df["count"].min())

            train_df = top_n_per_group(train_df, "label", min_label_count)

    return train_df, test_df


def save_parquet(df, path):
    output_path = path.absolute().as_uri()
    (df.write.mode("overwrite").parquet(output_path))


def save_train(df, path_dir):
    path = path_dir / "train.parquet"
    save_parquet(df, path)


def save_test(df, path_dir):
    path = path_dir / "test.parquet"
    save_parquet(df, path)


def create_train_test_for_task(
    df,
    label_col,
    test_size,
    data_dir_path,
    known_ratio,
    unknown_train_ratio,
    experiment_type,
    minority_classes,
):
    if experiment_type == "exp1":
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )

        # 1. Get all labels and split them into three groups
        all_labels = [row.label for row in task_df.select("label").distinct().collect()]
        print(f"DEBUG: Found {len(all_labels)} unique labels: {all_labels}")
        random.Random(9876).shuffle(all_labels)

        num_known = int(len(all_labels) * known_ratio)
        print(f"DEBUG: Calculated num_known: {num_known}")
        num_unknown_train = int(len(all_labels) * unknown_train_ratio)

        known_labels = all_labels[:num_known]
        unknown_train_labels = all_labels[num_known : num_known + num_unknown_train]
        unknown_test_labels = all_labels[num_known + num_unknown_train :]
        
        # The new label for the 'unknown' class will be the next available integer
        unknown_class_label = max(all_labels) + 1

        print(f"Known labels: {known_labels}")
        print(f"Unknown labels for training: {unknown_train_labels}")
        print(f"Unknown labels for testing: {unknown_test_labels}")
        print(f"'Unknown' class label will be: {unknown_class_label}")

        # 2. Create a final mapping for the labels to be contiguous from 0
        final_labels = sorted(known_labels + [unknown_class_label])
        label_mapping = {label: i for i, label in enumerate(final_labels)}
        print(f"Final label mapping for the model: {label_mapping}")

        # 3. Create Training Set
        known_train_df = task_df.filter(col("label").isin(known_labels))
        unknown_train_df = task_df.filter(col("label").isin(unknown_train_labels)) \
                                  .withColumn("label", lit(unknown_class_label))
        
        train_df = known_train_df.unionAll(unknown_train_df)

        # 4. Create Test Set
        known_test_df = task_df.filter(col("label").isin(known_labels))
        unknown_test_df = task_df.filter(col("label").isin(unknown_test_labels)) \
                                 .withColumn("label", lit(unknown_class_label))
        test_df = known_test_df.unionAll(unknown_test_df)

        # 5. Remap labels in both dataframes to be contiguous
        from pyspark.sql.functions import create_map
        from itertools import chain
        mapping_expr = create_map([lit(x) for x in chain(*label_mapping.items())])

        train_df = train_df.withColumn("label", mapping_expr[col("label")])
        test_df = test_df.withColumn("label", mapping_expr[col("label")])

        print("saving train")
        save_train(train_df, data_dir_path)
        print("saving train done")
        print("saving test")
        save_test(test_df, data_dir_path)
        print("saving test done")
    elif experiment_type == "exp2":
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )
        train_df, test_df = split_train_test(task_df, test_size)
        print("saving train")
        save_train(train_df, data_dir_path)
        print("saving train done")
        print("saving test")
        save_test(test_df, data_dir_path)
        print("saving test done")
    elif experiment_type == "exp3":
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )
        # get label list with count of each label
        label_count_df = task_df.groupby("label").count().toPandas()

        # get min label count in train set for under sampling
        min_label_count = int(label_count_df["count"].min())

        balanced_df = top_n_per_group(task_df, "label", min_label_count)

        train_df, test_df = split_train_test(balanced_df, test_size, under_sampling_train=False)
        print("saving train")
        save_train(train_df, data_dir_path)
        print("saving train done")
        print("saving test")
        save_test(test_df, data_dir_path)
        print("saving test done")
    
    elif experiment_type == "exp8_majority":
        majority_classes = [2, 3, 5, 10]
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )
        filtered_df = task_df.filter(col('label').isin(majority_classes))
        
        label_mapping = {original_label: i for i, original_label in enumerate(sorted(majority_classes))}
        from pyspark.sql.functions import create_map
        from itertools import chain
        mapping_expr = create_map([lit(x) for x in chain(*label_mapping.items())])
        
        mapped_df = filtered_df.withColumn("label", mapping_expr[col("label")])
        
        train_df, test_df = split_train_test(mapped_df, test_size, under_sampling_train=False)
        
        print("--- Creating dataset for MAJORITY expert ---")
        print(f"Original classes: {majority_classes}")
        print(f"Remapped to: {list(label_mapping.values())}")
        save_train(train_df, data_dir_path)
        save_test(test_df, data_dir_path)
        print("------------------------------------------")
    elif experiment_type == "exp8_minority":
        if not minority_classes:
            raise ValueError("The --minority-classes argument is required for experiment_type 'exp8_minority'")
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )
        filtered_df = task_df.filter(col('label').isin(*minority_classes))
        
        label_mapping = {original_label: i for i, original_label in enumerate(sorted(minority_classes))}
        from pyspark.sql.functions import create_map
        from itertools import chain
        mapping_expr = create_map([lit(x) for x in chain(*label_mapping.items())])
        
        mapped_df = filtered_df.withColumn("label", mapping_expr[col("label")])
        
        train_df, test_df = split_train_test(mapped_df, test_size, under_sampling_train=False)

        print("--- Creating dataset for MINORITY expert ---")
        print(f"Original classes: {minority_classes}")
        print(f"Remapped to: {list(label_mapping.values())}")
        save_train(train_df, data_dir_path)
        save_test(test_df, data_dir_path)
        print("------------------------------------------")
    elif experiment_type == "exp_open_set":
        known_classes = [2, 3, 5, 1, 7, 8, 9, 11, 13, 14]
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )

        # Split the full dataset to get a representative test set
        train_df_full, test_df = split_train_test(task_df, test_size, under_sampling_train=False)

        # The training set should only contain known classes
        train_df = train_df_full.filter(col('label').isin(known_classes))

        # Remap labels for the training set to be contiguous from 0-9
        label_mapping = {original_label: i for i, original_label in enumerate(sorted(known_classes))}
        from pyspark.sql.functions import create_map
        from itertools import chain
        mapping_expr = create_map([lit(x) for x in chain(*label_mapping.items())])
        
        train_df = train_df.withColumn("label", mapping_expr[col("label")])

        print("--- Creating dataset for Open-Set experiment ---")
        print(f"Known classes: {known_classes}")
        print(f"Training set will be remapped to: {list(label_mapping.values())}")
        print("Test set will contain all original 15 classes.")
        save_train(train_df, data_dir_path)
        save_test(test_df, data_dir_path)
        print("---------------------------------------------")
    elif experiment_type == "exp_open_set_majority":
        majority_classes = [2, 3, 5]
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )
        filtered_df = task_df.filter(col('label').isin(majority_classes))
        
        label_mapping = {original_label: i for i, original_label in enumerate(sorted(majority_classes))}
        from pyspark.sql.functions import create_map
        from itertools import chain
        mapping_expr = create_map([lit(x) for x in chain(*label_mapping.items())])
        
        mapped_df = filtered_df.withColumn("label", mapping_expr[col("label")])
        
        train_df, test_df = split_train_test(mapped_df, test_size, under_sampling_train=False)
        
        print("--- Creating dataset for Open-Set MAJORITY expert ---")
        print(f"Original classes: {majority_classes}")
        print(f"Remapped to: {list(label_mapping.values())}")
        save_train(train_df, data_dir_path)
        save_test(test_df, data_dir_path)
        print("------------------------------------------")
    elif experiment_type == "exp_open_set_minority":
        minority_classes = [1, 7, 8, 9, 11, 13, 14]
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )
        filtered_df = task_df.filter(col('label').isin(minority_classes))
        
        label_mapping = {original_label: i for i, original_label in enumerate(sorted(minority_classes))}
        from pyspark.sql.functions import create_map
        from itertools import chain
        mapping_expr = create_map([lit(x) for x in chain(*label_mapping.items())])
        
        mapped_df = filtered_df.withColumn("label", mapping_expr[col("label")])
        
        train_df, test_df = split_train_test(mapped_df, test_size, under_sampling_train=False)

        print("--- Creating dataset for Open-Set MINORITY expert ---")
        print(f"Original classes: {minority_classes}")
        print(f"Remapped to: {list(label_mapping.values())}")
        save_train(train_df, data_dir_path)
        save_test(test_df, data_dir_path)
        print("------------------------------------------")
    elif experiment_type == "exp_incremental_new":
        new_classes = [10, 0, 4, 6, 12]
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )
        filtered_df = task_df.filter(col('label').isin(new_classes))
        
        label_mapping = {original_label: i for i, original_label in enumerate(sorted(new_classes))}
        from pyspark.sql.functions import create_map
        from itertools import chain
        mapping_expr = create_map([lit(x) for x in chain(*label_mapping.items())])
        
        mapped_df = filtered_df.withColumn("label", mapping_expr[col("label")])
        
        train_df, test_df = split_train_test(mapped_df, test_size, under_sampling_train=False)

        print("--- Creating dataset for Incremental NEW classes ---")
        print(f"Original classes: {new_classes}")
        print(f"Remapped to: {list(label_mapping.values())}")
        save_train(train_df, data_dir_path)
        save_test(test_df, data_dir_path)
        print("------------------------------------------")
    elif experiment_type == "imbalanced":
        task_df = df.filter(col(label_col).isNotNull()).selectExpr(
            "feature", f"{label_col} as label"
        )
        train_df, test_df = split_train_test(task_df, test_size, under_sampling_train=False)
        print("--- Creating dataset with imbalanced train/test split ---")
        save_train(train_df, data_dir_path)
        save_test(test_df, data_dir_path)
        print("--------------------------------------------------------")


def print_df_label_distribution(spark, path):
    print(path)
    print(
        spark.read.parquet(path.absolute().as_uri()).groupby("label").count().toPandas()
    )


@click.command()
@click.option(
    "-s",
    "--source",
    help="path to the directory containing preprocessed files",
    required=True,
)
@click.option(
    "-t",
    "--target",
    help="path to the directory for persisting train and test set",
    required=True,
)
@click.option("--test_size", default=0.2, help="size of test size", type=float)
@click.option(
    "--known_ratio",
    default=0.3,
    help="Ratio of classes to be used as 'known' classes.",
    type=float,
)
@click.option(
    "--unknown_train_ratio",
    default=0.3,
    help="Ratio of classes to be used as 'unknown' for training.",
    type=float,
)
@click.option(
    "--experiment_type",
    type=click.Choice(["exp1", "exp2", "exp3", "exp8_majority", "exp8_minority", "exp_open_set", "exp_open_set_majority", "exp_open_set_minority", "exp_incremental_new", "imbalanced"], case_sensitive=False),
    default="exp1",
    help="Type of experiment to generate data for.",
)
@click.option(
    "--fraction",
    default=None,
    help="Fraction of the total dataset to sample. Overrides file-level sampling if set.",
    type=float,
)
@click.option(
    "--batch_size",
    default=50,
    help="Number of files to process in each batch when using fractional sampling.",
    type=int,
)
@click.option(
    "--minority-classes",
    type=int,
    multiple=True,
    help="List of class labels to be treated as minority classes for exp8_minority."
)
def main(source, target, test_size, known_ratio, unknown_train_ratio, experiment_type, fraction, batch_size, minority_classes):
    source_data_dir_path = Path(source)
    target_data_dir_path = Path(target)

    # prepare dir for dataset
    application_data_dir_path = target_data_dir_path / "application_classification"
    traffic_data_dir_path = target_data_dir_path / "traffic_classification"

    # initialise local spark
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    memory_gb = psutil.virtual_memory().available // 1024 // 1024 // 1024
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.memory", f"{memory_gb}g")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )

    # read data
    schema = StructType(
        [
            StructField("app_label", LongType(), True),
            StructField("traffic_label", LongType(), True),
            StructField("feature", ArrayType(DoubleType()), True),
        ]
    )

    if fraction is not None:
        print(f"Starting fractional sampling with fraction: {fraction}, batch size: {batch_size}")
        all_files = [p.absolute().as_uri() for p in source_data_dir_path.glob("*.json.gz")]
        if not all_files:
            raise ValueError(f"No .json.gz files found in source directory: {source}")

        temp_dir = target_data_dir_path / "temp_sampled_data"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{-(-len(all_files)//batch_size)}...")
            batch_df = spark.read.schema(schema).json(batch_files)
            sampled_batch_df = batch_df.sample(fraction=fraction, seed=42)

            # Coalesce into a single partition to avoid issues with empty partitions
            coalesced_df = sampled_batch_df.coalesce(1)

            if coalesced_df.count() > 0:
                coalesced_df.write.mode("append").parquet(temp_dir.absolute().as_uri())

        print("Consolidating sampled data...")
        df = spark.read.parquet(temp_dir.absolute().as_uri())
        
        # Cache the dataframe and trigger an action to force materialization
        df = df.cache()
        print(f"Consolidated dataframe has {df.count()} rows.")

        # Clean up temporary directory now that the data is cached
        shutil.rmtree(temp_dir)
        
        # The user-provided experiment_type should be respected.
        # The --fraction flag is only for initial data sampling.
        pass

    else:
        # Original logic for other experiments that load the entire dataset
        print("Loading all files from source for non-fractional experiment...")
        # Use a recursive glob pattern to find all part files.
        df = spark.read.schema(schema).json(
            f"{source_data_dir_path.absolute().as_uri()}/*.json.gz"
        )

    # prepare data for application classification and traffic classification
    print("processing application classification dataset")
    create_train_test_for_task(
        df=df,
        label_col="app_label",
        test_size=test_size,
        data_dir_path=application_data_dir_path,
        known_ratio=known_ratio,
        unknown_train_ratio=unknown_train_ratio,
        experiment_type=experiment_type,
        minority_classes=minority_classes,
    )

    print("processing traffic classification dataset")
    create_train_test_for_task(
        df=df,
        label_col="traffic_label",
        test_size=test_size,
        data_dir_path=traffic_data_dir_path,
        known_ratio=known_ratio,
        unknown_train_ratio=unknown_train_ratio,
        experiment_type=experiment_type,
        minority_classes=minority_classes,
    )

    # stats
    print("Final label distribution for application classification:")
    print_df_label_distribution(spark, application_data_dir_path / "train.parquet")
    print_df_label_distribution(spark, application_data_dir_path / "test.parquet")
    
    print("Final label distribution for traffic classification:")
    print_df_label_distribution(spark, traffic_data_dir_path / "train.parquet")
    print_df_label_distribution(spark, traffic_data_dir_path / "test.parquet")


if __name__ == "__main__":
    main()