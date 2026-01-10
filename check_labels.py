
import argparse
import glob
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, ArrayType, LongType, DoubleType
from utils import ID_TO_TRAFFIC, ID_TO_APP

DEFAULT_SOURCE_DIR = "/Users/weisman/Documents/repo_root/graduate/Deep-Packet-Ultar/processed_data/vpn"

def main():
    parser = argparse.ArgumentParser(description="Analyze label distribution in processed data.")
    parser.add_argument("-s", "--source", default=DEFAULT_SOURCE_DIR, help="Path to the source data directory")
    parser.add_argument("-t", "--type", choices=["traffic", "app"], default="traffic", help="Label type to analyze (traffic or app)")
    args = parser.parse_args()

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )

    schema = StructType([
        StructField("app_label", LongType(), True),
        StructField("traffic_label", LongType(), True),
        StructField("feature", ArrayType(DoubleType()), True),
    ])

    # Recursive search for json.gz files using Python's glob
    # Spark's local FS reader often fails with ** globbing, so we resolve it here.
    search_pattern = os.path.join(args.source, "**/*.json.gz")
    print(f"Searching for data files in: {args.source}...")
    
    file_list = glob.glob(search_pattern, recursive=True)
    
    if not file_list:
        print(f"Error: No .json.gz files found in {args.source}")
        spark.stop()
        sys.exit(1)
        
    print(f"Found {len(file_list)} files. Loading data...")

    # Spark requires absolute paths (or URIs) sometimes to be safe, but local paths usually work if full.
    # Let's use absolute paths to be safe.
    file_list = [os.path.abspath(p) for p in file_list]

    df = spark.read.schema(schema).json(file_list)
    
    label_col = "traffic_label" if args.type == "traffic" else "app_label"
    mapping_dict = ID_TO_TRAFFIC if args.type == "traffic" else ID_TO_APP
    
    total_count = df.count()
    null_count = df.filter(col(label_col).isNull()).count()
    print(f"Total records: {total_count}")
    print(f"Records with null {label_col}: {null_count}")

    label_counts = df.filter(col(label_col).isNotNull()).groupBy(label_col).count().orderBy(label_col).toPandas()
    
    if not label_counts.empty:
        # Map IDs to names. Handle cases where ID might not be in mapping.
        label_counts["name"] = label_counts[label_col].map(lambda x: mapping_dict.get(x, f"Unknown_ID_{x}"))
        print(f"\n{args.type.capitalize()} Label Distribution:")
        print(label_counts[["name", label_col, "count"]].to_string(index=False))
    else:
        print(f"No non-null {label_col} found.")

    spark.stop()

if __name__ == "__main__":
    main()
