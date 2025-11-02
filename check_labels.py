
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, ArrayType, LongType, DoubleType
from utils import ID_TO_TRAFFIC

source_data_dir = "/Users/weisman/Documents/repo_root/graduate/Deep-Packet-Ultar/processed_data/vpn"

def main():
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

    df = spark.read.schema(schema).json(f"{source_data_dir}/*.json.gz")
    traffic_label_counts = df.filter(col("traffic_label").isNotNull()).groupBy("traffic_label").count().orderBy("traffic_label").toPandas()

    traffic_label_counts["traffic_name"] = traffic_label_counts["traffic_label"].map(ID_TO_TRAFFIC)
    print("Traffic Label Distribution:")
    print(traffic_label_counts[["traffic_name", "traffic_label", "count"]].to_string(index=False))

    spark.stop()

if __name__ == "__main__":
    main()
