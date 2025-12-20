import argparse
import gzip
import json
import os
from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_fgnet_dataset(data_dir: str, output_dir: str, fixed_length: int):
    """
    Processes the fgnet dataset, converting flow data into fixed-length feature vectors.

    Args:
        data_dir (str): The root directory of the fgnet dataset (e.g., 'fgnet/dataset').
        output_dir (str): The directory to save the processed .json.gz files.
        fixed_length (int): The fixed length to pad/truncate the packet sequences to.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    app_name_to_label = {}
    label_counter = 0

    # Find all json files and group them by app name
    files_by_app = {}
    for json_file in tqdm(sorted(data_path.rglob('*.json')), desc="Finding files"):
        try:
            app_name = json_file.parts[-3]
            if app_name not in files_by_app:
                files_by_app[app_name] = []
            files_by_app[app_name].append(json_file)
        except IndexError:
            logging.warning(f"Could not determine app name for file: {json_file}")
            continue

    for app_name, files in tqdm(files_by_app.items(), desc="Processing apps"):
        if app_name not in app_name_to_label:
            app_name_to_label[app_name] = label_counter
            label_counter += 1
        label = app_name_to_label[app_name]

        output_file_path = output_path / f"{app_name}.json.gz"

        with gzip.open(output_file_path, 'wt', encoding='utf-8') as gz_file:
            for json_file in tqdm(files, desc=f"Processing {app_name}", leave=False):
                with open(json_file, 'r') as f:
                    try:
                        flows = json.load(f)
                        for flow in flows:
                            packet_lengths = flow.get("packet_length", [])
                            
                            # Pad or truncate the sequence
                            if len(packet_lengths) > fixed_length:
                                features = packet_lengths[:fixed_length]
                            else:
                                features = packet_lengths + [0] * (fixed_length - len(packet_lengths))
                            
                            # Normalize features to be in [-1, 1] range approximately
                            # A simple approach for now, assuming max packet size is around 1500
                            features = np.array(features) / 1500.0

                            processed_record = {
                                "label": label,
                                "features": features.tolist()
                            }
                            gz_file.write(json.dumps(processed_record) + '\n')

                    except json.JSONDecodeError:
                        logging.warning(f"Skipping corrupted JSON file: {json_file}")
                    except Exception as e:
                        logging.error(f"An error occurred processing {json_file}: {e}")

    # Save the label mapping
    label_map_path = output_path / 'label_map.json'
    with open(label_map_path, 'w') as f:
        json.dump(app_name_to_label, f, indent=4)
    
    logging.info(f"Processing complete. Label mapping saved to {label_map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the FG-Net dataset.")
    parser.add_argument("--data-dir", type=str, required=True, help="Root directory of the fgnet dataset (e.g., 'fgnet/dataset').")
    parser.add_argument("--output-dir", type=str, default="processed_data/fgnet", help="Directory to save the processed .json.gz files.")
    parser.add_argument("--fixed-length", type=int, default=74, help="Fixed length for packet sequences.")
    
    args = parser.parse_args()

    process_fgnet_dataset(args.data_dir, args.output_dir, args.fixed_length)
