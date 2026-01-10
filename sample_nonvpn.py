import os
import sys
import glob
import json
import gzip
import random
from collections import defaultdict
from pathlib import Path
import click
from joblib import Parallel, delayed
from utils import PREFIX_TO_APP_ID

def process_class(label, files, output_dir, target_count):
    reservoir = []
    seen_count = 0
    
    # Reservoir Sampling
    for file_path in files:
        try:
            with gzip.open(file_path, 'rt') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if row.get('app_label') != label:
                            continue
                        
                        # Add to reservoir
                        if seen_count < target_count:
                            reservoir.append(row)
                        else:
                            # Replace existing item with decreasing probability
                            j = random.randint(0, seen_count)
                            if j < target_count:
                                reservoir[j] = row
                        
                        seen_count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

    print(f"Class {label}: Scanned {seen_count} samples. Retained {len(reservoir)}.")
    
    # Write back to output directory
    output_filename = f"class_{label}_sampled.json.gz"
    output_path = output_dir / output_filename
    
    with gzip.open(output_path, 'wt') as f_out:
        for row in reservoir:
            f_out.write(json.dumps(row) + '\n')
            
    print(f"Class {label}: Saved to {output_path}")

@click.command()
@click.option('--source', '-s', required=True, help='Source directory containing .json.gz files')
@click.option('--target', '-t', required=True, help='Target directory to save sampled files')
@click.option('--count', '-c', default=10000, help='Number of samples per class')
@click.option('--njob', '-n', default=4, help='Number of parallel jobs')
def main(source, target, count, njob):
    source_path = Path(source)
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print("Scanning files...")
    files = list(source_path.glob("*.json.gz"))
    if not files:
        files = list(source_path.glob("**/*.json.gz"))
        
    print(f"Found {len(files)} files.")
    
    files_by_label = defaultdict(list)
    
    def get_label_from_filename(file_path):
        # Filename format: prefix.pcap.transformed_part_XXXX.json.gz
        # Example: ftps_up_2a.pcap.transformed_part_0006.json.gz
        try:
            filename = file_path.name
            # Extract prefix: everything before .pcap
            if ".pcap" in filename:
                prefix = filename.split(".pcap")[0]
                label = PREFIX_TO_APP_ID.get(prefix)
                if label is not None:
                    return label, file_path
            
            # Fallback or different format
            # Try splitting by underscore if needed, but PREFIX_TO_APP_ID keys are quite specific
            pass
        except Exception:
            pass
            
        # Fallback to reading file if filename parsing fails
        try:
            with gzip.open(file_path, 'rt') as f:
                first_line = f.readline()
                if first_line:
                    row = json.loads(first_line)
                    return row.get('app_label'), file_path
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return None, None

    # Use parallel processing to speed up file scanning
    results = Parallel(n_jobs=njob)(delayed(get_label_from_filename)(f) for f in files)
    
    for label, f_path in results:
        if label is not None and f_path is not None:
            files_by_label[label].append(f_path)
            
    print(f"Identified {len(files_by_label)} classes.")
    
    Parallel(n_jobs=njob)(
        delayed(process_class)(label, file_list, target_path, count)
        for label, file_list in files_by_label.items()
    )
    
    print("Done.")

if __name__ == '__main__':
    main()
