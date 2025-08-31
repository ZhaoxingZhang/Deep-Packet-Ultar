import gzip
import os
from pathlib import Path

def check_gzip_files(directory):
    """Checks for corrupted gzip files in a directory."""
    print(f"Checking for corrupted gzip files in: {directory}")
    corrupted_files = []
    for filepath in Path(directory).rglob('*.json.gz'):
        try:
            with gzip.open(filepath, 'rb') as f:
                while f.read(1024*1024):
                    pass
            print(f"[OK] {filepath}")
        except (gzip.BadGzipFile, EOFError, IOError) as e:
            print(f"[CORRUPTED] {filepath}: {e}")
            corrupted_files.append(filepath)

    if corrupted_files:
        print("\n--- Corrupted files found ---")
        for f in corrupted_files:
            print(f)
    else:
        print("\n--- No corrupted files found ---")

if __name__ == "__main__":
    # The user ran the script with -s processed_data, so we'll check that directory.
    check_gzip_files("processed_data")