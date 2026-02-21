#!/usr/bin/env python3
"""
Analyze traffic types in each dataset
"""
import os
import json
from pathlib import Path
from collections import defaultdict

# Traffic type mapping (from utils.py)
ID_TO_TRAFFIC = {
    0: "Chat",
    1: "Email",
    2: "File Transfer",
    3: "Streaming",
    4: "VoIP",
    5: "VPN: Chat",
    6: "VPN: File Transfer",
    7: "VPN: Email",
    8: "VPN: Streaming",
    9: "VPN: Torrent",
    10: "VPN: Voip",
}

def analyze_dataset(dataset_path, dataset_name):
    """Analyze traffic types in a dataset"""
    print(f"\n{'='*80}")
    print(f"{dataset_name}: {dataset_path}")
    print(f"{'='*80}")

    if not os.path.exists(dataset_path):
        print("  ❌ Dataset does not exist!")
        return set()

    traffic_types = set()
    subdir_stats = defaultdict(lambda: {"count": 0, "types": set()})

    # Walk through all subdirectories
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.json.gz'):
                file_path = os.path.join(root, file)
                subdir = os.path.relpath(root, dataset_path)

                try:
                    # Read first line to get traffic_label
                    import gzip
                    with gzip.open(file_path, 'rt') as f:
                        line = f.readline()
                        data = json.loads(line)
                        traffic_label = data.get('traffic_label')
                        if traffic_label is not None:
                            traffic_types.add(traffic_label)
                            subdir_stats[subdir]["count"] += 1
                            subdir_stats[subdir]["types"].add(traffic_label)
                except Exception as e:
                    pass

    # Print statistics
    if traffic_types:
        print(f"\n📊 Traffic Types Found ({len(traffic_types)}):")
        print("-" * 80)
        for t in sorted(traffic_types):
            type_name = ID_TO_TRAFFIC.get(t, f"Unknown({t})")
            print(f"  Type {t:2d}: {type_name}")

        print(f"\n📁 Subdirectory Statistics:")
        print("-" * 80)
        for subdir in sorted(subdir_stats.keys()):
            stats = subdir_stats[subdir]
            types_str = ", ".join([str(t) for t in sorted(stats["types"])])
            print(f"  {subdir:30s} {stats['count']:6d} files  [Types: {types_str}]")

        print(f"\n📈 Total Files: {sum(s['count'] for s in subdir_stats.values())}")
    else:
        print("  ⚠️  No traffic data found!")

    return traffic_types

def main():
    print("="*80)
    print("DATASET TRAFFIC TYPE ANALYSIS")
    print("="*80)

    datasets = [
        ("Dataset 1 (traffic)", "processed_data/traffic"),
        ("Dataset 2 (traffic_v2)", "processed_data/traffic_v2"),
        ("Dataset 3 (traffic_v3)", "processed_data/traffic_v3"),
        ("Original VPN", "processed_data/vpn"),
    ]

    all_types = {}
    for name, path in datasets:
        types = analyze_dataset(path, name)
        all_types[name] = types

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print()

    for name, types in all_types.items():
        if types:
            types_str = ", ".join([str(t) for t in sorted(types)])
            print(f"{name:30s} Types: {types_str}")
        else:
            print(f"{name:30s} No data")

    print()
    print("="*80)
    print("✅ Analysis Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
