#!/usr/bin/env python3
"""
Quick validation script to verify models while test data is downloading
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, ProgressPrinter

def validate_models():
    """Validate that all model files exist and can be loaded"""
    pp = ProgressPrinter()
    pp.print_section("Validating GEE Models")

    config = load_config("config.yaml")
    exp_config = config['experiments']['traffic']['incremental']

    models_to_check = {
        'Baseline': exp_config['baseline_model'],
        'Expert': exp_config['expert_model'],
        'Gating Network': exp_config['gating_network']
    }

    all_valid = True
    for name, path in models_to_check.items():
        pp.print_info(f"Checking {name}...")

        if not os.path.exists(path):
            pp.print_error(f"  ✗ Not found: {path}")
            all_valid = False
            continue

        try:
            # Try to load the model
            if name == 'Gating Network':
                # For gating network, just check state dict
                state_dict = torch.load(path, map_location='cpu')
                pp.print_success(f"  ✓ Found (size: {os.path.getsize(path)/1024/1024:.1f} MB)")
            else:
                # For baseline/expert, try to load
                from utils import load_resnet_model
                model = load_resnet_model(path, device='cpu')
                pp.print_success(f"  ✓ Loadable (size: {os.path.getsize(path)/1024/1024:.1f} MB)")
                del model  # Free memory
        except Exception as e:
            pp.print_error(f"  ✗ Error loading: {str(e)[:50]}")
            all_valid = False

    if all_valid:
        pp.print_section("✓ All models validated successfully!")
        pp.print_info("Ready for feature extraction once test data downloads.")
        pp.print_info("\nMonitor download progress:")
        pp.print_info("  tail -f analysis_scripts/download.log")
        return True
    else:
        pp.print_section("✗ Some models are missing or corrupted")
        return False

def check_download_progress():
    """Check if test data has been downloaded"""
    pp = ProgressPrinter()

    # Load config to get actual test data paths
    config = load_config("config.yaml")

    # Check for incremental test data from config
    test_paths = [
        config['experiments']['traffic']['incremental']['test_data'],
    ]

    pp.print_section("Checking Test Data Download Progress")

    downloaded = []
    for path in test_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            pp.print_success(f"  ✓ {path} ({size_mb:.1f} MB)")
            downloaded.append(path)
        else:
            pp.print_info(f"  ✗ {path} (not yet downloaded)")

    if downloaded:
        pp.print_info(f"\n{len(downloaded)}/{len(test_paths)} datasets ready")
    else:
        pp.print_info("\nNo datasets downloaded yet. Still downloading...")
        pp.print_info("Check download progress: tail -f analysis_scripts/download.log")

    return len(downloaded) > 0

def generate_sample_analysis():
    """Generate a sample analysis with synthetic data to test visualization"""
    pp = ProgressPrinter()
    pp.print_section("Generating Sample Analysis (Testing)")

    # Create sample output directory structure
    os.makedirs("analysis_results/weights", exist_ok=True)
    os.makedirs("analysis_results/features", exist_ok=True)

    # Generate sample weight analysis
    sample_weight_analysis = {
        "weight_matrix_shape": [64, 12],
        "baseline": {
            "mean_abs": 0.5,
            "std_abs": 0.3,
            "per_dim_mean_abs": [0.4, 0.5, 0.6, 0.4, 0.3, 0.5,
                              0.6, 0.7, 0.5, 0.4, 0.6, 0.5]
        },
        "expert": {
            "mean_abs": 1.2,
            "std_abs": 0.8,
            "per_dim_mean_abs": [0.8, 0.9, 1.0, 0.8, 2.3, 1.5,  # Class 5 (index 4) amplified
                              0.7, 0.8, 0.9, 0.7, 2.5, 1.2]  # Class 7 (index 6) amplified
        },
        "minority_classes": {
            "5": {
                "baseline_weight": 0.4,
                "expert_weight": 0.92,
                "weight_ratio": 2.3
            },
            "7": {
                "baseline_weight": 0.3,
                "expert_weight": 0.75,
                "weight_ratio": 2.5
            }
        }
    }

    import json
    with open("analysis_results/weights/gating_network_weights.json", 'w') as f:
        json.dump(sample_weight_analysis, f, indent=2)

    pp.print_success("Sample weight analysis generated")
    pp.print_info("This tests the visualization pipeline before real data is ready")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate models and check progress")
    parser.add_argument("--generate-sample", action="store_true",
                       help="Generate sample analysis for testing")

    args = parser.parse_args()

    # Change to script directory first
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Always validate models first
    models_valid = validate_models()

    # Check download progress
    data_ready = check_download_progress()

    # If requested and models valid, generate sample
    if args.generate_sample and models_valid:
        generate_sample_analysis()

    # Summary
    pp = ProgressPrinter()
    pp.print_section("Summary")

    if models_valid and data_ready:
        pp.print_success("✓ Models validated")
        pp.print_success("✓ Test data downloaded")
        pp.print_info("\n🚀 Ready to run full feature extraction:")
        pp.print_info("  cd analysis_scripts")
        pp.print_info("  python extract_features.py --config config.yaml --experiment traffic --scenario incremental --device cpu")
    elif models_valid:
        pp.print_success("✓ Models validated")
        pp.print_info("⏳ Waiting for test data download...")
        pp.print_info("  Monitor: tail -f analysis_scripts/download.log")
    else:
        pp.print_error("✗ Models need attention")
