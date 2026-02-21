#!/bin/bash
set -e

# ==================================================================================
# Verify Dataset Construction
# ==================================================================================

echo "==================================================================="
echo "--- Dataset Verification Report ---"
echo "==================================================================="
echo ""

# Function to count files in a directory
count_files() {
    local dir=$1
    if [ -d "$dir" ]; then
        find "$dir" -name "*.json.gz" | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# Function to check dataset
check_dataset() {
    local dataset_name=$1
    local dataset_path=$2

    echo "==================================================================="
    echo "$dataset_name: $dataset_path"
    echo "==================================================================="

    if [ ! -d "$dataset_path" ]; then
        echo "  ❌ Dataset directory does not exist!"
        echo ""
        return
    fi

    echo ""
    echo "Subdirectories and file counts:"
    echo "-----------------------------------"

    total_files=0
    for subdir in "$dataset_path"/*; do
        if [ -d "$subdir" ]; then
            subdir_name=$(basename "$subdir")
            file_count=$(count_files "$subdir")
            total_files=$((total_files + file_count))
            printf "  %-25s %6d files\n" "$subdir_name:" "$file_count"
        fi
    done

    echo "-----------------------------------"
    printf "  %-25s %6d files\n" "TOTAL:" "$total_files"
    echo ""
}

# Check all datasets
check_dataset "Dataset 1 (traffic)" "processed_data/traffic"
check_dataset "Dataset 2 (traffic_v2)" "processed_data/traffic_v2"
check_dataset "Dataset 3 (traffic_v3)" "processed_data/traffic_v3"
check_dataset "Original VPN" "processed_data/vpn"

echo "==================================================================="
echo "--- Verification Complete ---"
echo "==================================================================="
