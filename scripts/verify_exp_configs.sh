#!/bin/bash
set -e

# ==================================================================================
# Verify Experiment Scripts Configuration
# ==================================================================================

echo "==================================================================="
echo "--- Experiment Scripts Configuration Verification ---"
echo "==================================================================="
echo ""

# Function to extract config from script
extract_config() {
    local script_path=$1
    local pattern=$2

    grep -E "^$pattern" "$script_path" | head -1 | sed 's/^[^=]*=//' | tr -d ' "'
}

# Function to verify script
verify_script() {
    local name=$1
    local script_path=$2

    if [ ! -f "$script_path" ]; then
        echo "  ❌ $script_path not found"
        return 1
    fi

    if [ ! -x "$script_path" ]; then
        echo "  ⚠️  $script_path not executable"
        return 1
    fi

    echo "  ✅ $name"
    return 0
}

echo "Dataset 1: exp_traffic"
echo "===================="
echo "Traffic Types: 0 2 4 6 7 8 9 10 (8 types)"
echo "Minority Classes: 4 (VoIP), 7 (VPN:Email)"
echo ""
echo "Scripts:"
verify_script "Incremental Baseline" "scripts/exp_traffic/run_incremental_resnet_baseline.sh"
verify_script "Incremental GEE" "scripts/exp_traffic/run_incremental_resnet_gee.sh"
verify_script "Open Set Baseline" "scripts/exp_traffic/run_open_set_resnet_baseline.sh"
verify_script "Open Set GEE" "scripts/exp_traffic/run_open_set_resnet_gee.sh"
echo ""

echo "Dataset 2: exp_traffic_v2"
echo "========================"
echo "Traffic Types: 0 2 4 5 6 8 (6 types)"
echo "Minority Classes: 5 (VPN:Chat), 8 (VPN:Streaming)"
echo ""
echo "Scripts:"
verify_script "Incremental Baseline" "scripts/exp_traffic_v2/run_incremental_resnet_baseline.sh"
verify_script "Incremental GEE" "scripts/exp_traffic_v2/run_incremental_resnet_gee.sh"
verify_script "Open Set Baseline" "scripts/exp_traffic_v2/run_open_set_resnet_baseline.sh"
verify_script "Open Set GEE" "scripts/exp_traffic_v2/run_open_set_resnet_gee.sh"

# Verify configs
gee_script="scripts/exp_traffic_v2/run_incremental_resnet_gee.sh"
if [ -f "$gee_script" ]; then
    minority=$(grep "^MINORITY_CLASSES=" "$gee_script" | head -1 | sed 's/^[^=]*=//' | tr -d ' "')
    if [ "$minority" = "58" ] || [ "$minority" = "5 8" ]; then
        echo "  ✅ Minority classes configured correctly: 5, 8"
    else
        echo "  ⚠️  Minority classes: $minority (expected: 5 8)"
    fi
fi

baseline_script="scripts/exp_traffic_v2/run_open_set_resnet_baseline.sh"
if [ -f "$baseline_script" ]; then
    all_classes=$(grep "^ALL_CLASSES=" "$baseline_script" | head -1 | sed 's/^[^=]*=//' | tr -d ' "')
    if [ "$all_classes" = "024568" ] || [ "$all_classes" = "0 2 4 5 6 8" ]; then
        echo "  ✅ All classes configured correctly: 0 2 4 5 6 8"
    else
        echo "  ⚠️  All classes: $all_classes (expected: 0 2 4 5 6 8)"
    fi
fi
echo ""

echo "Dataset 3: exp_traffic_v3"
echo "========================"
echo "Traffic Types: 0 2 4 7 9 10 (6 types)"
echo "Minority Classes: 7 (VPN:Email), 9 (VPN:Torrent)"
echo ""
echo "Scripts:"
verify_script "Incremental Baseline" "scripts/exp_traffic_v3/run_incremental_resnet_baseline.sh"
verify_script "Incremental GEE" "scripts/exp_traffic_v3/run_incremental_resnet_gee.sh"
verify_script "Open Set Baseline" "scripts/exp_traffic_v3/run_open_set_resnet_baseline.sh"
verify_script "Open Set GEE" "scripts/exp_traffic_v3/run_open_set_resnet_gee.sh"

# Verify configs
gee_script="scripts/exp_traffic_v3/run_incremental_resnet_gee.sh"
if [ -f "$gee_script" ]; then
    minority=$(grep "^MINORITY_CLASSES=" "$gee_script" | head -1 | sed 's/^[^=]*=//' | tr -d ' "')
    if [ "$minority" = "79" ] || [ "$minority" = "7 9" ]; then
        echo "  ✅ Minority classes configured correctly: 7, 9"
    else
        echo "  ⚠️  Minority classes: $minority (expected: 7 9)"
    fi
fi

baseline_script="scripts/exp_traffic_v3/run_open_set_resnet_baseline.sh"
if [ -f "$baseline_script" ]; then
    all_classes=$(grep "^ALL_CLASSES=" "$baseline_script" | head -1 | sed 's/^[^=]*=//' | tr -d ' "')
    if [ "$all_classes" = "0247910" ] || [ "$all_classes" = "0 2 4 7 9 10" ]; then
        echo "  ✅ All classes configured correctly: 0 2 4 7 9 10"
    else
        echo "  ⚠️  All classes: $all_classes (expected: 0 2 4 7 9 10)"
    fi
fi
echo ""

echo "==================================================================="
echo "--- Dataset Availability ---"
echo "==================================================================="
echo ""

for dataset in "processed_data/traffic" "processed_data/traffic_v2" "processed_data/traffic_v3"; do
    if [ -d "$dataset" ]; then
        count=$(find "$dataset" -name "*.json.gz" | wc -l | tr -d ' ')
        echo "✅ $dataset: $count files"
    else
        echo "❌ $dataset: not found"
    fi
done

echo ""
echo "==================================================================="
echo "--- Verification Complete ---"
echo "==================================================================="
echo ""
echo "Next Steps:"
echo "  1. Run Dataset 1:  bash scripts/exp_traffic/exe.sh"
echo "  2. Run Dataset 2:  bash scripts/exp_traffic_v2/exe.sh"
echo "  3. Run Dataset 3:  bash scripts/exp_traffic_v3/exe.sh"
echo ""
echo "Monitor logs:"
echo "  tail -f log/exp_traffic.log"
echo "  tail -f log/exp_traffic_v2.log"
echo "  tail -f log/exp_traffic_v3.log"
echo ""
