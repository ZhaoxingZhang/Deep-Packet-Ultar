#!/bin/bash
set -e

# ==================================================================================
# Generate Experiment Scripts for All Three Datasets
# ==================================================================================

echo "==================================================================="
echo "--- Generating Experiment Scripts for All Datasets ---"
echo "==================================================================="
echo ""

# List of datasets
declare -a DATASETS=(
    "traffic:processed_data/traffic:exp_traffic"
    "traffic_v2:processed_data/traffic_v2:exp_traffic_v2"
    "traffic_v3:processed_data/traffic_v3:exp_traffic_v3"
)

for dataset_info in "${DATASETS[@]}"; do
    IFS=':' read -r name path exp_name <<< "$dataset_info"

    echo ">>> Creating scripts for $name..."

    EXP_DIR="scripts/${exp_name}"
    mkdir -p "$EXP_DIR"

    # Create exe.sh
    cat > "$EXP_DIR/exe.sh" << EOF
#!/bin/bash
set -e

nohup sh -c '
  bash ${EXP_DIR}/run_incremental_resnet_baseline.sh
  bash ${EXP_DIR}/run_incremental_resnet_gee.sh
  bash ${EXP_DIR}/run_open_set_resnet_baseline.sh
  bash ${EXP_DIR}/run_open_set_resnet_gee.sh
' > log/${exp_name}.log 2>&1 &
echo "Experiment started. Check log/${exp_name}.log for progress."
EOF

    chmod +x "$EXP_DIR/exe.sh"

    # Create incremental baseline script
    cat > "$EXP_DIR/run_incremental_resnet_baseline.sh" << EOF
#!/bin/bash
set -e

# ==================================================================================
# Incremental Learning Baseline (ResNet) - ${name} Dataset
# ==================================================================================

# Config
DATA_SOURCE="${path}"
TARGET_DIR="train_test_data/${exp_name}/incremental/baseline_resnet"
MODEL_DIR="model/${exp_name}/incremental"
EVAL_DIR="evaluation_results/${exp_name}/incremental/baseline_resnet"

mkdir -p "\$TARGET_DIR" "\$MODEL_DIR" "\$EVAL_DIR"

MODEL_PATH="\${MODEL_DIR}/resnet_baseline_all.pt"

# 1. Generate Dataset (Imbalanced, All Classes)
echo "--> Step 1: Generating dataset (Imbalanced)..."
if [ ! -f "\${TARGET_DIR}/traffic_classification/train.parquet/_SUCCESS" ]; then
    python -u create_train_test_set.py \\
        -s "\$DATA_SOURCE" \\
        -t "\$TARGET_DIR" \\
        --experiment_type imbalanced \\
        --task-type traffic \\
        --fraction 0.01
else
    echo "    Dataset already exists."
fi

# 2. Train ResNet Baseline
echo "--> Step 2: Training ResNet Baseline..."
if [ ! -f "\${MODEL_PATH}.ckpt" ]; then
    python -u train_resnet.py \\
        --data_path "\${TARGET_DIR}/traffic_classification" \\
        --model_path "\${MODEL_PATH}" \\
        --task traffic \\
        --epochs 50 \\
        --validation_split 0.2
else
    echo "    Model already exists."
fi

# 3. Evaluate
echo "--> Step 3: Evaluating ResNet Baseline..."
python -u evaluation.py \\
    --data_path "\${TARGET_DIR}/traffic_classification/test.parquet" \\
    --model_path "\${MODEL_PATH}.ckpt" \\
    --output_dir "\$EVAL_DIR" \\
    --model_type resnet \\
    --eval-mode standard

echo "Done."
EOF

    chmod +x "$EXP_DIR/run_incremental_resnet_baseline.sh"

    # Create placeholder for other scripts
    touch "$EXP_DIR/run_incremental_resnet_gee.sh"
    touch "$EXP_DIR/run_open_set_resnet_baseline.sh"
    touch "$EXP_DIR/run_open_set_resnet_gee.sh"

    echo "  Created: $EXP_DIR/"
done

echo ""
echo "==================================================================="
echo "--- Experiment Scripts Generated Successfully! ---"
echo "==================================================================="
echo ""
echo "Generated directories:"
echo "  - scripts/exp_traffic/      (Dataset 1)"
echo "  - scripts/exp_traffic_v2/   (Dataset 2)"
echo "  - scripts/exp_traffic_v3/   (Dataset 3)"
echo ""
echo "Note: Only incremental baseline scripts are fully generated."
echo "      You need to adapt other scripts based on each dataset's traffic types."
echo ""
