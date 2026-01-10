#!/bin/bash
set -e

# ==================================================================================
# Incremental Learning Baseline (ResNet) - Traffic Dataset
# ==================================================================================

# Config
DATA_SOURCE="processed_data/traffic"
TARGET_DIR="train_test_data/exp_traffic/incremental/baseline_resnet"
MODEL_DIR="model/exp_traffic/incremental"
EVAL_DIR="evaluation_results/exp_traffic/incremental/baseline_resnet"

mkdir -p "$TARGET_DIR" "$MODEL_DIR" "$EVAL_DIR"

MODEL_PATH="${MODEL_DIR}/resnet_baseline_all.pt"

# 1. Generate Dataset (Imbalanced, All Classes)
echo "--> Step 1: Generating dataset (Imbalanced)..."
if [ ! -f "${TARGET_DIR}/traffic_classification/train.parquet/_SUCCESS" ]; then
    python -u create_train_test_set.py \
        -s "$DATA_SOURCE" \
        -t "$TARGET_DIR" \
        --experiment_type imbalanced \
        --task-type traffic \
        --fraction 0.005
else
    echo "    Dataset already exists."
fi

# 2. Train ResNet Baseline
echo "--> Step 2: Training ResNet Baseline..."
if [ ! -f "${MODEL_PATH}.ckpt" ]; then
    python -u train_resnet.py \
        --data_path "${TARGET_DIR}/traffic_classification" \
        --model_path "${MODEL_PATH}" \
        --task traffic \
        --epochs 50 \
        --validation_split 0.2
else
    echo "    Model already exists."
fi

# 3. Evaluate
echo "--> Step 3: Evaluating ResNet Baseline..."
python -u evaluation.py \
    --data_path "${TARGET_DIR}/traffic_classification/test.parquet" \
    --model_path "${MODEL_PATH}.ckpt" \
    --output_dir "${EVAL_DIR}" \
    --model_type resnet \
    --eval-mode standard

echo "Done."
