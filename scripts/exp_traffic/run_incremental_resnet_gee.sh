#!/bin/bash
set -e

# ==================================================================================
# Incremental Learning GEE (ResNet) - Traffic Dataset
# ==================================================================================

# Config
DATA_SOURCE="processed_data/traffic"
BASE_DIR="train_test_data/exp_traffic/incremental/gee_resnet"
MODEL_DIR="model/exp_traffic/incremental/gee_resnet"
EVAL_DIR="evaluation_results/exp_traffic/incremental/gee_resnet"

# Classes 4 (VoIP) and 7 (VPN-Email) are treated as "Minority/Expert" classes for this experiment
MINORITY_CLASSES=(4 7)
MINORITY_CLASSES_ARGS="--minority-classes 4 --minority-classes 7"
MINORITY_CLASSES_ARGS_US="--minority_classes 4 --minority_classes 7"

mkdir -p "$BASE_DIR" "$MODEL_DIR" "$EVAL_DIR"

BASELINE_MODEL_PATH="${MODEL_DIR}/resnet_baseline.pt"
MINORITY_MODEL_PATH="${MODEL_DIR}/resnet_minority_expert.pt"
GATING_MODEL_PATH="${MODEL_DIR}/gating_network_resnet.pt"

# 1. Generate Datasets
echo "--> Step 1: Generating Datasets..."

# A) Main Dataset (Imbalanced, used for Baseline and Gating training)
echo "    Generating Main Dataset..."
if [ ! -f "${BASE_DIR}/main/traffic_classification/train.parquet/_SUCCESS" ]; then
    python -u create_train_test_set.py \
        -s "$DATA_SOURCE" \
        -t "${BASE_DIR}/main" \
        --experiment_type imbalanced \
        --task-type traffic \
        --fraction 1.0
else
    echo "    Main Dataset exists."
fi

# B) Minority Dataset (Only classes 4 and 7)
echo "    Generating Minority Dataset..."
if [ ! -f "${BASE_DIR}/minority/traffic_classification/train.parquet/_SUCCESS" ]; then
    python -u create_train_test_set.py \
        -s "$DATA_SOURCE" \
        -t "${BASE_DIR}/minority" \
        --experiment_type exp8_minority \
        --task-type traffic \
        ${MINORITY_CLASSES_ARGS} \
        --fraction 1.0
else
    echo "    Minority Dataset exists."
fi

# 2. Train Models
echo "--> Step 2: Training Models..."

# A) Baseline ResNet
echo "    Training Baseline ResNet..."
if [ ! -f "${BASELINE_MODEL_PATH}.ckpt" ]; then
    python -u train_resnet.py \
        --data_path "${BASE_DIR}/main/traffic_classification" \
        --model_path "${BASELINE_MODEL_PATH}" \
        --task traffic \
        --epochs 50 \
        --validation_split 0.2
else
    echo "    Baseline Model exists."
fi

# B) Minority Expert ResNet
echo "    Training Minority Expert ResNet..."
if [ ! -f "${MINORITY_MODEL_PATH}.ckpt" ]; then
    python -u train_resnet.py \
        --data_path "${BASE_DIR}/minority/traffic_classification" \
        --model_path "${MINORITY_MODEL_PATH}" \
        --task traffic \
        --epochs 50 \
        --validation_split 0.2
else
    echo "    Minority Model exists."
fi

# C) Gating Network
echo "    Training Gating Network..."
if [ ! -f "${GATING_MODEL_PATH}" ]; then
    python -u train_gating_network.py \
        --train_data_path "${BASE_DIR}/main/traffic_classification/train.parquet" \
        --baseline_model_path "${BASELINE_MODEL_PATH}.ckpt" \
        --minority_model_path "${MINORITY_MODEL_PATH}.ckpt" \
        --baseline_model_type resnet \
        --minority_model_type resnet \
        ${MINORITY_CLASSES_ARGS_US} \
        --output_path "${GATING_MODEL_PATH}" \
        --epochs 100 \
        --lr 0.001
else
    echo "    Gating Model exists."
fi

# 3. Evaluate
echo "--> Step 3: Evaluating GEE..."
python -u evaluation.py \
    --data_path "${BASE_DIR}/main/traffic_classification/test.parquet" \
    --baseline_model_path "${BASELINE_MODEL_PATH}.ckpt" \
    --minority_model_path "${MINORITY_MODEL_PATH}.ckpt" \
    --gating_network_path "${GATING_MODEL_PATH}" \
    --baseline_model_type resnet \
    --minority_model_type resnet \
    --output_dir "${EVAL_DIR}" \
    --eval-mode gating_ensemble \
    ${MINORITY_CLASSES_ARGS_US}

echo "Done."
