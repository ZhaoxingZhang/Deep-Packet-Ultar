#!/bin/bash
set -e

echo "================================================================="
echo "--- Running CNN GEE Incremental Learning Evaluation (Optimized) ---"
echo "================================================================="

# --- Configuration ---
MINORITY_CLASSES=(5 7)
MINORITY_CLASSES_ARGS_HYPHEN=$(printf -- "--minority-classes %s " "${MINORITY_CLASSES[@]}")
MINORITY_CLASSES_ARGS_UNDERSCORE=$(printf -- "--minority_classes %s " "${MINORITY_CLASSES[@]}")

# Paths
BASE_DATA_DIR="train_test_data/from_featurize_vpn/traffic_classification"
TRAIN_PATH="${BASE_DATA_DIR}/train.parquet"
TEST_PATH="${BASE_DATA_DIR}/test.parquet"

MINORITY_EXPERT_DATA_DIR="train_test_data/traffic_minority_expert/traffic_classification"

BASELINE_MODEL_NAME="cnn_vpn_baseline"
MINORITY_MODEL_NAME="cnn_minority_expert"
GATING_MODEL_NAME="cnn_gating_network_optimized"

BASELINE_MODEL_PATH="model/${BASELINE_MODEL_NAME}.model"
MINORITY_MODEL_PATH="model/${MINORITY_MODEL_NAME}.model"
GATING_MODEL_PATH="model/${GATING_MODEL_NAME}.pth"

FINAL_BASELINE_PATH="${BASELINE_MODEL_PATH}.ckpt"
FINAL_MINORITY_PATH="${MINORITY_MODEL_PATH}.ckpt"

EVAL_DIR="evaluation_results/cnn_gee_incremental_optimized"
mkdir -p "${EVAL_DIR}"

# --- 1. Train Baseline CNN Model ---
echo "--> Step 1: Training baseline CNN model..."
if [ ! -f "${FINAL_BASELINE_PATH}" ]; then
    python -u train_cnn.py \
      --data_path "${TRAIN_PATH}" \
      --model_path "${BASELINE_MODEL_PATH}" \
      --task traffic
else
    echo "    - Baseline CNN model already exists. Skipping training."
fi

# --- 2. Train Minority Expert CNN Model ---
echo "--> Step 2: Training minority expert CNN model..."
if [ ! -f "${FINAL_MINORITY_PATH}" ]; then
    python -u train_cnn.py \
      --data_path "${MINORITY_EXPERT_DATA_DIR}" \
      --model_path "${MINORITY_MODEL_PATH}" \
      --task traffic
else
    echo "    - Minority expert CNN model already exists. Skipping training."
fi

# --- 3. Train Gating Network (Using train_on_val strategy) ---
echo "--> Step 3: Training Optimized Gating Network..."
# --train_on_val 让门控网络学习基准模型在验证集（未见过的数据）上的错误模式
python -u train_gating_network.py \
    --train_data_path "${TRAIN_PATH}" \
    --baseline_model_path "${FINAL_BASELINE_PATH}" \
    --minority_model_path "${FINAL_MINORITY_PATH}" \
    --baseline_model_type cnn \
    --minority_model_type cnn \
    ${MINORITY_CLASSES_ARGS_UNDERSCORE} \
    --output_path "${GATING_MODEL_PATH}" \
    --epochs 300 \
    --lr 0.0003 \
    --train_on_val \
    --validation_split 0.1

# --- 4. Evaluate GEE (CNN) Model ---
echo "--> Step 4: Evaluating full GEE (CNN) architecture..."
python -u evaluation.py \
  --data_path "${TEST_PATH}" \
  --output_dir "${EVAL_DIR}" \
  --eval-mode gating_ensemble \
  --baseline_model_path "${FINAL_BASELINE_PATH}" \
  --minority_model_path "${FINAL_MINORITY_PATH}" \
  --baseline_model_type cnn \
  --minority_model_type cnn \
  --gating_network_path "${GATING_MODEL_PATH}" \
  ${MINORITY_CLASSES_ARGS_UNDERSCORE}

echo "================================================================="
echo "--- CNN GEE Incremental Learning Evaluation Complete ---"
echo "--- Results are in ${EVAL_DIR} ---"
echo "================================================================="