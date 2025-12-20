#!/bin/bash
set -e

echo "================================================================="
echo "--- Running CNN Baseline Incremental Learning Evaluation ---"
echo "================================================================="

# --- Configuration ---
DATA_DIR="train_test_data/from_featurize_vpn/traffic_classification"
TRAIN_PATH="${DATA_DIR}/train.parquet"
TEST_PATH="${DATA_DIR}/test.parquet"
MODEL_NAME="cnn_vpn_baseline"
MODEL_PATH="model/${MODEL_NAME}.model"
EVAL_DIR="evaluation_results/${MODEL_NAME}"
FINAL_MODEL_PATH="${MODEL_PATH}.ckpt"

# --- 1. Train Baseline CNN Model ---
echo "--> Step 1: Training baseline CNN model..."
if [ ! -f "${FINAL_MODEL_PATH}" ]; then
    python -u train_cnn.py \
      --data_path "${TRAIN_PATH}" \
      --model_path "${MODEL_PATH}" \
      --task traffic
else
    echo "    - Baseline CNN model already exists. Skipping training."
fi


# --- 2. Evaluate Baseline CNN Model ---
echo "--> Step 2: Evaluating baseline CNN model..."
python -u evaluation.py \
  --model_path "${FINAL_MODEL_PATH}" \
  --data_path "${TEST_PATH}" \
  --output_dir "${EVAL_DIR}" \
  --model_type cnn \
  --eval-mode standard

echo "================================================================="
echo "--- CNN Baseline Incremental Learning Evaluation Complete ---"
echo "================================================================="
