#!/bin/bash

# ==================================================================================
#
# Gated Expert Ensemble (GEE) Open-Set Recognition Evaluation Script
#
# ==================================================================================
#
# Description:
# This script automates the 6-fold cross-validation process for evaluating the
# open-set recognition capabilities of the GEE model.
#
# For each fold, it performs the following steps:
# 1.  **Data Generation**:
#     - Creates a main dataset where one class is held out as "unknown".
#     - Creates a specialized dataset for the minority expert, also excluding the
#       "unknown" class.
# 2.  **Model Training**:
#     - Trains the baseline (majority) expert on the main training set.
#     - Trains the minority expert on its specialized training set.
#     - Trains the gating network using the outputs from the two experts.
# 3.  **Evaluation**:
#     - Runs the final evaluation using the trained GEE model against a test
#       set that includes the "unknown" class.
#     - Calculates both classification metrics (Accuracy, F1-score) and
#       open-set metrics (AUROC, FPR@TPR95).
#
# ==================================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
ALL_CLASSES=(5 6 7 8 9 10)
MINORITY_CLASSES_BASE=(5 7)
CLASSES_TO_EXCLUDE=(5 6 7 8 9 10) # Classes to be used as "unknown" in each fold

SOURCE_DATA_DIR="/home/featurize/data/processed_data/vpn"
BASE_TRAIN_TEST_DIR="train_test_data/open_set_gee"
BASE_MODEL_DIR="model/open_set_gee"
BASE_EVAL_DIR="evaluation_results/open_set_gee"

# --- Main Loop for 6-Fold Cross-Validation ---

for EXCLUDED_CLASS in "${CLASSES_TO_EXCLUDE[@]}"; do
    echo "================================================================="
    echo "--- Starting GEE Open-Set Fold: Excluded Class ${EXCLUDED_CLASS} as Unknown ---"
    echo "================================================================="

    # --- Define Fold-Specific Variables ---
    KNOWN_CLASSES_ARGS=""
    LABEL_MAP_STRING=""
    NEW_LABEL_IDX=0
    for c in "${ALL_CLASSES[@]}"; do
        if [ "$c" != "$EXCLUDED_CLASS" ]; then
            KNOWN_CLASSES_ARGS="$KNOWN_CLASSES_ARGS --known-classes $c"
            if [ -z "$LABEL_MAP_STRING" ]; then
                LABEL_MAP_STRING="${NEW_LABEL_IDX}:${c}"
            else
                LABEL_MAP_STRING="${LABEL_MAP_STRING},${NEW_LABEL_IDX}:${c}"
            fi
            NEW_LABEL_IDX=$((NEW_LABEL_IDX + 1))
        fi
    done

    MINORITY_CLASSES_FOLD=()
    for mc in "${MINORITY_CLASSES_BASE[@]}"; do
        if [ "$mc" != "$EXCLUDED_CLASS" ]; then
            MINORITY_CLASSES_FOLD+=("$mc")
        fi
    done
    MINORITY_CLASSES_FOLD_STR_ARGS=$(printf -- "--minority-classes %s " "${MINORITY_CLASSES_FOLD[@]}")


    FOLD_DATA_DIR="${BASE_TRAIN_TEST_DIR}/exp_exclude_${EXCLUDED_CLASS}"
    FOLD_MODEL_DIR="${BASE_MODEL_DIR}/exclude_${EXCLUDED_CLASS}"
    FOLD_EVAL_DIR="${BASE_EVAL_DIR}/exclude_${EXCLUDED_CLASS}"

    # Define base names for models (without .ckpt, used for passing to train_resnet.py)
    BASE_NAME_BASELINE="${FOLD_MODEL_DIR}/baseline.pt"
    BASE_NAME_MINORITY="${FOLD_MODEL_DIR}/minority_expert.pt"

    # Define actual model paths (with .ckpt, for checking existence and downstream use)
    FINAL_BASELINE_MODEL_PATH="${BASE_NAME_BASELINE}.ckpt"
    FINAL_MINORITY_EXPERT_PATH="${BASE_NAME_MINORITY}.ckpt"
    GATING_NETWORK_PATH="${FOLD_MODEL_DIR}/gating_network.pt"


    mkdir -p "${FOLD_DATA_DIR}" "${FOLD_MODEL_DIR}" "${FOLD_EVAL_DIR}"

    # --- 1. Data Generation ---
    echo "--> Step 1: Generating datasets for Fold ${EXCLUDED_CLASS}..."

    # a) Main dataset for baseline expert and gating network training
    MAIN_DATA_DIR="${FOLD_DATA_DIR}/main/traffic_classification"
    MAIN_TRAIN_PATH="${MAIN_DATA_DIR}/train.parquet"
    MAIN_TEST_PATH="${MAIN_DATA_DIR}/test.parquet"

    if [ -f "${MAIN_TRAIN_PATH}/_SUCCESS" ] && \
       [ -n "$(find "${MAIN_TRAIN_PATH}" -maxdepth 1 -name 'part-*.parquet' -print -quit)" ] && \
       [ -f "${MAIN_TEST_PATH}/_SUCCESS" ] && \
       [ -n "$(find "${MAIN_TEST_PATH}" -maxdepth 1 -name 'part-*.parquet' -print -quit)" ]; then
        echo "    - Main dataset already exists and is valid. Skipping generation."
    else
        echo "    - Generating main dataset..."
        python -u create_train_test_set.py \
            --source "${SOURCE_DATA_DIR}" \
            --target "${FOLD_DATA_DIR}/main" \
            --experiment_type open_set_hold_out \
            --exclude-classes "${EXCLUDED_CLASS}" \
            --task-type traffic \
            --fraction 0.01 \
            --batch_size 50
    fi

    # b) Minority expert dataset
    MINORITY_DATA_DIR="${FOLD_DATA_DIR}/minority/traffic_classification"
    MINORITY_TRAIN_PATH="${MINORITY_DATA_DIR}/train.parquet"
    MINORITY_TEST_PATH="${MINORITY_DATA_DIR}/test.parquet"

    if [ -f "${MINORITY_TRAIN_PATH}/_SUCCESS" ] && \
       [ -n "$(find "${MINORITY_TRAIN_PATH}" -maxdepth 1 -name 'part-*.parquet' -print -quit)" ] && \
       [ -f "${MINORITY_TEST_PATH}/_SUCCESS" ] && \
       [ -n "$(find "${MINORITY_TEST_PATH}" -maxdepth 1 -name 'part-*.parquet' -print -quit)" ]; then
        echo "    - Minority expert dataset already exists and is valid. Skipping generation."
    else
        echo "    - Generating minority expert dataset..."
        python -u create_train_test_set.py \
            --source "${SOURCE_DATA_DIR}" \
            --target "${FOLD_DATA_DIR}/minority" \
            --experiment_type open_set_hold_out \
            --exclude-classes "${EXCLUDED_CLASS}" \
            ${MINORITY_CLASSES_FOLD_STR_ARGS} \
            --task-type traffic \
            --fraction 0.01 \
            --batch_size 50
    fi

    # --- 2. Model Training ---
    echo "--> Step 2: Training models for Fold ${EXCLUDED_CLASS}..."

    # a) Train Baseline (Majority) Expert
    if [ -f "${FINAL_BASELINE_MODEL_PATH}.ckpt" ]; then
        echo "    - Found baseline model with incorrect .ckpt.ckpt suffix. Renaming to .ckpt."
        mv "${FINAL_BASELINE_MODEL_PATH}.ckpt" "${FINAL_BASELINE_MODEL_PATH}"
    fi

    if [ -f "${FINAL_BASELINE_MODEL_PATH}" ]; then
        echo "    - Baseline expert model already exists. Skipping training."
    else
        echo "    - Training baseline expert..."
        python -u train_resnet.py \
            --data_path "${FOLD_DATA_DIR}/main/traffic_classification" \
            --model_path "${BASE_NAME_BASELINE}" \
            --task traffic
    fi

    # b) Train Minority Expert
    if [ -f "${FINAL_MINORITY_EXPERT_PATH}.ckpt" ]; then
        echo "    - Found minority expert model with incorrect .ckpt.ckpt suffix. Renaming to .ckpt."
        mv "${FINAL_MINORITY_EXPERT_PATH}.ckpt" "${FINAL_MINORITY_EXPERT_PATH}"
    fi

    if [ -f "${FINAL_MINORITY_EXPERT_PATH}" ]; then
        echo "    - Minority expert model already exists. Skipping training."
    else
        echo "    - Training minority expert..."
        python -u train_resnet.py \
            --data_path "${FOLD_DATA_DIR}/minority/traffic_classification" \
            --model_path "${BASE_NAME_MINORITY}" \
            --task traffic
    fi

    # c) Train Gating Network
    echo "    - Training gating network..."
    python -u train_gating_network.py \
        --train_data_path "${FOLD_DATA_DIR}/main/traffic_classification/train.parquet" \
        --baseline_model_path "${BASELINE_MODEL_PATH}" \
        --minority_model_path "${MINORITY_EXPERT_PATH}" \
        ${MINORITY_CLASSES_FOLD_STR_ARGS} \
        --output_path "${GATING_NETWORK_PATH}" \
        --epochs 20 \
        --lr 0.001

    # --- 3. Evaluation ---
    echo "--> Step 3: Evaluating GEE model for Fold ${EXCLUDED_CLASS}..."
    python -u evaluation.py \
        --data_path "${FOLD_DATA_DIR}/main/traffic_classification/test.parquet" \
        --model_path "${BASELINE_MODEL_PATH}" \
        --gating_network_path "${GATING_NETWORK_PATH}" \
        --expert_model_path "${MINORITY_EXPERT_PATH}" \
        --output_dir "${FOLD_EVAL_DIR}" \
        --eval-mode gating_ensemble \
        --open-set-eval \
        --unknown-classes "${EXCLUDED_CLASS}" \
        --label-map "${LABEL_MAP_STRING}" \
        ${KNOWN_CLASSES_ARGS} \
        ${MINORITY_CLASSES_FOLD_STR_ARGS} \
        --task-type traffic

    echo "--- Finished Fold ${EXCLUDED_CLASS} ---"
    echo ""

done

echo "================================================================="
echo "--- GEE Open-Set Evaluation Completed ---"
echo "================================================================="

# Optional: Add a final step to aggregate results from all folds
# python process_analysis_results.py --results_dir "${BASE_EVAL_DIR}"
