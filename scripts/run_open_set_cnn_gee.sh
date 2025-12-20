#!/bin/bash
set -e

echo "================================================================="
echo "--- Running 6-Fold CNN-GEE Open-Set Recognition Evaluation ---"
echo "================================================================="

# --- Configuration ---
ALL_CLASSES=(5 6 7 8 9 10)
MINORITY_CLASSES_BASE=(5 7)
SOURCE_DATA_DIR="processed_data/vpn"
BASE_DATA_DIR="train_test_data/open_set_cnn_gee" # New directory for CNN GEE data
BASE_MODEL_DIR="model/open_set_cnn_gee"
BASE_EVAL_DIR="evaluation_results/open_set_cnn_gee"

# --- Main Loop for 6-Fold Cross-Validation ---
for EXCLUDED_CLASS in "${ALL_CLASSES[@]}"; do
    echo "================================================================="
    echo "--- Fold: Excluded Class ${EXCLUDED_CLASS} as Unknown ---"
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
    MINORITY_CLASSES_FOLD_STR_ARGS_HYPHEN=$(printf -- "--minority-classes %s " "${MINORITY_CLASSES_FOLD[@]}")
    MINORITY_CLASSES_FOLD_STR_ARGS_UNDERSCORE=$(printf -- "--minority_classes %s " "${MINORITY_CLASSES_FOLD[@]}")

    FOLD_DATA_DIR="${BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}"
    FOLD_MODEL_DIR="${BASE_MODEL_DIR}/exclude_${EXCLUDED_CLASS}"
    FOLD_EVAL_DIR="${BASE_EVAL_DIR}/exclude_${EXCLUDED_CLASS}"

    # Define model paths
    BASELINE_MODEL_BASE="${FOLD_MODEL_DIR}/baseline_cnn.pt"
    MINORITY_MODEL_BASE="${FOLD_MODEL_DIR}/minority_expert_cnn.pt"
    FINAL_BASELINE_MODEL_PATH="${BASELINE_MODEL_BASE}.ckpt"
    FINAL_MINORITY_EXPERT_PATH="${MINORITY_MODEL_BASE}.ckpt"
    GATING_NETWORK_PATH="${FOLD_MODEL_DIR}/gating_network_cnn_with_garbage.pt"

    mkdir -p "${FOLD_DATA_DIR}" "${FOLD_MODEL_DIR}" "${FOLD_EVAL_DIR}"

    # --- 1. Data Generation (if needed) ---
    echo "--> Step 1: Generating datasets..."
    # a) Main dataset for baseline expert
    MAIN_DATA_DIR="${FOLD_DATA_DIR}/main"
    if [ -f "${MAIN_DATA_DIR}/traffic_classification/train.parquet/_SUCCESS" ]; then
        echo "    - Main dataset already exists. Skipping."
    else
        python -u create_train_test_set.py \
            -s "${SOURCE_DATA_DIR}" -t "${MAIN_DATA_DIR}" \
            --experiment_type open_set_hold_out --exclude-classes "${EXCLUDED_CLASS}" \
            --task-type traffic --fraction 0.01 --batch_size 50
    fi
    # b) Minority expert dataset
    MINORITY_DATA_DIR="${FOLD_DATA_DIR}/minority"
    if [ ${#MINORITY_CLASSES_FOLD[@]} -gt 0 ] && [ ! -f "${MINORITY_DATA_DIR}/traffic_classification/train.parquet/_SUCCESS" ]; then
        python -u create_train_test_set.py \
            -s "${SOURCE_DATA_DIR}" -t "${MINORITY_DATA_DIR}" \
            --experiment_type open_set_hold_out --exclude-classes "${EXCLUDED_CLASS}" \
            ${MINORITY_CLASSES_FOLD_STR_ARGS_HYPHEN} --task-type traffic --fraction 0.01 --batch_size 50
    fi
    # c) Unknown (Garbage) class dataset
    UNKNOWN_DATA_DIR="${FOLD_DATA_DIR}/unknown"
    if [ ! -f "${UNKNOWN_DATA_DIR}/traffic_classification/train.parquet/_SUCCESS" ]; then
        python -u create_train_test_set.py \
            -s "${SOURCE_DATA_DIR}" -t "${UNKNOWN_DATA_DIR}" \
            --experiment_type select_classes --minority-classes "${EXCLUDED_CLASS}" \
            --task-type traffic --fraction 0.01 --batch_size 50
    fi

    # --- 2. Model Training ---
    echo "--> Step 2: Training models..."
    # a) Train Baseline CNN Expert
    if [ ! -f "${FINAL_BASELINE_MODEL_PATH}" ]; then
        python -u train_cnn.py --data_path "${MAIN_DATA_DIR}/traffic_classification" --model_path "${BASELINE_MODEL_BASE}" --task traffic
    else
        echo "    - Baseline CNN expert already exists. Skipping."
    fi
    # b) Train Minority CNN Expert
    if [ ${#MINORITY_CLASSES_FOLD[@]} -gt 0 ] && [ ! -f "${FINAL_MINORITY_EXPERT_PATH}" ]; then
        python -u train_cnn.py --data_path "${MINORITY_DATA_DIR}/traffic_classification" --model_path "${MINORITY_MODEL_BASE}" --task traffic
    else
        echo "    - Minority CNN expert already exists or not needed for this fold. Skipping."
    fi
    # c) Train Gating Network
    if [ ! -f "${GATING_NETWORK_PATH}" ]; then
        python -u train_gating_network.py \
            --train_data_path "${MAIN_DATA_DIR}/traffic_classification/train.parquet" \
            --baseline_model_path "${FINAL_BASELINE_MODEL_PATH}" \
            --minority_model_path "${FINAL_MINORITY_EXPERT_PATH}" \
            --baseline_model_type cnn \
            --minority_model_type cnn \
            ${MINORITY_CLASSES_FOLD_STR_ARGS_UNDERSCORE} \
            --output_path "${GATING_NETWORK_PATH}" \
            --epochs 200 --lr 0.001 --use-garbage-class \
            --unknown-class-data-path "${UNKNOWN_DATA_DIR}/traffic_classification/train.parquet"
    else
        echo "    - Gating network already exists. Skipping."
    fi

    # --- 3. Evaluation ---
    echo "--> Step 3: Evaluating CNN-GEE model for open-set performance..."
    python -u evaluation.py \
        --data_path "${MAIN_DATA_DIR}/traffic_classification/test.parquet" \
        --baseline_model_path "${FINAL_BASELINE_MODEL_PATH}" \
        --gating_network_path "${GATING_NETWORK_PATH}" \
        --minority_model_path "${FINAL_MINORITY_EXPERT_PATH}" \
        --output_dir "${FOLD_EVAL_DIR}" \
        --eval-mode gating_ensemble \
        --baseline_model_type cnn \
        --minority_model_type cnn \
        --open-set-eval --unknown-classes "${EXCLUDED_CLASS}" \
        --label-map "${LABEL_MAP_STRING}" \
        ${KNOWN_CLASSES_ARGS} \
        ${MINORITY_CLASSES_FOLD_STR_ARGS_UNDERSCORE} \
        --gating-has-garbage-class

    echo "--- Finished Fold ${EXCLUDED_CLASS} ---"
done

echo "================================================================="
echo "--- CNN-GEE Open-Set Evaluation Completed ---"
echo "================================================================="
