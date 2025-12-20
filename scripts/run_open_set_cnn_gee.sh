#!/bin/bash
set -e

# Try to set JAVA_HOME dynamically if not set
if [ -z "$JAVA_HOME" ]; then
    if [ -x "/usr/libexec/java_home" ]; then
        export JAVA_HOME=$(/usr/libexec/java_home)
    elif [ -d "/usr/lib/jvm/java-11-openjdk-amd64" ]; then
        export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
    fi
fi

echo "================================================================="
echo "--- Running 6-Fold CNN-GEE Open-Set Recognition Evaluation ---"
echo "================================================================="

# --- Configuration ---
ALL_CLASSES=(5 6 7 8 9 10)
MINORITY_CLASSES_BASE=(5 7)
SOURCE_DATA_DIR="processed_data/vpn"
LOCAL_BASE_DATA_DIR="train_test_data/open_set_cnn_gee"
SHARED_BASE_DATA_DIR="train_test_data/open_set_gee"
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

    FOLD_DATA_DIR="${LOCAL_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}"
    FOLD_MODEL_DIR="${BASE_MODEL_DIR}/exclude_${EXCLUDED_CLASS}"
    FOLD_EVAL_DIR="${BASE_EVAL_DIR}/exclude_${EXCLUDED_CLASS}"

    mkdir -p "${FOLD_DATA_DIR}" "${FOLD_MODEL_DIR}" "${FOLD_EVAL_DIR}"

    # --- Step 1: Data Generation / Reuse Logic ---
    echo "--> Step 1: Preparing datasets..."

    # a) Main dataset
    LOCAL_MAIN="${LOCAL_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}/main/traffic_classification"
    SHARED_MAIN="${SHARED_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}/main/traffic_classification"
    if [ -d "${LOCAL_MAIN}/train.parquet" ]; then
        echo "    - Using local main dataset."
        MAIN_TRAIN_PATH="${LOCAL_MAIN}/train.parquet"
        MAIN_TEST_PATH="${LOCAL_MAIN}/test.parquet"
        MAIN_DATA_DIR_TRAIN="${LOCAL_MAIN}"
    elif [ -d "${SHARED_MAIN}/train.parquet" ]; then
        echo "    - Found shared main dataset. Using it."
        MAIN_TRAIN_PATH="${SHARED_MAIN}/train.parquet"
        MAIN_TEST_PATH="${SHARED_MAIN}/test.parquet"
        MAIN_DATA_DIR_TRAIN="${SHARED_MAIN}"
    else
        echo "    - Generating local main dataset..."
        python -u create_train_test_set.py \
            -s "${SOURCE_DATA_DIR}" -t "${LOCAL_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}/main" \
            --experiment_type open_set_hold_out --exclude-classes "${EXCLUDED_CLASS}" \
            --task-type traffic --fraction 0.01 --batch_size 50
        MAIN_TRAIN_PATH="${LOCAL_MAIN}/train.parquet"
        MAIN_TEST_PATH="${LOCAL_MAIN}/test.parquet"
        MAIN_DATA_DIR_TRAIN="${LOCAL_MAIN}"
    fi

    # b) Minority expert dataset
    if [ ${#MINORITY_CLASSES_FOLD[@]} -gt 0 ]; then
        LOCAL_MIN="${LOCAL_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}/minority/traffic_classification"
        SHARED_MIN="${SHARED_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}/minority/traffic_classification"
        if [ -d "${LOCAL_MIN}/train.parquet" ]; then
            echo "    - Using local minority dataset."
            MINORITY_TRAIN_DIR="${LOCAL_MIN}"
        elif [ -d "${SHARED_MIN}/train.parquet" ]; then
            echo "    - Found shared minority dataset. Using it."
            MINORITY_TRAIN_DIR="${SHARED_MIN}"
        else
            echo "    - Generating local minority dataset..."
            python -u create_train_test_set.py \
                -s "${SOURCE_DATA_DIR}" -t "${LOCAL_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}/minority" \
                --experiment_type open_set_hold_out --exclude-classes "${EXCLUDED_CLASS}" \
                ${MINORITY_CLASSES_FOLD_STR_ARGS_HYPHEN} --task-type traffic --fraction 0.01 --batch_size 50
            MINORITY_TRAIN_DIR="${LOCAL_MIN}"
        fi
    fi

    # c) Unknown (Garbage) class dataset
    LOCAL_UNK="${LOCAL_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}/unknown/traffic_classification"
    SHARED_UNK="${SHARED_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}/unknown/traffic_classification"
    if [ -d "${LOCAL_UNK}/train.parquet" ]; then
        echo "    - Using local unknown dataset."
        UNKNOWN_TRAIN_PATH="${LOCAL_UNK}/train.parquet"
    elif [ -d "${SHARED_UNK}/train.parquet" ]; then
        echo "    - Found shared unknown dataset. Using it."
        UNKNOWN_TRAIN_PATH="${SHARED_UNK}/train.parquet"
    else
        echo "    - Generating local unknown dataset..."
        python -u create_train_test_set.py \
            -s "${SOURCE_DATA_DIR}" -t "${LOCAL_BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}/unknown" \
            --experiment_type select_classes --minority-classes "${EXCLUDED_CLASS}" \
            --task-type traffic --fraction 0.01 --batch_size 50
        UNKNOWN_TRAIN_PATH="${LOCAL_UNK}/train.parquet"
    fi

    # --- Step 2: Model Training ---
    echo "--> Step 2: Training models..."
    BASELINE_MODEL_BASE="${FOLD_MODEL_DIR}/baseline_cnn.pt"
    MINORITY_MODEL_BASE="${FOLD_MODEL_DIR}/minority_expert_cnn.pt"
    FINAL_BASELINE_MODEL_PATH="${BASELINE_MODEL_BASE}.ckpt"
    FINAL_MINORITY_EXPERT_PATH="${MINORITY_MODEL_BASE}.ckpt"
    GATING_NETWORK_PATH="${FOLD_MODEL_DIR}/gating_network_cnn_with_garbage.pt"

    # a) Train Baseline CNN Expert
    if [ ! -f "${FINAL_BASELINE_MODEL_PATH}" ]; then
        python -u train_cnn.py --data_path "${MAIN_DATA_DIR_TRAIN}" --model_path "${BASELINE_MODEL_BASE}" --task traffic
    fi
    # b) Train Minority CNN Expert
    if [ ${#MINORITY_CLASSES_FOLD[@]} -gt 0 ] && [ ! -f "${FINAL_MINORITY_EXPERT_PATH}" ]; then
        python -u train_cnn.py --data_path "${MINORITY_TRAIN_DIR}" --model_path "${MINORITY_MODEL_BASE}" --task traffic
    fi
    # c) Train Gating Network
    if [ ! -f "${GATING_NETWORK_PATH}" ]; then
        python -u train_gating_network.py \
            --train_data_path "${MAIN_TRAIN_PATH}" \
            --baseline_model_path "${FINAL_BASELINE_MODEL_PATH}" \
            --minority_model_path "${FINAL_MINORITY_EXPERT_PATH}" \
            --baseline_model_type cnn --minority_model_type cnn \
            ${MINORITY_CLASSES_FOLD_STR_ARGS_UNDERSCORE} \
            --output_path "${GATING_NETWORK_PATH}" \
            --epochs 200 --lr 0.001 --use-garbage-class \
            --unknown-class-data-path "${UNKNOWN_TRAIN_PATH}"
    fi

    # --- Step 3: Evaluation ---
    echo "--> Step 3: Evaluating CNN-GEE model..."
    python -u evaluation.py \
        --data_path "${MAIN_TEST_PATH}" \
        --baseline_model_path "${FINAL_BASELINE_MODEL_PATH}" \
        --gating_network_path "${GATING_NETWORK_PATH}" \
        --minority_model_path "${FINAL_MINORITY_EXPERT_PATH}" \
        --output_dir "${FOLD_EVAL_DIR}" \
        --eval-mode gating_ensemble \
        --baseline_model_type cnn --minority_model_type cnn \
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