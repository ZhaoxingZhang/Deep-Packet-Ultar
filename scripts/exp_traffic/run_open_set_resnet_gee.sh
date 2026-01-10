#!/bin/bash
set -e

# ==================================================================================
# Open Set Recognition GEE (ResNet) - Traffic Dataset (6-fold / Multi-fold)
# ==================================================================================

DATA_SOURCE="processed_data/traffic"
ALL_CLASSES="0 2 4 6 7 8 9 10"
# We define "Expert" classes as the ones that are usually harder or minority.
# For consistency across folds, let's say 4 and 7 are potential candidates for "Minority Expert".
# If one of them is excluded, the other remains the expert.
MINORITY_CANDIDATES="4 7"

BASE_DIR="train_test_data/exp_traffic/openset/gee_resnet"
MODEL_DIR="model/exp_traffic/openset/gee_resnet"
EVAL_DIR="evaluation_results/exp_traffic/openset/gee_resnet"

mkdir -p "$BASE_DIR" "$MODEL_DIR" "$EVAL_DIR"

for EXCLUDED_CLASS in $ALL_CLASSES; do
    echo "================================================================="
    echo "--- GEE Fold: Exclude Class ${EXCLUDED_CLASS} ---"
    echo "================================================================="

    FOLD_DATA_DIR="${BASE_DIR}/exclude_${EXCLUDED_CLASS}"
    FOLD_MODEL_DIR="${MODEL_DIR}/exclude_${EXCLUDED_CLASS}"
    FOLD_EVAL_DIR="${EVAL_DIR}/exclude_${EXCLUDED_CLASS}"
    
    mkdir -p "$FOLD_DATA_DIR" "$FOLD_MODEL_DIR" "$FOLD_EVAL_DIR"

    # Define paths
    BASELINE_MODEL="${FOLD_MODEL_DIR}/baseline_resnet.pt"
    MINORITY_MODEL="${FOLD_MODEL_DIR}/minority_expert_resnet.pt"
    GATING_MODEL="${FOLD_MODEL_DIR}/gating_resnet.pt"

    # Determine Minority Classes for this fold (exclude the Unknown class)
    CURRENT_MINORITY=""
    MINORITY_ARGS_HYPHEN=""
    MINORITY_ARGS_UNDERSCORE=""
    for C in $MINORITY_CANDIDATES; do
        if [ "$C" != "$EXCLUDED_CLASS" ]; then
            CURRENT_MINORITY="$CURRENT_MINORITY $C"
            MINORITY_ARGS_HYPHEN="$MINORITY_ARGS_HYPHEN --minority-classes $C"
            MINORITY_ARGS_UNDERSCORE="$MINORITY_ARGS_UNDERSCORE --minority_classes $C"
        fi
    done
    
    # 1. Generate Datasets
    echo "--> Step 1: Generating Datasets..."
    
    # A) Main (Known)
    if [ ! -f "${FOLD_DATA_DIR}/main/traffic_classification/train.parquet/_SUCCESS" ]; then
        python -u create_train_test_set.py \
            -s "$DATA_SOURCE" \
            -t "${FOLD_DATA_DIR}/main" \
            --experiment_type open_set_hold_out \
            --task-type traffic \
            --exclude-classes "${EXCLUDED_CLASS}" \
            --fraction 1.0
    fi
    
    # B) Minority
    if [ ! -f "${FOLD_DATA_DIR}/minority/traffic_classification/train.parquet/_SUCCESS" ]; then
        # If CURRENT_MINORITY is empty (e.g. if we only had 1 candidate and it was excluded), 
        # this logic would fail. But we have 2 candidates (4, 7), so at least one remains.
        python -u create_train_test_set.py \
            -s "$DATA_SOURCE" \
            -t "${FOLD_DATA_DIR}/minority" \
            --experiment_type open_set_hold_out \
            --task-type traffic \
            --exclude-classes "${EXCLUDED_CLASS}" \
            ${MINORITY_ARGS_HYPHEN} \
            --fraction 1.0
    fi
    
    # C) Garbage (The Unknown Class for Training Gate)
    if [ ! -f "${FOLD_DATA_DIR}/unknown/traffic_classification/train.parquet/_SUCCESS" ]; then
        python -u create_train_test_set.py \
            -s "$DATA_SOURCE" \
            -t "${FOLD_DATA_DIR}/unknown" \
            --experiment_type select_classes \
            --task-type traffic \
            --minority-classes "${EXCLUDED_CLASS}" \
            --fraction 1.0
    fi

    # 2. Train Models
    echo "--> Step 2: Training Models..."

    # A) Baseline
    if [ ! -f "${BASELINE_MODEL}.ckpt" ]; then
        python -u train_resnet.py \
            --data_path "${FOLD_DATA_DIR}/main/traffic_classification" \
            --model_path "${BASELINE_MODEL}" \
            --task traffic \
            --epochs 50 \
            --validation_split 0.2
    fi

    # B) Minority Expert
    if [ ! -f "${MINORITY_MODEL}.ckpt" ]; then
        python -u train_resnet.py \
            --data_path "${FOLD_DATA_DIR}/minority/traffic_classification" \
            --model_path "${MINORITY_MODEL}" \
            --task traffic \
            --epochs 50 \
            --validation_split 0.2
    fi

    # C) Gating Network
    if [ ! -f "${GATING_MODEL}" ]; then
        python -u train_gating_network.py \
            --train_data_path "${FOLD_DATA_DIR}/main/traffic_classification/train.parquet" \
            --baseline_model_path "${BASELINE_MODEL}.ckpt" \
            --minority_model_path "${MINORITY_MODEL}.ckpt" \
            --baseline_model_type resnet \
            --minority_model_type resnet \
            ${MINORITY_ARGS_UNDERSCORE} \
            --output_path "${GATING_MODEL}" \
            --epochs 100 \
            --lr 0.001 \
            --use-garbage-class \
            --unknown-class-data-path "${FOLD_DATA_DIR}/unknown/traffic_classification/train.parquet"
    fi

    # 3. Evaluate
    echo "--> Step 3: Evaluating..."
    
    # Construct args
    KNOWN_CLASSES_ARGS=""
    LABEL_MAP_STRING=""
    NEW_LABEL_IDX=0
    for C in $ALL_CLASSES; do
        if [ "$C" != "$EXCLUDED_CLASS" ]; then
            KNOWN_CLASSES_ARGS="$KNOWN_CLASSES_ARGS --known-classes $C"
            if [ -z "$LABEL_MAP_STRING" ]; then
                LABEL_MAP_STRING="${NEW_LABEL_IDX}:${C}"
            else
                LABEL_MAP_STRING="${LABEL_MAP_STRING},${NEW_LABEL_IDX}:${C}"
            fi
            NEW_LABEL_IDX=$((NEW_LABEL_IDX + 1))
        fi
    done

    python -u evaluation.py \
        --data_path "${FOLD_DATA_DIR}/main/traffic_classification/test.parquet" \
        --baseline_model_path "${BASELINE_MODEL}.ckpt" \
        --minority_model_path "${MINORITY_MODEL}.ckpt" \
        --gating_network_path "${GATING_MODEL}" \
        --baseline_model_type resnet \
        --minority_model_type resnet \
        --output_dir "${FOLD_EVAL_DIR}" \
        --eval-mode gating_ensemble \
        --open-set-eval \
        --unknown-classes "${EXCLUDED_CLASS}" \
        --label-map "${LABEL_MAP_STRING}" \
        ${KNOWN_CLASSES_ARGS} \
        ${MINORITY_ARGS_UNDERSCORE} \
        --gating-has-garbage-class

done

echo "Done."
