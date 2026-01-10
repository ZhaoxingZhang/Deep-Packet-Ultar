#!/bin/bash
set -e

# ==================================================================================
# Open Set Recognition Baseline (ResNet) - Traffic Dataset (6-fold / Multi-fold)
# ==================================================================================

DATA_SOURCE="processed_data/traffic"
# Classes present in traffic dataset: 0 2 4 6 7 8 9 10
ALL_CLASSES="0 2 4 6 7 8 9 10"

BASE_DIR="train_test_data/exp_traffic/openset/baseline_resnet"
MODEL_DIR="model/exp_traffic/openset/baseline_resnet"
EVAL_DIR="evaluation_results/exp_traffic/openset/baseline_resnet"

mkdir -p "$BASE_DIR" "$MODEL_DIR" "$EVAL_DIR"

aurocs=()
fprs=()

for EXCLUDED_CLASS in $ALL_CLASSES; do
    echo "================================================================="
    echo "--- Fold: Exclude Class ${EXCLUDED_CLASS} ---"
    echo "================================================================="
    
    FOLD_DATA_DIR="${BASE_DIR}/exclude_${EXCLUDED_CLASS}"
    MODEL_PATH="${MODEL_DIR}/exclude_${EXCLUDED_CLASS}.pt"
    FOLD_EVAL_DIR="${EVAL_DIR}/exclude_${EXCLUDED_CLASS}"
    
    mkdir -p "$FOLD_EVAL_DIR"

    # 1. Generate Dataset (Hold Out Class)
    echo "--> Step 1: Generating Dataset..."
    if [ ! -f "${FOLD_DATA_DIR}/traffic_classification/train.parquet/_SUCCESS" ]; then
        python -u create_train_test_set.py \
            -s "$DATA_SOURCE" \
            -t "$FOLD_DATA_DIR" \
            --experiment_type open_set_hold_out \
            --task-type traffic \
            --exclude-classes "${EXCLUDED_CLASS}" \
            --fraction 1.0
    fi

    # 2. Train Baseline ResNet
    echo "--> Step 2: Training ResNet..."
    if [ ! -f "${MODEL_PATH}.ckpt" ]; then
        python -u train_resnet.py \
            --data_path "${FOLD_DATA_DIR}/traffic_classification" \
            --model_path "${MODEL_PATH}" \
            --task traffic \
            --epochs 50 \
            --validation_split 0.2
    fi

    # 3. Evaluate Open Set
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
        --model_path "${MODEL_PATH}.ckpt" \
        --data_path "${FOLD_DATA_DIR}/traffic_classification/test.parquet" \
        --output_dir "${FOLD_EVAL_DIR}" \
        --model_type resnet \
        --eval-mode standard \
        --open-set-eval \
        --unknown-classes "${EXCLUDED_CLASS}" \
        --label-map "${LABEL_MAP_STRING}" \
        ${KNOWN_CLASSES_ARGS}

    # Extract metrics
    RESULT_FILE="${FOLD_EVAL_DIR}/evaluation_summary.txt"
    if [ -f "$RESULT_FILE" ]; then
        AUROC=$(grep "AUROC:" "$RESULT_FILE" | awk '{print $2}')
        FPR=$(grep "FPR@TPR95:" "$RESULT_FILE" | awk '{print $2}')
        if [ -n "$AUROC" ]; then
            aurocs+=($AUROC)
            fprs+=($FPR)
            echo "--> Result: AUROC=${AUROC}, FPR=${FPR}"
        fi
    fi
done

# Aggregate results
echo ""
echo "================================================================="
echo "Aggregated Results"
echo "================================================================="
echo "AUROCs: ${aurocs[@]}"
echo "FPRs: ${fprs[@]}"
