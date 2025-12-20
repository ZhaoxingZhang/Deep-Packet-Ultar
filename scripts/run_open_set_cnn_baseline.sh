#!/bin/bash
set -e

echo "================================================================="
echo "--- Running 6-Fold CNN Baseline Open-Set Recognition Evaluation ---"
echo "================================================================="

# --- Configuration ---
ALL_CLASSES=(5 6 7 8 9 10)
SOURCE_DATA_DIR="processed_data/vpn"
BASE_DATA_DIR="train_test_data/open_set_cnn_baseline" # New directory for CNN baseline data
BASE_MODEL_DIR="model/open_set_cnn_baseline"
BASE_EVAL_DIR="evaluation_results/open_set_cnn_baseline"

aurocs=()
fprs=()

# --- Main Loop for 6-Fold Cross-Validation ---
for EXCLUDED_CLASS in "${ALL_CLASSES[@]}"; do
    echo "================================================================="
    echo "--- Fold: Excluded Class ${EXCLUDED_CLASS} as Unknown ---"
    echo "================================================================="

    # --- Define Fold-Specific Variables ---
    FOLD_DATA_DIR="${BASE_DATA_DIR}/exp_exclude_${EXCLUDED_CLASS}"
    FOLD_MODEL_DIR="${BASE_MODEL_DIR}/exclude_${EXCLUDED_CLASS}"
    FOLD_EVAL_DIR="${BASE_EVAL_DIR}/exclude_${EXCLUDED_CLASS}"

    DATA_PATH_TRAIN="${FOLD_DATA_DIR}/traffic_classification/train.parquet"
    DATA_PATH_TEST="${FOLD_DATA_DIR}/traffic_classification/test.parquet"
    
    MODEL_NAME="baseline_exclude_${EXCLUDED_CLASS}.pt"
    MODEL_PATH_BASE="${FOLD_MODEL_DIR}/${MODEL_NAME}"
    FINAL_MODEL_PATH="${MODEL_PATH_BASE}.ckpt"

    mkdir -p "${FOLD_DATA_DIR}" "${FOLD_MODEL_DIR}" "${FOLD_EVAL_DIR}"

    # --- 1. Data Generation (if needed) ---
    echo "--> Step 1: Generating dataset..."
    if [ -f "${DATA_PATH_TRAIN}/_SUCCESS" ]; then
        echo "    - Dataset already exists. Skipping generation."
    else
        python -u create_train_test_set.py \
            -s "${SOURCE_DATA_DIR}" \
            -t "${FOLD_DATA_DIR}" \
            --experiment_type open_set_hold_out \
            --task-type traffic \
            --exclude-classes "${EXCLUDED_CLASS}" \
            --fraction 0.01 \
            --batch_size 50
    fi

    # --- 2. Train CNN Baseline Model ---
    echo "--> Step 2: Training CNN baseline model..."
    if [ ! -f "${FINAL_MODEL_PATH}" ]; then
        python -u train_cnn.py \
            --data_path "${DATA_PATH_TRAIN}" \
            --model_path "${MODEL_PATH_BASE}" \
            --task traffic
    else
        echo "    - Model already exists. Skipping training."
    fi

    # --- 3. Evaluate CNN Baseline Model (Open-Set) ---
    echo "--> Step 3: Evaluating CNN model for open-set performance..."
    KNOWN_CLASSES_ARGS=""
    LABEL_MAP_STRING=""
    NEW_LABEL_IDX=0
    for C in "${ALL_CLASSES[@]}"; do
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
        --model_path "${FINAL_MODEL_PATH}" \
        --data_path "${DATA_PATH_TEST}" \
        --output_dir "${FOLD_EVAL_DIR}" \
        --model_type cnn \
        --eval-mode standard \
        --open-set-eval \
        --unknown-classes "${EXCLUDED_CLASS}" \
        --label-map "${LABEL_MAP_STRING}" \
        ${KNOWN_CLASSES_ARGS}

    # --- 4. Extract and Store Results ---
    RESULT_FILE="${FOLD_EVAL_DIR}/evaluation_summary.txt"
    if [ -f "$RESULT_FILE" ]; then
        AUROC=$(grep "AUROC:" "$RESULT_FILE" | awk '{print $2}')
        FPR=$(grep "FPR@TPR95:" "$RESULT_FILE" | awk '{print $2}')
        
        if [ -n "$AUROC" ] && [ -n "$FPR" ]; then
            aurocs+=($AUROC)
            fprs+=($FPR)
            echo "--> Results for Fold ${EXCLUDED_CLASS}: AUROC=${AUROC}, FPR@TPR95=${FPR}"
        else
            echo "--> WARNING: Could not extract results from ${RESULT_FILE}"
        fi
    else
        echo "--> WARNING: Result file not found at ${RESULT_FILE}"
    fi

    echo ""
done

# --- Aggregate and Print Final Results ---
echo "================================================================="
echo "--- Aggregated CNN Baseline Open-Set Evaluation Results ---"
echo "================================================================="
if [ ${#aurocs[@]} -eq 0 ]; then
    echo "No results were collected. Cannot calculate average."
    exit 1
fi
echo "Individual AUROCs: ${aurocs[@]}"
echo "Individual FPRs@TPR95: ${fprs[@]}"
echo ""
auroc_str="${aurocs[*]}"
fpr_str="${fprs[*]}"
awk -v aurocs="$auroc_str" 'BEGIN { n = split(aurocs, arr, " "); sum=0; for (i=1; i<=n; i++) { sum+=arr[i]; } mean=sum/n; printf "Average AUROC: %.4f\n", mean; }'
awk -v fprs="$fpr_str" 'BEGIN { n = split(fprs, arr, " "); sum=0; for (i=1; i<=n; i++) { sum+=arr[i]; } mean=sum/n; printf "Average FPR@TPR95: %.4f\n", mean; }'

echo "================================================================="
echo "--- CNN Baseline OSR Evaluation Complete ---"
echo "================================================================="
