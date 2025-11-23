#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define all classes as a space-separated string for POSIX compatibility.
ALL_CLASSES="5 6 7 8 9 10"

# Array to store results
aurocs=()
fprs=()

echo "Starting 6-fold open-set evaluation..."
echo ""

# Loop through each class, treating it as the "unknown" class for one fold
for EXCLUDED_CLASS in $ALL_CLASSES
do
  echo "================================================================="
  echo "--- Evaluating Fold: Excluded Class ${EXCLUDED_CLASS} as Unknown ---"
  echo "================================================================="

  # Define paths for this fold
  MODEL_PATH="model/open_set_hold_out/baseline_exclude_${EXCLUDED_CLASS}.pt.ckpt"
  DATA_PATH="train_test_data/open_set_hold_out/exp_exclude_${EXCLUDED_CLASS}/traffic_classification/test.parquet"
  OUTPUT_DIR="evaluation_results/open_set_hold_out/baseline_exclude_${EXCLUDED_CLASS}"
  
  # Check if the required model file exists
  if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found at ${MODEL_PATH}"
    echo "Please ensure the training script has been run successfully for all folds."
    exit 1
  fi

  # Determine the known classes for this fold
  KNOWN_CLASSES=""
  for C in $ALL_CLASSES;
  do
    if [ "$C" != "$EXCLUDED_CLASS" ]; then
      KNOWN_CLASSES="$KNOWN_CLASSES $C"
    fi
  done

  # --- Run Evaluation ---
  echo "--> Running evaluation with model ${MODEL_PATH}..."
  python evaluation.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_type resnet \
    --eval-mode standard \
    --open-set-eval \
    --unknown-classes "${EXCLUDED_CLASS}" \
    --known-classes ${KNOWN_CLASSES} # No quotes to pass as multiple args

  # --- Extract results ---
  RESULT_FILE="${OUTPUT_DIR}/evaluation_summary.txt"
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
echo "--- Aggregated Open-Set Evaluation Results ---"
echo "================================================================="

if [ ${#aurocs[@]} -eq 0 ]; then
  echo "No results were collected. Cannot calculate average."
  exit 1
fi

echo "Individual AUROCs: ${aurocs[@]}"
echo "Individual FPRs@TPR95: ${fprs[@]}"
echo ""

# Use awk for calculation
# Convert bash array to space-separated string for awk
auroc_str="${aurocs[*]}"
fpr_str="${fprs[*]}"

awk -v aurocs="$auroc_str" 'BEGIN { 
  n = split(aurocs, arr, " "); 
  sum=0; sumsq=0; 
  for (i=1; i<=n; i++) { sum+=arr[i]; sumsq+=arr[i]*arr[i]; }
  mean=sum/n; 
  stdev=sqrt(sumsq/n - mean*mean);
  printf "Average AUROC: %.4f (+/- %.4f)\n", mean, stdev;
}'

awk -v fprs="$fpr_str" 'BEGIN { 
  n = split(fprs, arr, " "); 
  sum=0; sumsq=0; 
  for (i=1; i<=n; i++) { sum+=arr[i]; sumsq+=arr[i]*arr[i]; }
  mean=sum/n; 
  stdev=sqrt(sumsq/n - mean*mean);
  printf "Average FPR@TPR95: %.4f (+/- %.4f)\n", mean, stdev;
}'
