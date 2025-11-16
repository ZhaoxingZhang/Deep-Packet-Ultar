#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Define the classes to be excluded in each fold
CLASSES=(5 6 7 8 9 10)

# Loop through each class, treating it as the "unknown" class for one fold
for CLASS in "${CLASSES[@]}"
do
  echo "================================================================="
  echo "--- Starting Fold: Exclude Class ${CLASS} ---"
  echo "================================================================="

  # Define paths for this fold
  TARGET_DIR="train_test_data/open_set_hold_out/exp_exclude_${CLASS}"
  DATA_PATH="${TARGET_DIR}/traffic_classification/train.parquet"
  MODEL_PATH="model/open_set_hold_out/baseline_exclude_${CLASS}.pt"

  # --- Step 1: Generate Dataset ---
  echo "--> Step 1: Generating dataset, excluding class ${CLASS} from training set..."
  python -u create_train_test_set.py \
    -s processed_data/vpn \
    -t "${TARGET_DIR}" \
    --experiment_type open_set_hold_out \
    --task-type traffic \
    --exclude-classes "${CLASS}" \
    --fraction 0.01 \
    --batch_size 50

  # --- Step 2: Train Baseline Model ---
  echo "--> Step 2: Training baseline model on the new dataset..."
  python -u train_resnet.py \
    --data_path "${DATA_PATH}" \
    --model_path "${MODEL_PATH}" \
    --task traffic
    
  echo "--- Finished Fold for Class ${CLASS} ---"
  echo ""
done

echo "================================================================="
echo "All 6 folds have been processed successfully."
echo "================================================================="
