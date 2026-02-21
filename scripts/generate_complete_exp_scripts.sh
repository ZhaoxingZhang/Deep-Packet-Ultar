#!/bin/bash
set -e

# ==================================================================================
# Generate Complete Experiment Scripts for Dataset 2 and 3
# ==================================================================================

echo "==================================================================="
echo "--- Generating Complete Experiment Scripts ---"
echo "==================================================================="
echo ""

# Function to generate incremental GEE script
generate_incremental_gee() {
    local name=$1
    local path=$2
    local exp_name=$3
    local minority_classes=$4
    local minority_args_hyphen=""
    local minority_args_underscore=""

    for c in $minority_classes; do
        minority_args_hyphen="$minority_args_hyphen --minority-classes $c"
        minority_args_underscore="$minority_args_underscore --minority_classes $c"
    done

    cat > "scripts/${exp_name}/run_incremental_resnet_gee.sh" << EOF
#!/bin/bash
set -e

# ==================================================================================
# Incremental Learning GEE (ResNet) - ${name} Dataset
# ==================================================================================

# Config
DATA_SOURCE="${path}"
BASE_DIR="train_test_data/${exp_name}/incremental/gee_resnet"
MODEL_DIR="model/${exp_name}/incremental/gee_resnet"
EVAL_DIR="evaluation_results/${exp_name}/incremental/gee_resnet"

# Classes ${minority_classes} are treated as "Minority/Expert" classes
MINORITY_CLASSES="${minority_classes}"
MINORITY_CLASSES_ARGS="${minority_args_hyphen}"
MINORITY_CLASSES_ARGS_US="${minority_args_underscore}"

mkdir -p "\$BASE_DIR" "\$MODEL_DIR" "\$EVAL_DIR"

BASELINE_MODEL_PATH="\${MODEL_DIR}/resnet_baseline.pt"
MINORITY_MODEL_PATH="\${MODEL_DIR}/resnet_minority_expert.pt"
GATING_MODEL_PATH="\${MODEL_DIR}/gating_network_resnet.pt"

# 1. Generate Datasets
echo "--> Step 1: Generating Datasets..."

# A) Main Dataset (Imbalanced)
echo "    Generating Main Dataset..."
if [ ! -f "\${BASE_DIR}/main/traffic_classification/train.parquet/_SUCCESS" ]; then
    python -u create_train_test_set.py \\
        -s "\$DATA_SOURCE" \\
        -t "\${BASE_DIR}/main" \\
        --experiment_type imbalanced \\
        --task-type traffic \\
        --fraction  0.01
else
    echo "    Main Dataset exists."
fi

# B) Minority Dataset
echo "    Generating Minority Dataset..."
if [ ! -f "\${BASE_DIR}/minority/traffic_classification/train.parquet/_SUCCESS" ]; then
    python -u create_train_test_set.py \\
        -s "\$DATA_SOURCE" \\
        -t "\${BASE_DIR}/minority" \\
        --experiment_type exp8_minority \\
        --task-type traffic \\
        \${MINORITY_CLASSES_ARGS} \\
        --fraction  0.2
else
    echo "    Minority Dataset exists."
fi

# 2. Train Models
echo "--> Step 2: Training Models..."

# A) Baseline ResNet
echo "    Training Baseline ResNet..."
if [ ! -f "\${BASELINE_MODEL_PATH}.ckpt" ]; then
    python -u train_resnet.py \\
        --data_path "\${BASE_DIR}/main/traffic_classification" \\
        --model_path "\${BASELINE_MODEL_PATH}" \\
        --task traffic \\
        --epochs 50 \\
        --validation_split 0.2
else
    echo "    Baseline Model exists."
fi

# B) Minority Expert ResNet
echo "    Training Minority Expert ResNet..."
if [ ! -f "\${MINORITY_MODEL_PATH}.ckpt" ]; then
    python -u train_resnet.py \\
        --data_path "\${BASE_DIR}/minority/traffic_classification" \\
        --model_path "\${MINORITY_MODEL_PATH}" \\
        --task traffic \\
        --epochs 50 \\
        --validation_split 0.2
else
    echo "    Minority Model exists."
fi

# C) Gating Network
echo "    Training Gating Network..."
if [ ! -f "\${GATING_MODEL_PATH}" ]; then
    python -u train_gating_network.py \\
        --train_data_path "\${BASE_DIR}/main/traffic_classification/train.parquet" \\
        --baseline_model_path "\${BASELINE_MODEL_PATH}.ckpt" \\
        --minority_model_path "\${MINORITY_MODEL_PATH}.ckpt" \\
        --baseline_model_type resnet \\
        --minority_model_type resnet \\
        \${MINORITY_CLASSES_ARGS_US} \\
        --output_path "\${GATING_MODEL_PATH}" \\
        --epochs 100 \\
        --lr 0.001
else
    echo "    Gating Model exists."
fi

# 3. Evaluate
echo "--> Step 3: Evaluating GEE..."
python -u evaluation.py \\
    --data_path "\${BASE_DIR}/main/traffic_classification/test.parquet" \\
    --baseline_model_path "\${BASELINE_MODEL_PATH}.ckpt" \\
    --minority_model_path "\${MINORITY_MODEL_PATH}.ckpt" \\
    --gating_network_path "\${GATING_MODEL_PATH}" \\
    --baseline_model_type resnet \\
    --minority_model_type resnet \\
    --output_dir "\$EVAL_DIR" \\
    --eval-mode gating_ensemble \\
    \${MINORITY_CLASSES_ARGS_US}

echo "Done."
EOF

    chmod +x "scripts/${exp_name}/run_incremental_resnet_gee.sh"
}

# Function to generate open set baseline script
generate_open_set_baseline() {
    local name=$1
    local path=$2
    local exp_name=$3
    local all_classes=$4

    cat > "scripts/${exp_name}/run_open_set_resnet_baseline.sh" << EOF
#!/bin/bash
set -e

# ==================================================================================
# Open Set Recognition Baseline (ResNet) - ${name} Dataset (6-fold)
# ==================================================================================

DATA_SOURCE="${path}"
ALL_CLASSES="${all_classes}"

BASE_DIR="train_test_data/${exp_name}/openset/baseline_resnet"
MODEL_DIR="model/${exp_name}/openset/baseline_resnet"
EVAL_DIR="evaluation_results/${exp_name}/openset/baseline_resnet"

mkdir -p "\$BASE_DIR" "\$MODEL_DIR" "\$EVAL_DIR"

aurocs=""
fprs=""

for EXCLUDED_CLASS in \$ALL_CLASSES; do
    echo "================================================================="
    echo "--- Fold: Exclude Class \${EXCLUDED_CLASS} ---"
    echo "================================================================="

    FOLD_DATA_DIR="\${BASE_DIR}/exclude_\${EXCLUDED_CLASS}"
    MODEL_PATH="\${MODEL_DIR}/exclude_\${EXCLUDED_CLASS}.pt"
    FOLD_EVAL_DIR="\${EVAL_DIR}/exclude_\${EXCLUDED_CLASS}"

    mkdir -p "\$FOLD_EVAL_DIR"

    # 1. Generate Dataset
    echo "--> Step 1: Generating Dataset..."
    if [ ! -f "\${FOLD_DATA_DIR}/traffic_classification/train.parquet/_SUCCESS" ]; then
        python -u create_train_test_set.py \\
            -s "\$DATA_SOURCE" \\
            -t "\${FOLD_DATA_DIR}" \\
            --experiment_type open_set_hold_out \\
            --task-type traffic \\
            --exclude-classes "\${EXCLUDED_CLASS}" \\
            --fraction 1.0
    fi

    # 2. Train Baseline ResNet
    echo "--> Step 2: Training ResNet..."
    if [ ! -f "\${MODEL_PATH}.ckpt" ]; then
        python -u train_resnet.py \\
            --data_path "\${FOLD_DATA_DIR}/traffic_classification" \\
            --model_path "\${MODEL_PATH}" \\
            --task traffic \\
            --epochs 50 \\
            --validation_split 0.2
    fi

    # 3. Evaluate Open Set
    echo "--> Step 3: Evaluating..."

    # Construct args
    KNOWN_CLASSES_ARGS=""
    LABEL_MAP_STRING=""
    NEW_LABEL_IDX=0
    for C in \$ALL_CLASSES; do
        if [ "\$C" != "\${EXCLUDED_CLASS}" ]; then
            KNOWN_CLASSES_ARGS="\$KNOWN_CLASSES_ARGS --known-classes \$C"
            if [ -z "\$LABEL_MAP_STRING" ]; then
                LABEL_MAP_STRING="\${NEW_LABEL_IDX}:\$C"
            else
                LABEL_MAP_STRING="\${LABEL_MAP_STRING},\${NEW_LABEL_IDX}:\$C"
            fi
            NEW_LABEL_IDX=\$((NEW_LABEL_IDX + 1))
        fi
    done

    python -u evaluation.py \\
        --model_path "\${MODEL_PATH}.ckpt" \\
        --data_path "\${FOLD_DATA_DIR}/traffic_classification/test.parquet" \\
        --output_dir "\$FOLD_EVAL_DIR" \\
        --model_type resnet \\
        --eval-mode standard \\
        --open-set-eval \\
        --unknown-classes "\${EXCLUDED_CLASS}" \\
        --label-map "\${LABEL_MAP_STRING}" \\
        \${KNOWN_CLASSES_ARGS}

    # Extract metrics
    RESULT_FILE="\${FOLD_EVAL_DIR}/evaluation_summary.txt"
    if [ -f "\$RESULT_FILE" ]; then
        AUROC=\$(grep "AUROC:" "\$RESULT_FILE" | awk '{print \$2}')
        FPR=\$(grep "FPR@TPR95:" "\$RESULT_FILE" | awk '{print \$2}')
        if [ -n "\$AUROC" ]; then
            aurocs="\$aurocs \$AUROC"
            fprs="\$fprs \$FPR"
            echo "--> Result: AUROC=\${AUROC}, FPR=\${FPR}"
        fi
    fi
done

# Aggregate results
echo ""
echo "================================================================="
echo "Aggregated Results"
echo "================================================================="
echo "AUROCs: \$aurocs"
echo "FPRs: \$fprs"
EOF

    chmod +x "scripts/${exp_name}/run_open_set_resnet_baseline.sh"
}

# Function to generate open set GEE script
generate_open_set_gee() {
    local name=$1
    local path=$2
    local exp_name=$3
    local all_classes=$4
    local minority_candidates=$5

    cat > "scripts/${exp_name}/run_open_set_resnet_gee.sh" << EOF
#!/bin/bash
set -e

# ==================================================================================
# Open Set Recognition GEE (ResNet) - ${name} Dataset (6-fold)
# ==================================================================================

DATA_SOURCE="${path}"
ALL_CLASSES="${all_classes}"
MINORITY_CANDIDATES="${minority_candidates}"

BASE_DIR="train_test_data/${exp_name}/openset/gee_resnet"
MODEL_DIR="model/${exp_name}/openset/gee_resnet"
EVAL_DIR="evaluation_results/${exp_name}/openset/gee_resnet"

mkdir -p "\$BASE_DIR" "\$MODEL_DIR" "\$EVAL_DIR"

for EXCLUDED_CLASS in \$ALL_CLASSES; do
    echo "================================================================="
    echo "--- GEE Fold: Exclude Class \${EXCLUDED_CLASS} ---"
    echo "================================================================="

    FOLD_DATA_DIR="\${BASE_DIR}/exclude_\${EXCLUDED_CLASS}"
    FOLD_MODEL_DIR="\${MODEL_DIR}/exclude_\${EXCLUDED_CLASS}"
    FOLD_EVAL_DIR="\${EVAL_DIR}/exclude_\${EXCLUDED_CLASS}"

    mkdir -p "\$FOLD_DATA_DIR" "\$FOLD_MODEL_DIR" "\$FOLD_EVAL_DIR"

    # Define paths
    BASELINE_MODEL="\${FOLD_MODEL_DIR}/baseline_resnet.pt"
    MINORITY_MODEL="\${FOLD_MODEL_DIR}/minority_expert_resnet.pt"
    GATING_MODEL="\${FOLD_MODEL_DIR}/gating_resnet.pt"

    # Determine Minority Classes for this fold
    CURRENT_MINORITY=""
    MINORITY_ARGS_HYPHEN=""
    MINORITY_ARGS_UNDERSCORE=""
    for C in \$MINORITY_CANDIDATES; do
        if [ "\$C" != "\${EXCLUDED_CLASS}" ]; then
            CURRENT_MINORITY="\$CURRENT_MINORITY \$C"
            MINORITY_ARGS_HYPHEN="\$MINORITY_ARGS_HYPHEN --minority-classes \$C"
            MINORITY_ARGS_UNDERSCORE="\$MINORITY_ARGS_UNDERSCORE --minority_classes \$C"
        fi
    done

    # 1. Generate Datasets
    echo "--> Step 1: Generating Datasets..."

    # A) Main (Known)
    if [ ! -f "\${FOLD_DATA_DIR}/main/traffic_classification/train.parquet/_SUCCESS" ]; then
        python -u create_train_test_set.py \\
            -s "\$DATA_SOURCE" \\
            -t "\${FOLD_DATA_DIR}/main" \\
            --experiment_type open_set_hold_out \\
            --task-type traffic \\
            --exclude-classes "\${EXCLUDED_CLASS}" \\
            --fraction  0.01
    fi

    # B) Minority
    if [ ! -f "\${FOLD_DATA_DIR}/minority/traffic_classification/train.parquet/_SUCCESS" ]; then
        python -u create_train_test_set.py \\
            -s "\$DATA_SOURCE" \\
            -t "\${FOLD_DATA_DIR}/minority" \\
            --experiment_type open_set_hold_out \\
            --task-type traffic \\
            --exclude-classes "\${EXCLUDED_CLASS}" \\
            \${MINORITY_ARGS_HYPHEN} \\
            --fraction  0.01
    fi

    # C) Garbage
    if [ ! -f "\${FOLD_DATA_DIR}/unknown/traffic_classification/train.parquet/_SUCCESS" ]; then
        python -u create_train_test_set.py \\
            -s "\$DATA_SOURCE" \\
            -t "\${FOLD_DATA_DIR}/unknown" \\
            --experiment_type select_classes \\
            --task-type traffic \\
            --minority-classes "\${EXCLUDED_CLASS}" \\
            --fraction  0.01
    fi

    # 2. Train Models
    echo "--> Step 2: Training Models..."

    # A) Baseline
    if [ ! -f "\${BASELINE_MODEL}.ckpt" ]; then
        python -u train_resnet.py \\
            --data_path "\${FOLD_DATA_DIR}/main/traffic_classification" \\
            --model_path "\${BASELINE_MODEL}" \\
            --task traffic \\
            --epochs 50 \\
            --validation_split 0.2
    fi

    # B) Minority Expert
    if [ ! -f "\${MINORITY_MODEL}.ckpt" ]; then
        python -u train_resnet.py \\
            --data_path "\${FOLD_DATA_DIR}/minority/traffic_classification" \\
            --model_path "\${MINORITY_MODEL}" \\
            --task traffic \\
            --epochs 50 \\
            --validation_split 0.2
    fi

    # C) Gating Network
    if [ ! -f "\${GATING_MODEL}" ]; then
        python -u train_gating_network.py \\
            --train_data_path "\${FOLD_DATA_DIR}/main/traffic_classification/train.parquet" \\
            --baseline_model_path "\${BASELINE_MODEL}.ckpt" \\
            --minority_model_path "\${MINORITY_MODEL}.ckpt" \\
            --baseline_model_type resnet \\
            --minority_model_type resnet \\
            \${MINORITY_ARGS_UNDERSCORE} \\
            --output_path "\${GATING_MODEL}" \\
            --epochs 100 \\
            --lr 0.001 \\
            --use-garbage-class \\
            --unknown-class-data-path "\${FOLD_DATA_DIR}/unknown/traffic_classification/train.parquet"
    fi

    # 3. Evaluate
    echo "--> Step 3: Evaluating..."

    # Construct args
    KNOWN_CLASSES_ARGS=""
    LABEL_MAP_STRING=""
    NEW_LABEL_IDX=0
    for C in \$ALL_CLASSES; do
        if [ "\$C" != "\${EXCLUDED_CLASS}" ]; then
            KNOWN_CLASSES_ARGS="\$KNOWN_CLASSES_ARGS --known-classes \$C"
            if [ -z "\$LABEL_MAP_STRING" ]; then
                LABEL_MAP_STRING="\${NEW_LABEL_IDX}:\$C"
            else
                LABEL_MAP_STRING="\${LABEL_MAP_STRING},\${NEW_LABEL_IDX}:\$C"
            fi
            NEW_LABEL_IDX=\$((NEW_LABEL_IDX + 1))
        fi
    done

    python -u evaluation.py \\
        --data_path "\${FOLD_DATA_DIR}/main/traffic_classification/test.parquet" \\
        --baseline_model_path "\${BASELINE_MODEL}.ckpt" \\
        --minority_model_path "\${MINORITY_MODEL}.ckpt" \\
        --gating_network_path "\${GATING_MODEL}" \\
        --baseline_model_type resnet \\
        --minority_model_type resnet \\
        --output_dir "\${FOLD_EVAL_DIR}" \\
        --eval-mode gating_ensemble \\
        --open-set-eval \\
        --unknown-classes "\${EXCLUDED_CLASS}" \\
        --label-map "\${LABEL_MAP_STRING}" \\
        \${KNOWN_CLASSES_ARGS} \\
        \${MINORITY_ARGS_UNDERSCORE} \\
        --gating-has-garbage-class

done

echo "Done."
EOF

    chmod +x "scripts/${exp_name}/run_open_set_resnet_gee.sh"
}

# Generate Dataset 2 scripts
echo ">>> Generating Dataset 2 (traffic_v2) scripts..."
generate_incremental_gee \
    "traffic_v2" \
    "processed_data/traffic_v2" \
    "exp_traffic_v2" \
    "5 8"

generate_open_set_baseline \
    "traffic_v2" \
    "processed_data/traffic_v2" \
    "exp_traffic_v2" \
    "0 2 4 5 6 8"

generate_open_set_gee \
    "traffic_v2" \
    "processed_data/traffic_v2" \
    "exp_traffic_v2" \
    "0 2 4 5 6 8" \
    "5 8"

# Generate Dataset 3 scripts
echo ">>> Generating Dataset 3 (traffic_v3) scripts..."
generate_incremental_gee \
    "traffic_v3" \
    "processed_data/traffic_v3" \
    "exp_traffic_v3" \
    "7 9"

generate_open_set_baseline \
    "traffic_v3" \
    "processed_data/traffic_v3" \
    "exp_traffic_v3" \
    "0 2 4 7 9 10"

generate_open_set_gee \
    "traffic_v3" \
    "processed_data/traffic_v3" \
    "exp_traffic_v3" \
    "0 2 4 7 9 10" \
    "7 9"

# Update exe.sh files
echo ">>> Updating exe.sh files..."

cat > scripts/exp_traffic_v2/exe.sh << 'EOF'
#!/bin/bash
set -e
nohup sh -c '
  bash scripts/exp_traffic_v2/run_incremental_resnet_baseline.sh
  bash scripts/exp_traffic_v2/run_incremental_resnet_gee.sh
  bash scripts/exp_traffic_v2/run_open_set_resnet_baseline.sh
  bash scripts/exp_traffic_v2/run_open_set_resnet_gee.sh
' > log/exp_traffic_v2.log 2>&1 &
echo "Dataset 2 experiments started. Check log/exp_traffic_v2.log for progress."
EOF

cat > scripts/exp_traffic_v3/exe.sh << 'EOF'
#!/bin/bash
set -e
nohup sh -c '
  bash scripts/exp_traffic_v3/run_incremental_resnet_baseline.sh
  bash scripts/exp_traffic_v3/run_incremental_resnet_gee.sh
  bash scripts/exp_traffic_v3/run_open_set_resnet_baseline.sh
  bash scripts/exp_traffic_v3/run_open_set_resnet_gee.sh
' > log/exp_traffic_v3.log 2>&1 &
echo "Dataset 3 experiments started. Check log/exp_traffic_v3.log for progress."
EOF

chmod +x scripts/exp_traffic_v2/exe.sh scripts/exp_traffic_v3/exe.sh

echo ""
echo "==================================================================="
echo "--- All Scripts Generated Successfully! ---"
echo "==================================================================="
echo ""
echo "Generated Scripts:"
echo ""
echo "Dataset 2 (traffic_v2 - Types: 0,2,4,5,6,8):"
echo "  - scripts/exp_traffic_v2/run_incremental_resnet_baseline.sh"
echo "  - scripts/exp_traffic_v2/run_incremental_resnet_gee.sh"
echo "  - scripts/exp_traffic_v2/run_open_set_resnet_baseline.sh"
echo "  - scripts/exp_traffic_v2/run_open_set_resnet_gee.sh"
echo "  Minority classes: 5 (VPN:Chat), 8 (VPN:Streaming)"
echo ""
echo "Dataset 3 (traffic_v3 - Types: 0,2,4,7,9,10):"
echo "  - scripts/exp_traffic_v3/run_incremental_resnet_baseline.sh"
echo "  - scripts/exp_traffic_v3/run_incremental_resnet_gee.sh"
echo "  - scripts/exp_traffic_v3/run_open_set_resnet_baseline.sh"
echo "  - scripts/exp_traffic_v3/run_open_set_resnet_gee.sh"
echo "  Minority classes: 7 (VPN:Email), 9 (VPN:Torrent)"
echo ""
echo "To run experiments:"
echo "  bash scripts/exp_traffic_v2/exe.sh"
echo "  bash scripts/exp_traffic_v3/exe.sh"
echo ""
