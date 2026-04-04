#!/bin/bash
# Quick start script: Download data AND prepare analysis in parallel

set -e

echo "==================================================================="
echo "  Parallel Execution: Download Data + Prepare Analysis"
echo "==================================================================="
echo ""

# Start data download in background
echo "[1/2] Starting test data download in background..."
cd analysis_scripts

# Create log directory
mkdir -p logs

# Start download in background
nohup bash download_test_data.sh > logs/download.log 2>&1 &
DOWNLOAD_PID=$!

echo "  ✓ Download started (PID: $DOWNLOAD_PID)"
echo "  Monitor: tail -f analysis_scripts/logs/download.log"
echo ""

# While data downloads, do validation and preparation
echo "[2/2] Doing preparation work while waiting..."
echo ""

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "  Installing Python dependencies..."
    pip install -q -r requirements.txt
else
    echo "  ✓ Dependencies already installed"
fi

# Validate models while data downloads
echo ""
echo "  Running model validation..."
python validate_and_wait.py --generate-sample

echo ""
echo "==================================================================="
echo "  Status Check"
echo "==================================================================="

# Check if download is still running
if ps -p $DOWNLOAD_PID > /dev/null; then
    echo "✓ Download still running (PID: $DOWNLOAD_PID)"
    echo "  Progress: tail -f analysis_scripts/logs/download.log"
else
    echo "✓ Download completed (or failed)"
fi

echo ""
echo "Next steps:"
echo "1. Wait for download: tail -f analysis_scripts/logs/download.log"
echo "2. When download completes, run:"
echo "   cd analysis_scripts"
echo "   python extract_features.py --config config.yaml --experiment traffic --scenario incremental --device cpu"
echo ""
echo "Or check current status:"
echo "   python validate_and_wait.py"
