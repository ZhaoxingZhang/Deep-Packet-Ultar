#!/bin/bash
set -e

nohup sh -c '
  bash scripts/exp_traffic/run_incremental_resnet_baseline.sh
  bash scripts/exp_traffic/run_incremental_resnet_gee.sh
  bash scripts/exp_traffic/run_open_set_resnet_baseline.sh
  bash scripts/exp_traffic/run_open_set_resnet_gee.sh
' > log/exp_traffic.log 2>&1 &
echo "Experiment started. Check log/exp_traffic.log for progress."
