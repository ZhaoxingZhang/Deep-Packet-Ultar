#!/bin/bash
set -e
nohup sh -c '
  bash scripts/exp_traffic_v2/run_incremental_resnet_baseline.sh
  bash scripts/exp_traffic_v2/run_incremental_resnet_gee.sh
  bash scripts/exp_traffic_v2/run_open_set_resnet_baseline.sh
  bash scripts/exp_traffic_v2/run_open_set_resnet_gee.sh
' > log/exp_traffic_v2_0221.log 2>&1 &
echo "Dataset 2 experiments started. Check log/exp_traffic_v2.log for progress."
