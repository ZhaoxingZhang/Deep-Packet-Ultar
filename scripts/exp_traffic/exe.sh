nohup sh -c '
  sh scripts/exp_traffic/run_incremental_resnet_baseline.sh
  sh scripts/exp_traffic/run_incremental_resnet_gee.sh
  sh scripts/exp_traffic/run_open_set_resnet_baseline.sh
  sh scripts/exp_traffic/run_open_set_resnet_gee.sh
' > log/exp_traffic.log 2>&1 &