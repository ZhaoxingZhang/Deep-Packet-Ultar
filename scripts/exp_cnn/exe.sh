nohup sh -c '
  bash scripts/exp_cnn/run_incremental_cnn_baseline.sh
  bash scripts/exp_cnn/run_incremental_cnn_gee.sh
  bash scripts/exp_cnn/run_open_set_cnn_baseline.sh
  bash scripts/exp_cnn/run_open_set_cnn_gee.sh
' > log/exp_cnn_021901.log 2>&1 &