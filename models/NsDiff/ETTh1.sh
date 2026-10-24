#!/usr/bin/env bash
set -e
export PYTHONPATH=./
HERE="$(cd "$(dirname "$0")" && pwd)"

PRED_LEN=192
N_SAMPLES=20
ROOT=./dataset/ETT-small/

for DATA in ETTh1; do
  case "$DATA" in
    ETTh*) FREQ=h ;;
    ETTm*) FREQ=t ;;
  esac
  for SEED in 2021 2022 2023; do
    python "$HERE/run_nsdiff_on_pearl_data.py" \
      --dataset_type=${DATA} --device=cuda:3 --seed=${SEED} \
      --epochs=10 --patience=5 --n_samples=${N_SAMPLES} --select_by=mse \
      --pearl_root_path=${ROOT} --pearl_data_path=${DATA}.csv \
      --pearl_data=${DATA} --pearl_features=M --pearl_seq_len=96 \
      --pearl_label_len=48 --pearl_pred_len=${PRED_LEN} --pearl_enc_in=7 \
      --pearl_embed=timeF --pearl_freq=${FREQ} --pearl_batch_size=32 \
      --out_prefix=./exported_samples/nsdiff_${DATA}_pl${PRED_LEN}_s${SEED}
  done
done