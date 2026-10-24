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
    python "$HERE/run_diffusionts_on_pearl_data.py" \
      --seed=${SEED} --device=cuda:0 --n_samples=${N_SAMPLES} \
      --sampling_steps=50 \
      --max_steps=6000 \
      --test_stride=2 \
      --val_windows=20 \
      --sample_batch=256 \
      --pearl_root_path=${ROOT} --pearl_data_path=${DATA}.csv \
      --pearl_data=${DATA} --pearl_features=M --pearl_seq_len=96 \
      --pearl_label_len=48 --pearl_pred_len=${PRED_LEN} --pearl_enc_in=7 \
      --pearl_embed=timeF --pearl_freq=${FREQ} \
      --out_prefix=./exported_samples/diffusionts_${DATA}_pl${PRED_LEN}_s${SEED}
  done
done