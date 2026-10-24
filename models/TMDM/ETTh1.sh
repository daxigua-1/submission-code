#!/usr/bin/env bash
set -e
export PYTHONPATH=./
HERE="$(cd "$(dirname "$0")" && pwd)"

PRED_LEN=192
N_SAMPLES=20
TEST_BATCH_SIZE=8
ROOT=./dataset/ETT-small/

for DATA in ETTh1; do
  case "$DATA" in
    ETTh*) FREQ=h ;;
    ETTm*) FREQ=t ;;
  esac
  for SEED in 2021 2022 2023; do
    CUDA_VISIBLE_DEVICES=0 python "$HERE/run_tmdm_on_pearl_data.py" \
      --seed=${SEED} --epochs=50 --patience=10 \
      --n_z_samples=${N_SAMPLES} --test_batch_size=${TEST_BATCH_SIZE} \
      --diffusion_config_dir=./models/TMDM/model9_NS_transformer/configs/toy_8gauss.yml \
      --pearl_root_path=${ROOT} --pearl_data_path=${DATA}.csv \
      --pearl_data=${DATA} --pearl_features=M --pearl_seq_len=96 \
      --pearl_label_len=48 --pearl_pred_len=${PRED_LEN} --pearl_enc_in=7 \
      --pearl_embed=timeF --pearl_freq=${FREQ} --pearl_batch_size=32 \
      --out_prefix=./exported_samples/tmdm_${DATA}_pl${PRED_LEN}_s${SEED}
  done
done