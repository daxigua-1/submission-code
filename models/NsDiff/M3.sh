#!/usr/bin/env bash
set -e
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=7
HERE="$(cd "$(dirname "$0")" && pwd)"

PRED_LEN=192
N_SAMPLES=20
ROOT="./SMD dataset/"

DATA=custom
DATA_PATH=machine-3-1
FREQ=t
for SEED in 2021 2022 2023; do
  python "$HERE/run_nsdiff_on_pearl_data.py" \
    --dataset_type=${DATA} --seed=${SEED} --pearl_target=f_0 \
    --epochs=10 --patience=5 --n_samples=${N_SAMPLES} --select_by=mse \
    --pearl_root_path="${ROOT}" --pearl_data_path=${DATA_PATH}.csv \
    --pearl_data=${DATA} --pearl_features=M --pearl_seq_len=96 \
    --pearl_label_len=48 --pearl_pred_len=${PRED_LEN} --pearl_enc_in=38 \
    --pearl_embed=timeF --pearl_freq=${FREQ} --pearl_batch_size=32 \
    --out_prefix=./exported_samples/nsdiff_M3_pl${PRED_LEN}_s${SEED}
done