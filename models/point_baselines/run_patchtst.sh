#!/usr/bin/env bash
set -e
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=3
HERE="$(cd "$(dirname "$0")" && pwd)"

PRED_LEN=192
ROOT="./SMD dataset/"

for MODEL in PatchTST; do
  LOWER=$(echo "$MODEL" | tr 'A-Z' 'a-z')
  DATA=custom
  FREQ=t
  for i in 1 2 3; do
    DATA_PATH="machine-${i}-1"
    for SEED in 2021 2022 2023; do
      PREFIX="./exported_samples/${LOWER}_M${i}_pl${PRED_LEN}_s${SEED}"
      python "$HERE/run_point_baselines.py" \
        --model=${MODEL} --seed=${SEED} --pearl_target=f_0 \
        --epochs=100 --patience=20 \
        --lr=0.0001 --batch_size=128 \
        --override e_layers=3 n_heads=16 d_model=128 d_ff=256 dropout=0.2 fc_dropout=0.2 patch_len=16 stride=8 head_dropout=0 pct_start=0.3 \
        --lradj=type3 \
        --pearl_root_path="${ROOT}" --pearl_data_path=${DATA_PATH}.csv \
        --pearl_data=${DATA} --pearl_features=M --pearl_seq_len=96 \
        --pearl_label_len=48 --pearl_pred_len=${PRED_LEN} --pearl_enc_in=38 \
        --pearl_embed=timeF --pearl_freq=${FREQ} \
        --out_prefix="${PREFIX}"

      python "$HERE/evaluate_point_baseline.py" --prefix="${PREFIX}"
    done
  done
done