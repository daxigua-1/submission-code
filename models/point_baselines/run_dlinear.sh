#!/usr/bin/env bash
set -e
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
HERE="$(cd "$(dirname "$0")" && pwd)"

PRED_LENS=${PRED_LENS:-"192"}
SEEDS=${SEEDS:-"2021 2022 2023"}
MODEL=DLinear
LOWER=dlinear

ROOT=./dataset/ETT-small/
for PRED_LEN in ${PRED_LENS}; do
  for DATA in ETTh1 ETTh2 ETTm1 ETTm2; do
    case "$DATA" in
      ETTh1) FREQ=h; LR=0.0001; BS=8 ;;
      ETTh2) FREQ=h; LR=0.05;   BS=32 ;;
      ETTm1) FREQ=t; LR=0.0001; BS=8 ;;
      ETTm2) FREQ=t; BS=32
             case "$PRED_LEN" in
               96|192) LR=0.001 ;;
               336)    LR=0.01 ;;
               *)      LR=0.1 ;;
             esac ;;
    esac
    for SEED in ${SEEDS}; do
      PREFIX=./exported_samples/${LOWER}_${DATA}_pl${PRED_LEN}_s${SEED}
      python "$HERE/run_point_baselines.py" \
        --model=${MODEL} --seed=${SEED} \
        --epochs=10 --patience=3 --lr=${LR} --batch_size=${BS} --lradj=type1 \
        --pearl_root_path=${ROOT} --pearl_data_path=${DATA}.csv \
        --pearl_data=${DATA} --pearl_features=M --pearl_seq_len=96 \
        --pearl_label_len=48 --pearl_pred_len=${PRED_LEN} --pearl_enc_in=7 \
        --pearl_embed=timeF --pearl_freq=${FREQ} \
        --out_prefix=${PREFIX}
      python "$HERE/evaluate_point_baseline.py" --prefix=${PREFIX}
    done
  done
done

ROOT="./SMD dataset/"
for PRED_LEN in ${PRED_LENS}; do
  for i in 1 2 3; do
    for SEED in ${SEEDS}; do
      PREFIX="./exported_samples/${LOWER}_M${i}_pl${PRED_LEN}_s${SEED}"
      python "$HERE/run_point_baselines.py" \
        --model=${MODEL} --seed=${SEED} --pearl_target=f_0 \
        --epochs=10 --patience=3 --lr=0.0001 --batch_size=32 --lradj=type1 \
        --pearl_root_path="${ROOT}" --pearl_data_path="machine-${i}-1.csv" \
        --pearl_data=custom --pearl_features=M --pearl_seq_len=96 \
        --pearl_label_len=48 --pearl_pred_len=${PRED_LEN} --pearl_enc_in=38 \
        --pearl_embed=timeF --pearl_freq=t \
        --out_prefix="${PREFIX}"
      python "$HERE/evaluate_point_baseline.py" --prefix="${PREFIX}"
    done
  done
done