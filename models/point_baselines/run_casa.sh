##!/usr/bin/env bash
#set -e
#export PYTHONPATH=./
#export CUDA_VISIBLE_DEVICES=4
#HERE="$(cd "$(dirname "$0")" && pwd)"
#
#PRED_LEN=192
#ROOT=./dataset/ETT-small/
#
#for MODEL in CASA; do
#  LOWER=$(echo "$MODEL" | tr 'A-Z' 'a-z')
#
#  for DATA in ETTh1 ETTh2 ETTm1 ETTm2; do
#    case "$DATA" in
#      ETTh1)
#        FREQ=h; KERNEL=3; LR=0.00024; E_LAYERS=1; D_MODEL=128; BATCH=32; EPOCHS=20; PATIENCE=3
#        ;;
#      ETTh2)
#        FREQ=h; KERNEL=3; LR=0.0001; E_LAYERS=1; D_MODEL=256; BATCH=64; EPOCHS=20; PATIENCE=5
#        ;;
#      ETTm1)
#        FREQ=t; KERNEL=13; LR=0.00025; E_LAYERS=2; D_MODEL=512; BATCH=64; EPOCHS=50; PATIENCE=3
#        ;;
#      ETTm2)
#        FREQ=t; KERNEL=3; LR=0.0001; E_LAYERS=2; D_MODEL=512; BATCH=32; EPOCHS=20; PATIENCE=3
#        ;;
#    esac
#
#    for SEED in 2021 2022 2023; do
#      PREFIX=./exported_samples/${LOWER}_${DATA}_pl${PRED_LEN}_s${SEED}
#
#      python "$HERE/run_point_baselines.py" \
#        --model=${MODEL} --seed=${SEED} \
#        --epochs=${EPOCHS} --patience=${PATIENCE} \
#        --lr=${LR} --batch_size=${BATCH} --lradj=cosine \
#        --override e_layers=${E_LAYERS} d_model=${D_MODEL} kernel=${KERNEL} \
#        --pearl_root_path=${ROOT} --pearl_data_path=${DATA}.csv \
#        --pearl_data=${DATA} --pearl_features=M --pearl_seq_len=96 \
#        --pearl_label_len=48 --pearl_pred_len=${PRED_LEN} --pearl_enc_in=7 \
#        --pearl_embed=timeF --pearl_freq=${FREQ} \
#        --out_prefix=${PREFIX}
#
#      python "$HERE/evaluate_point_baseline.py" --prefix=${PREFIX}
#    done
#  done
#done
#


##############################################################################################SMD
#!/usr/bin/env bash
set -e
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=3
HERE="$(cd "$(dirname "$0")" && pwd)"

PRED_LEN=192
ROOT="./SMD dataset/"

for MODEL in CASA; do
  LOWER=$(echo "$MODEL" | tr 'A-Z' 'a-z')
  DATA=custom
  FREQ=t
  for i in 1 2 3; do
    DATA_PATH="machine-${i}-1"
    for SEED in 2021 2022 2023; do
      PREFIX="./exported_samples/${LOWER}_M${i}_pl${PRED_LEN}_s${SEED}"
      python "$HERE/run_point_baselines.py" \
        --model=${MODEL} --seed=${SEED} --pearl_target=f_0 \
        --epochs=10 --patience=10 \
        --lr=0.0003 --batch_size=32 --lradj=cosine \
        --override e_layers=4 d_model=512 kernel=3 \
        --pearl_root_path="${ROOT}" --pearl_data_path="${DATA_PATH}.csv" \
        --pearl_data=${DATA} --pearl_features=M --pearl_seq_len=96 \
        --pearl_label_len=48 --pearl_pred_len=${PRED_LEN} --pearl_enc_in=38 \
        --pearl_embed=timeF --pearl_freq=${FREQ} \
        --out_prefix="${PREFIX}"

      python "$HERE/evaluate_point_baseline.py" --prefix="${PREFIX}"
    done
  done
done