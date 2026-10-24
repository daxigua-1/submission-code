model_name=PEARL
seq_len=96
root_path=./
model_id_name=ETTh1
#export CUDA_VISIBLE_DEVICES=2

kernel=3
learning_rate=0.00024

#     --export_mu_prefix ./exported_samples/pearlmu_ETTh1_$pred_len \
for pred_len in 192
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 128 \
    --kernel $kernel \
    --learning_rate $learning_rate \
    --batch_size 32 \
    --lradj cosine \
    --train_epochs 20 \
    --patience 3 \
    --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
    --itr 1
done