model_name=PEARL
seq_len=96
model_id_name=M3raw
export CUDA_VISIBLE_DEVICES=3

# our method
kernel=3
learning_rate=0.0003

for pred_len in 192
do
  python -u run.py \
    --is_training 1 \
    --root_path './SMD dataset/' \
    --data_path machine-3-1.csv \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --export_mu_prefix ./exported_samples/pearlmu_$model_id_name_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --enc_in 38 \
    --dec_in 38 \
    --c_out 38 \
    --target f_0 \
    --d_model 512 \
    --kernel $kernel \
    --learning_rate $learning_rate \
    --batch_size 32 \
    --lradj cosine \
    --train_epochs 10 \
    --patience 3 \
    --des 'Exp_'$kernel'_'$learning_rate \
    --itr 1
done
