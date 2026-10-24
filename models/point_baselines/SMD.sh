export CUDA_VISIBLE_DEVICES=0

############################################################iTransformer
#model_name=iTransformer
#
#for data_name in machine-1-1 machine-2-1 machine-3-1; do  # machine-2-1 machine-3-1
#  for pred_len in 96 192 336 720; do  # 96 192 336 720
#    python -u run.py \
#      --is_training 1 \
#      --root_path ./dataset/ \
#      --data_path ${data_name}.csv \
#      --model_id ${data_name}_96_${pred_len} \
#      --model $model_name \
#      --target f_0 \
#      --data custom \
#      --features M \
#      --seq_len 96 \
#      --pred_len $pred_len \
#      --e_layers 3 \
#      --enc_in 38 \
#      --dec_in 38 \
#      --c_out 38 \
#      --des 'Exp' \
#      --d_model 512 \
#      --d_ff 512 \
#      --itr 1
#  done
#done


#############################################################CASA
#model_name=CASA
#seq_len=96
#
## our method
#kernel=3
#learning_rate=0.0003
#
#for data_name in machine-1-1 machine-2-1 machine-3-1; do  # machine-2-1 machine-3-1
#  for pred_len in 96 192 336 720
#  do
#    python -u run.py \
#      --is_training 1 \
#      --root_path ./dataset/ \
#      --data_path ${data_name}.csv \
#      --model_id ${data_name}'_'$seq_len'_'$pred_len \
#      --model $model_name \
#      --data custom \
#      --features M \
#      --seq_len $seq_len \
#      --pred_len $pred_len \
#      --target f_0 \
#      --e_layers 4 \
#      --enc_in 38 \
#      --dec_in 38 \
#      --c_out 38 \
#      --d_model 512 \
#      --kernel $kernel \
#      --learning_rate $learning_rate \
#      --batch_size 32 \
#      --lradj cosine \
#      --train_epochs 10 \
#      --patience 10 \
#      --des 'Exp_'$kernel'_'$stride'_'$stem_kernel_size'_'$learning_rate \
#      --itr 1
#  done
#done




##############################################################TFPS
#
#model_name=PatchTST_MoE_cluster
#seq_len=96
#
#for data_name in machine-1-1 machine-2-1 machine-3-1; do
#  for pred_len in 96 192 336 720; do
#    python -u run_longExp.py \
#      --random_seed 2021 \
#      --is_training 1 \
#      --root_path ./dataset/ \
#      --data_path ${data_name}.csv \
#      --model_id ${data_name}_${seq_len}_${pred_len} \
#      --model $model_name \
#      --data custom \
#      --features M \
#      --seq_len $seq_len \
#      --target f_0 \
#      --pred_len $pred_len \
#      --enc_in 38 \
#      --dec_in 38 \
#      --c_out 38 \
#      --T_num_expert 8 \
#      --F_num_expert 8 \
#      --e_layers 3 \
#      --n_heads 8 \
#      --d_model 32 \
#      --d_ff 64 \
#      --individual 0 \
#      --dropout 0.2 \
#      --fc_dropout 0.2 \
#      --head_dropout 0 \
#      --patch_len 16 \
#      --stride 8 \
#      --beta 0.1 \
#      --des 'Exp' \
#      --train_epochs 100 \
#      --patience 20 \
#      --lradj 'TST' \
#      --pct_start 0.4 \
#      --batch_size 32 \
#      --gpu 6 \
#      --learning_rate 0.0005 \
#      --itr 1
#  done
#done



#############################################################SOFTS

#model_name=SOFTS
#seq_len=96
#
#for data_name in machine-1-1 machine-2-1 machine-3-1; do  # machine-2-1 machine-3-1
#  for pred_len in 96 192 336 720; do
#    python -u run.py \
#      --task_name long_term_forecast \
#      --is_training 1 \
#      --root_path ./dataset/ \
#      --data_path $data_name.csv \
#      --model_id $data_name_96_$pred_len \
#      --model $model_name \
#      --data custom \
#      --features M \
#      --seq_len 96 \
#      --pred_len $pred_len \
#      --target f_0 \
#      --e_layers 3 \
#      --enc_in 38 \
#      --dec_in 38 \
#      --c_out 38 \
#      --d_model 512 \
#      --d_core 128 \
#      --d_ff 512 \
#      --batch_size 16 \
#      --learning_rate 0.0003 \
#      --train_epochs 10 \
#      --patience 10 \
#      --lradj cosine \
#      --des 'Exp' \
#      --itr 1
#    done
#done


##############################################################FreTS
#
#model_name=FreLinear
#seq_len=96
#
#for data_name in machine-1-1 machine-2-1 machine-3-1; do  # machine-2-1 machine-3-1
#  for pred_len in 96 192 336 720; do
#    python -u run_longExp.py \
#      --root_path ./dataset/ \
#      --data_path ${data_name}.csv \
#      --model_id ${data_name}_$seq_len'_'$pred_len \
#      --model $model_name \
#      --data weather \
#      --features M \
#      --seq_len $seq_len \
#      --target f_0 \
#      --pred_len 96 \
#      --enc_in 38 \
#      --dec_in 38 \
#      --c_out 38 \
#      --itr 1
#    done
#done


#
##############################################################DLinear
#
#for data_name in machine-1-1 machine-2-1 machine-3-1; do  # machine-2-1 machine-3-1
#  for pred_len in 96 192 336 720; do
#    python -u run_longExp.py \
#      --is_training 1 \
#      --root_path ./dataset/ \
#      --data_path ${data_name}.csv \
#      --model_id ${data_name}_96'_'$pred_len \
#      --model DLinear \
#      --data custom \
#      --features M \
#      --seq_len 96 \
#      --pred_len $pred_len \
#      --target f_0 \
#      --enc_in 38 \
#      --des 'Exp' \
#      --batch_size 16 \
#      --itr 1
#    done
#done



##############################################################FEDformer
#
#for data_name in machine-1-1 machine-2-1 machine-3-1; do  # machine-2-1 machine-3-1
#  for pred_len in 96 192 336 720; do
#    python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ \
#     --data_path ${data_name}.csv \
#     --task_id ${data_name} \
#     --model FEDformer \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --target f_0 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 38 \
#     --dec_in 38 \
#     --c_out 38 \
#     --des 'Exp' \
#     --itr 1
#done
#done


##############################################################PatchTST
#
#for data_name in machine-1-1 machine-2-1 machine-3-1; do  # machine-2-1 machine-3-1
#  for pred_len in 96 192 336 720; do
#    python -u run_longExp.py \
#      --is_training 1 \
#      --root_path ./dataset/ \
#      --data_path ${data_name}.csv \
#      --model_id ${data_name}_96'_'$pred_len \
#      --model PatchTST \
#      --data custom \
#      --features M \
#      --seq_len 96 \
#      --pred_len $pred_len \
#      --enc_in 38 \
#      --dec_in 38 \
#      --c_out 38 \
#      --target f_0 \
#      --e_layers 3 \
#      --n_heads 16 \
#      --d_model 128 \
#      --d_ff 256 \
#      --dropout 0.2\
#      --fc_dropout 0.2\
#      --head_dropout 0\
#      --patch_len 16\
#      --stride 8\
#      --des 'Exp' \
#      --train_epochs 100\
#      --patience 20\
#      --batch_size 128 \
#      --learning_rate 0.0001 \
#      --itr 1
#done
#done


#############################################################TimesNet
model_name=TimesNet

for data_name in machine-1-1 machine-2-1 machine-3-1; do  # machine-2-1 machine-3-1
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ${data_name}.csv \
    --model_id ${data_name}_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --target f_0 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 38 \
    --dec_in 38 \
    --c_out 38 \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ${data_name}.csv \
    --model_id ${data_name}_96_192 \
    --model $model_name \
    --data custom \
    --features M \
    --target f_0 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 38 \
    --dec_in 38 \
    --c_out 38 \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 1


  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ${data_name}.csv \
    --model_id ${data_name}_96_336 \
    --model $model_name \
    --data custom \
    --features M \
    --target f_0 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 38 \
    --dec_in 38 \
    --c_out 38 \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ${data_name}.csv \
    --model_id ${data_name}_96_720 \
    --model $model_name \
    --data custom \
    --features M \
    --target f_0 \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 38 \
    --dec_in 38 \
    --c_out 38 \
    --d_model 32 \
    --d_ff 32 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 1
done