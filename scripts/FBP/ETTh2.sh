if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi

model_name=FBP-L

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path ETTh2.csv \
    --model_id  ETTh2 \
    --model  $model_name  \
    --data ETTh2 \
    --features M \
    --seq_len 336\
    --pred_len $pred_len\
    --decomposition 0\
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7\
    --des 'Exp'\
    --train_epochs 100\
    --itr 1 --batch_size 128 --learning_rate  0.00005 >logs/LongForecasting_new/ETTh2__$model_name'_336_'$pred_len.log  
done

model_name=FBP-NL

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path ETTh2.csv \
    --model_id  ETTh2 \
    --model  $model_name  \
    --data ETTh2 \
    --features M \
    --seq_len 336\
    --pred_len $pred_len\
    --decomposition 0\
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7\
    --des 'Exp'\
    --train_epochs 100\
    --itr 1 --batch_size 128 --learning_rate  0.00005 >logs/LongForecasting_new/ETTh2__$model_name'_336_'$pred_len.log  
done


model_name=FBP-NP

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path ETTh2.csv \
    --model_id  ETTh2 \
    --model  $model_name  \
    --data ETTh2 \
    --features M \
    --seq_len 336\
    --pred_len $pred_len\
    --decomposition 0\
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7\
    --e_layers 3\
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3\
    --fc_dropout 0.3\
    --head_dropout 0\
    --stride 8\
    --patch_num 14\
    --des 'Exp'\
    --train_epochs 100\
    --itr 1 --batch_size 128 --learning_rate 0.0005 >logs/LongForecasting_new/ETTh2__$model_name'_336_'$pred_len.log  
done


# model_name=PatchTST

# for pred_len in 96 192 336 720
# do
#     python -u run_longExp.py \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path ETTh2.csv \
#     --model_id  ETTh2 \
#     --model  $model_name  \
#     --data ETTh2 \
#     --features M \
#     --seq_len 336\
#     --pred_len $pred_len\
#     --decomposition 0\
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7\
#     --e_layers 3\
#     --n_heads 4 \
#     --d_model 16 \
#     --d_ff 128 \
#     --dropout 0.3\
#     --fc_dropout 0.3\
#     --head_dropout 0\
#     --patch_len 16\
#     --stride 8\
#     --des 'Exp'\
#     --train_epochs 100\
#     --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting_new/ETTh2__$model_name'_336_'$pred_len.log  
# done

# model_name=NLinear

# for pred_len in 96 192 336 720
# do
#     python -u run_longExp.py \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path ETTh2.csv \
#     --model_id  ETTh2 \
#     --model  $model_name  \
#     --data ETTh2 \
#     --features M \
#     --seq_len 336\
#     --pred_len $pred_len\
#     --decomposition 0\
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7\
#     --des 'Exp'\
#     --train_epochs 100\
#     --itr 1 --batch_size 128 --learning_rate 0.001 >logs/LongForecasting_new/ETTh2__$model_name'_336_'$pred_len.log  
# done



