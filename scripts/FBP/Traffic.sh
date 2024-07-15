if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_new" ]; then
    mkdir ./logs/LongForecasting_new
fi

# model_name=FBP-L




# for pred_len in 96 192 336 720
# do
#     python -u run_longExp.py \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path traffic.csv\
#     --model_id  traffic\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len 336\
#     --pred_len $pred_len\
#     --decomposition 0\
#     --enc_in 20 \
#     --dec_in 20 \
#     --c_out 20\
#     --des 'Exp'\
#     --train_epochs 100\
#     --embedding 0\
#     --beta 1\
#     --itr 1 --batch_size 128 --learning_rate 0.00002 >logs/LongForecasting_new/Traffic__$model_name'_336_'$pred_len.log  
# done


#You may need to slightly tune the hyperparameters for this dataset default(learning rate: 0.000003, intermediate dimension: 1440, dropout: 0.15).
model_name=FBP-NL
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
    --is_training 1\
    --root_path ./dataset/ \
    --data_path traffic.csv\
    --model_id  traffic\
    --model  $model_name \
    --data custom \
    --features M \
    --seq_len 336\
    --pred_len $pred_len\
    --decomposition 0\
    --enc_in 20 \
    --dec_in 20 \
    --c_out 20\
    --des 'Exp'\
    --train_epochs 100\
    --embedding 0\
    --beta 1\
    --itr 1 --batch_size 128 --learning_rate 0.00003 >logs/LongForecasting_new/Traffic__$model_name'_336_'$pred_len.log  
done


# model_name=FBP-NP

# for pred_len in 96 192 336 720
# do
#     python -u run_longExp.py \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path traffic.csv\
#     --model_id  traffic\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len 336\
#     --pred_len $pred_len\
#     --decomposition 0\
#     --enc_in 20 \
#     --dec_in 20 \
#     --c_out 20\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --dropout 0.2\
#     --fc_dropout 0.2\
#     --head_dropout 0\
#     --patch_len 16\
#     --stride 8\
#     --des 'Exp'\
#     --train_epochs 100\
#     --embedding 0\
#     --beta 1\
#     --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic__$model_name'_336_'$pred_len.log  
# done




# model_name=PatchTST

# for pred_len in 96 192 336 720
# do
#     python -u run_longExp.py \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path traffic.csv\
#     --model_id  traffic\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len 336\
#     --pred_len $pred_len\
#     --decomposition 0\
#     --enc_in 20 \
#     --dec_in 20 \
#     --c_out 20\
#     --e_layers 3 \
#     --n_heads 16 \
#     --d_model 128 \
#     --d_ff 256 \
#     --dropout 0.2\
#     --fc_dropout 0.2\
#     --head_dropout 0\
#     --patch_len 16\
#     --stride 8\
#     --des 'Exp'\
#     --train_epochs 100\
#     --embedding 0\
#     --beta 1\
#     --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting_new/Traffic__$model_name'_336_'$pred_len.log  
# done

# model_name=NLinear

# for pred_len in 96 192 336 720
# do
#     python -u run_longExp.py \
#     --is_training 1\
#     --root_path ./dataset/ \
#     --data_path traffic.csv\
#     --model_id  traffic\
#     --model  $model_name \
#     --data custom \
#     --features M \
#     --seq_len 336\
#     --pred_len $pred_len\
#     --decomposition 0\
#     --enc_in 20 \
#     --dec_in 20 \
#     --c_out 20\
#     --des 'Exp'\
#     --train_epochs 100\
#     --embedding 0\
#     --beta 1\
#     --itr 1 --batch_size 128 --learning_rate 0.001 >logs/LongForecasting_new/Traffic__$model_name'_336_'$pred_len.log  
# done



