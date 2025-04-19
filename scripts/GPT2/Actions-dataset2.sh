
export CUDA_VISIBLE_DEVICES=0

seq_len=100
model=GPT2

for lr in 0.005
do

python main.py \
    --train_dataset_name ./dataset/action/train_dataset_yzt.csv \
    --test_dataset_name ./dataset/action/test_dataset_yzt.csv \
    --data yzt \
    --model_id Actions_$data'_'$model'_'$gpt_layer'_'$seq_len \
    --seq_len $seq_len \
    --batch_size 256 \
    --learning_rate $lr \
    --train_epochs 300 \
    --patch_size 16 \
    --stride 8 \
    --gpt_layer 2 \
    --model $model \
    --input_dim 1 \
    --num_classes 4 \
    --lora True \
    --lstm True \
    
done