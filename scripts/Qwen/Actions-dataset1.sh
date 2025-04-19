
export CUDA_VISIBLE_DEVICES=0

seq_len=100
model=Qwen

for lr in 0.005
do

python main.py \
    --train_dataset_name ./dataset/action/train_dataset_lyh_2.csv \
    --test_dataset_name ./dataset/action/test_dataset_lyh_2.csv \
    --answer_csv_name ./answer/lyh_2_model.csv \
    --answer_label_name ./answer/lyh_2_model_with_labels.csv \
    --data lyh_2 \
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
    --num_classes 5 \
    --lora True \
    --lstm True \
    
done