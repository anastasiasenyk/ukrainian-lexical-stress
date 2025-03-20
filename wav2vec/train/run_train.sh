#!/bin/bash
#SBATCH --nodes=1  # 1 node
#SBATCH --gpus=2   # 2 GPUs on that node

echo "Start!"

torchrun --nproc_per_node 2 run_one_training.py \
    --model_name_or_path="facebook/wav2vec2-xls-r-300m" \
    --output_dir="../data/result_wav2vec" \
    --num_train_epochs="15" \
    --per_device_train_batch_size="16" \
    --per_device_eval_batch_size="16"  \
    --save_steps="500"  \
    --eval_steps="500"  \
    --eval_strategy="steps" \
    --logging_steps="100"  \
    --learning_rate=4e-5 \
    --save_total_limit="1"  \
    --fp16 \
    --metric_for_best_model="wer" \
    --greater_is_better=False  \
    --gradient_accumulation_steps="1" \
    --activation_dropout=0.05 \
    --attention_dropout=0.05 \
    --hidden_dropout=0.05   \
    --feat_proj_dropout=0.05 \
    --mask_time_prob=0.05  \
    --layerdrop=0.05 \
    --push_to_hub=True


echo "Done!"