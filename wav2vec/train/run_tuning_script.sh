#!/bin/bash
#SBATCH --nodes=1  # 1 node
#SBATCH --gpus=2   # 2 GPUs on that node

echo "Start!"

torchrun --nproc_per_node=2 run.py \
    --optuna_trials 10 \
    --model_name_or_path="facebook/wav2vec2-xls-r-300m" \
    --output_dir="./wav2vec/data/result_wav2vec_tuning" \
    --num_train_epochs="5" \
    --save_steps="1000" \
    --eval_steps="500" \
    --logging_steps="100"

echo "Done!"
