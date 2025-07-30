#!/bin/bash

# Launch script for multi-GPU training with Accelerate
# Make sure to modify the number of GPUs and config path as needed

# Set the number of GPUs you want to use
export CUDA_VISIBLE_DEVICES=0,1  # Modify this based on your available GPUs

# Launch training with accelerate
accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port 29500 \
    train_cd_accelerate.py \
    --config /home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/second_cdmamba.json \
    --phase train \
    --gpu_ids 0,1

echo "Multi-GPU training completed!"
