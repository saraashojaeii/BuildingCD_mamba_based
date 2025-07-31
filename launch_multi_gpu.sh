#!/bin/bash

# Multi-GPU training launch script for BuildingCD model
# This script uses Hugging Face Accelerate for distributed training

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Set number of GPUs to use
NUM_GPUS=4

echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Launch training with accelerate
accelerate launch \
    --config_file accelerate_config_multi_gpu.yaml \
    --num_processes $NUM_GPUS \
    --main_process_port 29500 \
    train_cd_multi_gpu.py \
    --config /home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/second_cdmamba.json \
    --phase train

echo "Multi-GPU training completed!"
