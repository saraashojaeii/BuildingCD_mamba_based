#!/bin/bash

# Multi-GPU training launch script for BuildingCD model (Cluster Version)
# This script uses Hugging Face Accelerate for distributed training

# Detect cluster environment and set appropriate variables
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES not set, detecting available GPUs..."
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# Set memory management for cluster environments
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

# Auto-detect number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Detected $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

# Check if we're in a single GPU environment (common cluster issue)
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Single GPU detected, using single GPU training..."
    python train_cd_multi_gpu.py \
        --config /home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/second_cdmamba.json \
        --phase train
else
    echo "Multi-GPU training with $NUM_GPUS GPUs..."
    
    # Use torchrun for better cluster compatibility
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_cd_multi_gpu.py \
        --config /home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/second_cdmamba.json \
        --phase train
fi

echo "Training completed!"
