# Multi-GPU Training Setup with Accelerate

## Prerequisites

1. Install accelerate:
```bash
pip install accelerate>=0.20.0
```

2. Make sure you have multiple GPUs available:
```bash
nvidia-smi
```

## Training Commands

### For 2 GPUs:
```bash
export CUDA_VISIBLE_DEVICES=0,1
accelerate launch \
    --config_file accelerate_config_2gpu.yaml \
    --main_process_port 29500 \
    train_cd_accelerate.py \
    --config config/second_cdmamba/second_cdmamba.json \
    --phase train
```

### For 4 GPUs:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch \
    --config_file accelerate_config_4gpu.yaml \
    --main_process_port 29500 \
    train_cd_accelerate.py \
    --config config/second_cdmamba/second_cdmamba.json \
    --phase train
```

### Alternative: Using accelerate launch without config file
```bash
# For 2 GPUs
accelerate launch --num_processes=2 --mixed_precision=fp16 train_cd_accelerate.py --config config/second_cdmamba/second_cdmamba.json --phase train

# For 4 GPUs  
accelerate launch --num_processes=4 --mixed_precision=fp16 train_cd_accelerate.py --config config/second_cdmamba/second_cdmamba.json --phase train
```

## Key Changes Made

1. **Replaced DataParallel with Accelerate**: Better performance and memory efficiency
2. **Automatic device handling**: No need to manually move data to devices
3. **Proper distributed training**: Each GPU gets its own process
4. **Synchronized logging**: Only main process logs to avoid conflicts
5. **Model saving**: Properly unwraps distributed model before saving

## Files Created

- `train_cd_accelerate.py`: New training script with Accelerate support
- `accelerate_config_2gpu.yaml`: Configuration for 2 GPU training
- `accelerate_config_4gpu.yaml`: Configuration for 4 GPU training
- `requirement.txt`: Updated with accelerate dependency

## Benefits of Accelerate over DataParallel

- **Better memory efficiency**: Each GPU has its own process
- **Faster training**: True distributed training vs data parallel
- **Easier scaling**: Can easily scale to multiple nodes
- **Mixed precision**: Built-in support for fp16/bf16
- **Gradient accumulation**: Better handling of large batch sizes
