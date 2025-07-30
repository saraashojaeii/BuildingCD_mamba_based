<div align="center">
    <h2>
        CDMamba: Incorporating Local Clues into Mamba for Remote Sensing Image Multi-class Change Detection
    </h2>
</div>
<br>


## Installation

### Requirements

- Linux system, Windows is not tested, depending on whether `causal-conv1d` and `mamba-ssm` can be installed
- Python 3.8+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.1.0
- CUDA 11.7 or higher, recommended 12.1

### Environment Installation

It is recommended to use Miniconda for installation. The following commands will create a virtual environment named `cd_mamba` and install PyTorch. In the following installation steps, the default installed CUDA version is **12.1**. If your CUDA version is not 12.1, please modify it according to the actual situation.

Note: If you are experienced with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow the steps below.

<details open>

**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

**Step 1**: Create a virtual environment named `cd_mamba` and activate it.

```shell
conda create -n cd_mamba python=3.10
conda activate cd_mamba
```

**Step 2**: Install dependencies.

```shell
pip install -r requirements.txt
```

```shell
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
pip install opencv-python-headless==4.8.0.76
```
**Note**: Please refer to https://github.com/hustvl/Vim or https://blog.csdn.net/weixin_45667052/article/details/136311600 when installing mamba.


</details>


### Install CDMamba


You can download or clone the CDMamba repository.

```shell
git clone git@github.com:saraashojaeii/BuildingCD_mamba_based.git
cd BuildingCD_mamba_based
```

## Dataset Organization Method

You can also choose other sources to download the data, but you need to organize the dataset in the following format：

```
${DATASET_ROOT} # Dataset root directory, for example: /home/username/data/LEVIR-CD
├── A
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── B
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── label
│   ├── train_1_1.png
│   ├── train_1_2.png
│   ├──...
│   ├── val_1_1.png
│   ├── val_1_2.png
│   ├──...
│   ├── test_1_1.png
│   ├── test_1_2.png
│   └── ...
├── list
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
```

## Model Training and Testing

All configuration for model training and testing are stored in the local folder `config`

#### Example of Training on LEVIR-CD Dataset

```shell
python train.py --config/mamba/levir_cdmamba.json 
```

#### Example of Testing on LEVIR-CD Dataset

```shell
python test.py --config/mamba/levir_test_cdmamba.json 
```
#### CDMamba Weight

Google Drive download link [https://drive.google.com/file/d/1ImTvjN-vPnlJtVwfemzeHWcjoNMFsrS7/view?usp=drive_link]

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

