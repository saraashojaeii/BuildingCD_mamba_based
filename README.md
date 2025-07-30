<div align="center">
    <h2>
        CDMamba: Incorporating Local Clues into Mamba for Remote Sensing Image Multi-class Change Detection
    </h2>
</div>
<br>

<br>
<div align="center">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://ieeexplore.ieee.org/document/10902569">
    <span style="font-size: 20px; ">TGRS</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/zmoka-zht/CDMamba)](https://github.com/zmoka-zht/CDMamba)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![TGRS paper](https://img.shields.io/badge/TGRS-paper-00629B.svg)](https://ieeexplore.ieee.org/document/10902569) 



<br>
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

## Dataset Preparation


### Remote Sensing Change Detection Dataset

We provide the method of preparing the remote sensing change detection dataset used in the paper.

#### WHU-CD Dataset

- Data download link: [WHU-CD Dataset PanBaiDu](https://pan.baidu.com/s/1nh7znToO4XwaZHIOo7gCmw). Code:t2sb


#### LEVIR-CD Dataset 

- Data download link: [LEVIR-CD Dataset PanBaiDu](https://pan.baidu.com/s/1s5352sCRLxu50w2cEfSvWA). Code:qlvs


#### LEVIR+-CD Dataset

- Data download link: [LEVIR+-CD Dataset PanBaiDu](https://pan.baidu.com/s/1ymcsUei7oDyyMUBbpUTGAw ). Code: xtj8


#### Organization Method

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


If you have any other questions❓, please contact us in time 👬
