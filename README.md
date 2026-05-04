# MedFlowSeg

This repository hosts the official PyTorch implementation for our paper,  

**"MedFlowSeg: Flow Matching for Medical Image Segmentation with Frequency-Aware Attention"**.

## 🎯 Overview

<p align="center">
  <img src="https://github.com/yyxl123/MedFlowSeg/blob/new/assets/framework.png" width="600">
</p>

MedFlowSeg formulates medical image segmentation as a conditional flow matching problem. Instead of relying on iterative denoising as in diffusion-based segmentation methods, MedFlowSeg learns a time-dependent velocity field that transports a simple noise prior to the target segmentation distribution.

The framework adopts a two-stream architecture, consisting of a Flow Network and a Condition Network. The Condition Network provides clean semantic guidance, while the Flow Network models the generative trajectory from noisy intermediate states to segmentation masks.

## 🏋️‍♀️ Example Training

### 1. Clone Repository

```bash

cd MedFlowSeg

```

### 2. Create Environment

We recommend using Conda to manage the environment.

```bash
conda create -n medflowseg python=3.10 -y

conda activate medflowseg

# Install other dependencies
pip install -r requirements.txt
```

### 3. Prepare Datasets

**Cardiac Structures Segmentation from ACDC dataset**

1. Download the ACDC dataset from [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).

   Your final `ACDCdataset` directory structure should look similar to this:

~~~
ACDCdataset/
├── training
│   ├── patient001
│   │   ├──patient001_frame01.nii.gz
│   |   ├──patient001_frame01_gt.nii.gz
│   |   ├──patient001_frame12.nii.gz
│   |   ├──patient001_frame12_gt.nii.gz
│   │   └── (other documents)
│   ├── patient002
│   ... 
│   ├── patient100
│   └── MANDATORY_CITATION.md
├── testing
│   ├── patient101
│   │   ├──patient101_frame01.nii.gz
│   |   ├──patient101_frame01_gt.nii.gz
│   |   ├──patient101_frame14.nii.gz
│   |   ├──patient101_frame14_gt.nii.gz
│   │   └── (other documents)
│   ├── patient102
│   ... 
│   ├── patient150
│   └── MANDATORY_CITATION.md
└── (other documents)
~~~

2.  Preprocess Dataset

    ```bash
    bash PrepareACDCDataset.sh
    ```
    This script will preprocess and augment the dataset, split the files into 80% training, 10% validation, and 10% testing sets.
    You can find the preprocessed dataset in /dataset/ACDC.

**Gland Segmentation in Histology from GlaS dataset**

1. Download the GlaS dataset from [GlaS dataset](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation).

   Your final `GlaSdataset` directory structure should look similar to this:

~~~
GlaSdataset/
├── train_85.bmp
├── train_85_anno.bmp
... 
├── testA_2.bmp
├── testA_2_anno.bmp
...
├── testB_15.bmp
├── testB_15_anno.bmp
└── Grade.csv
~~~

2.  Preprocess Dataset

    ```bash
    bash PrepareGlaSDataset.sh
    ```
    This script will preprocess and augment the dataset, split the files into 80% training, 10% validation, and 10% testing sets.
    You can find the preprocessed dataset in /dataset/GlaS.


### 4. Model Training

Taking the ACDC dataset as an example, the training command is as follows:
```
python train.py \
    --train_root ./dataset/ACDC/Aug/train \
    --val_root ./dataset/ACDC/SDF/val \
    --img_in_ch 1 \
    --num_classes 3 \
    --sample_steps 50 \
    --lr 1e-4 \
    --batch_size 8 \
    --epochs 1000 \
    --output_root ./Outputs/MedFlowSeg-ACDC \
    --gpus 4
```


### 5. Model Evaluation

Similarly, taking the ACDC dataset as an example:
```
python test.py \
    --ckpt ./Outputs/MedFlowSeg-ACDC/checkpoints/your_best.ckpt \
    --data_root ./dataset/ACDC/SDF/test \
    --out_dir ./Outputs/MedFlowSeg_test_ACDC \
    --num_classes 3 \
    --sample_steps 50 \
    --img_in_ch 1 \
```

