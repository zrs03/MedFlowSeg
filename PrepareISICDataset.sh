#!/bin/bash

python utils/ISICPreprocess.py
python utils/Augment.py --in_root="./dataset/ISIC/SDF/train" --out_root="./dataset/ISIC/Aug/train"

python train.py --train_root="./dataset/ISIC/Aug/train" --val_root="./dataset/ISIC/SDF/val" --output_root="./Outputs/MedFlowSeg-ISIC" --num_classes=1 --img_in_ch=3