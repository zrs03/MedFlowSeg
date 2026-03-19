#!/bin/bash

python utils/MonuSegPreprocess.py
python utils/Augment.py --in_root="./dataset/MonuSeg/ONEHOT/train" --out_root="./dataset/MonuSeg/Aug/train"

python train.py --train_root="./dataset/MonuSeg/Aug/train" --val_root="./dataset/MonuSeg/ONEHOT/val" --output_root="./Outputs/MedFlowSeg-MonuSeg" --num_classes=1 --img_in_ch=3
