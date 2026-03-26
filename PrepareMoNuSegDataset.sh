#!/bin/bash

python utils/MoNuSegPreprocess.py
python utils/Augment.py --in_root="./dataset/MoNuSeg/ONEHOT/train" --out_root="./dataset/MoNuSeg/Aug/train"

python train.py --train_root="./dataset/MoNuSeg/Aug/train" --val_root="./dataset/MoNuSeg/ONEHOT/val" --output_root="./Outputs/MedFlowSeg-MoNuSeg" --num_classes=1 --img_in_ch=3
python test.py --data_root="./dataset/MoNuSeg/ONEHOT/test" --ckpt="./Outputs/MedFlowSeg-MoNuSeg/checkpoints/last.ckpt" --out_dir="./Outputs/MedFlowSeg-MoNuSeg_test" --num_classes=1 --img_in_ch=3