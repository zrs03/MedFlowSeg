#!/bin/bash

python utils/COVID19Preprocess.py

python train.py --train_root="./dataset/covid19/ONEHOT/train" --val_root="./dataset/covid19/ONEHOT/val" --output_root="./Outputs/MedFlowSeg-covid19" --num_classes=1 --img_in_ch=1
python test.py --data_root="./dataset/covid19/ONEHOT/test" --ckpt="./Outputs/MedFlowSeg-covid19/checkpoints/last.ckpt" --out_dir="./Outputs/MedFlowSeg-covid19_test" --num_classes=1 --img_in_ch=1