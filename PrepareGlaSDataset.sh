#!/bin/bash

python utils/GlaSPreprocess.py
python utils/Augment.py --in_root="./dataset/GlaS/ONEHOT/train" --out_root="./dataset/GlaS/Aug/train"

python train.py --train_root="./dataset/GlaS/Aug/train" --val_root="./dataset/GlaS/ONEHOT/val" --output_root="./Outputs/MedFlowSeg-GlaS" --num_classes=1 --img_in_ch=3
python test.py --data_root="./dataset/GlaS/ONEHOT/test" --ckpt="./Outputs/MedFlowSeg-GlaS/checkpoints/last.ckpt" --out_dir="./Outputs/MedFlowSeg-GlaS_test" --num_classes=1 --img_in_ch=3