#!/bin/bash

python utils/REFUGE2cupPreprocess.py
python utils/Augment.py --in_root="./dataset/REFUGE2_cup/SDF/train" --out_root="./dataset/REFUGE2_cup/Aug/train"

python train.py --train_root="./dataset/REFUGE2_cup/Aug/train" --val_root="./dataset/REFUGE2_cup/SDF/val" --output_root="./Outputs/MedFlowSeg-REFUGE2_cup" --num_classes=2