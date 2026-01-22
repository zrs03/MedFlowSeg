#!/bin/bash

python utils/REFUGE2Preprocess.py
python utils/Augment.py --in_root="./dataset/REFUGE2/SDF/train" --out_root="./dataset/REFUGE2/Aug/train"

python train.py --train_root="./dataset/REFUGE2/Aug/train" --val_root="./dataset/REFUGE2/SDF/val" --output_root="./Outputs/MedFlowSeg-REFUGE2" --num_classes=2