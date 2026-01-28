#!/bin/bash

python utils/DDTIPreprocess.py
python utils/Augment.py --in_root="./dataset/DDTI/SDF/train" --out_root="./dataset/DDTI/Aug/train"

python train.py --train_root="./dataset/DDTI/Aug/train" --val_root="./dataset/DDTI/SDF/val" --output_root="./Outputs/MedFlowSeg-DDTI" --num_classes=2