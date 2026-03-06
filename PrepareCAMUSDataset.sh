#!/bin/bash

python utils/CAMUSPreprocess.py
python utils/Augment.py --in_root="./dataset/CAMUS/SDF/train" --out_root="./dataset/CAMUS/Aug/train"

python train.py --train_root="./dataset/CAMUS/Aug/train" --val_root="./dataset/CAMUS/SDF/val" --output_root="./Outputs/MedFlowSeg-CAMUS" --num_classes=3