#!/bin/bash

python utils/BUSIPreprocess.py
python utils/Augment.py --in_root="./dataset/BUSI/ONEHOT/train" --out_root="./dataset/BUSI/Aug/train"

python train.py --train_root="./dataset/BUSI/Aug/train" --val_root="./dataset/BUSI/ONEHOT/val" --output_root="./Outputs/MedFlowSeg-BUSI" --num_classes=1
python test.py --data_root="./dataset/BUSI/ONEHOT/test" --ckpt="./Outputs/MedFlowSeg-BUSI/checkpoints/last.ckpt" --out_dir="./Outputs/MedFlowSeg-BUSI_test" --num_classes=1