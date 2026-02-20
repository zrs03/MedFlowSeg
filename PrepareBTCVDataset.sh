#!/bin/bash

python utils/BTCVPreprocess.py

python utils/Augment.py --in_root="./dataset/BTCV/SDF/train" --out_root="./dataset/BTCV/Aug/train"
python train.py --train_root="./dataset/BTCV/Aug/train" --val_root="./dataset/BTCV/SDF/val" --output_root="./Outputs/MedFlowSeg-BTCV" --num_classes=14
python test.py --data_root="./dataset/BTCV/SDF/test" --ckpt="./Outputs/MedFlowSeg-BTCV/checkpoints/last.ckpt" --out_dir="./Outputs/MedFlowSeg-BTCV_test" --num_classes=14