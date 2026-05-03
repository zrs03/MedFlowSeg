#!/bin/bash

python utils/Kvasir-SessilePreprocess.py
python utils/Augment.py --in_root="./dataset/Kvasir-Sessile/ONEHOT/train" --out_root="./dataset/Kvasir-Sessile/Aug/train"

python train.py --train_root="./dataset/Kvasir-Sessile/Aug/train" --val_root="./dataset/Kvasir-Sessile/ONEHOT/val" --output_root="./Outputs/MedFlowSeg-Kvasir-Sessile" --num_classes=1 --img_in_ch=3
python test.py --data_root="./dataset/Kvasir-Sessile/ONEHOT/test" --ckpt="./Outputs/MedFlowSeg-Kvasir-Sessile/checkpoints/last.ckpt" --out_dir="./Outputs/MedFlowSeg-Kvasir-Sessile_test" --num_classes=1 --img_in_ch=3