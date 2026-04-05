#!/bin/bash

python utils/REFUGE2discPreprocess.py
python utils/Augment.py --in_root="./dataset/REFUGE2_disc/ONEHOT/train" --out_root="./dataset/REFUGE2_disc/Aug/train"

python train.py --train_root="./dataset/REFUGE2_disc/Aug/train" --val_root="./dataset/REFUGE2_disc/ONEHOT/val" --output_root="./Outputs/MedFlowSeg-REFUGE2_disc" --num_classes=1 --img_in_ch=3 --rgb_vis
python test.py --data_root="./dataset/REFUGE2_disc/ONEHOT/test" --ckpt="./Outputs/MedFlowSeg-REFUGE2_disc/checkpoints/last.ckpt" --out_dir="./Outputs/MedFlowSeg-REFUGE2_disc_test" --num_classes=1 --img_in_ch=3