#!/bin/bash

python utils/REFUGE2discPreprocess.py
python utils/Augment.py --in_root="./dataset/REFUGE2_disc/SDF/train" --out_root="./dataset/REFUGE2_disc/Aug/train"

python train.py --train_root="./dataset/REFUGE2_disc/Aug/train" --val_root="./dataset/REFUGE2_disc/SDF/val" --output_root="./Outputs/MedFlowSeg-REFUGE2_disc" --num_classes=2
python test.py --data_root="./dataset/REFUGE2_disc/SDF/test" --ckpt="./Outputs/MedFlowSeg-REFUGE2_disc/checkpoints/last.ckpt" --out_dir="./Outputs/MedFlowSeg-REFUGE2_disc_test" --num_classes=2