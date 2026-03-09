#!/bin/bash

python utils/ISICPreprocess.py
python utils/Augment.py --in_root="./dataset/ISIC/SDF/train" --out_root="./dataset/ISIC/Aug/train" --img_in_ch=3