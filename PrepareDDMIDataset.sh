#!/bin/bash

python utils/DDMIPreprocess.py
python utils/Augment.py --in_root="./dataset/DDMI/SDF/train" --out_root="./dataset/DDMI/Aug/train"
