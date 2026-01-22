#!/bin/bash

python utils/DDTIPreprocess.py
python utils/Augment.py --in_root="./dataset/DDTI/SDF/train" --out_root="./dataset/DDTI/Aug/train"
