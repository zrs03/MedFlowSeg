#!/bin/bash

python utils/ACDCPreprocess.py
python utils/Augment.py --in_root="./dataset/ACDC/SDF/train" --out_root="./dataset/ACDC/Aug/train"
