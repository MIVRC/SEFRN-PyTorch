#!/bin/bash

#example

CUDA_VISIBLE_DEVICES=4 python main.py \
    --exp_dir=./result/SEFRN \
    --netType=SEFRN \
    --dataName=cc359 \
    --accer=4 \
    --dataMode=complex \
    --batchSize=4 \
    --challenge=singlecoil \
    --lr=0.0005 \
    --train_root="use your own root for training data" \
    --valid_root="use your own roow for valid data" \
    --accer=4 \
    --center_fractions=0.08 
    --resolution=256

