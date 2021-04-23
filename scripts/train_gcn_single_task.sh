#!/bin/sh

python train.py --input data/Reviews.csv \
                --device cuda:2 \
                --learning_rate 0.001 \
                --epoch 200 \
                --text_feature False \
                --multi_task False \
                --random_seed 42
