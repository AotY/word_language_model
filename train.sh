#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4


python main.py \
    --data ./data/wikitext-2/ \
    --model LSTM \
    --emsize 256 \
    --nhid 256 \
    --nlayers 2 \
    --lr 0.001 \
    --clip 1.0 \
    --batch_size 32 \
    --bptt 35 \
    --dropout 0.8 \
    --tied \
    --seed 7 \
    --device cuda \
    --log-interval 100 \
    --save ./models

/


