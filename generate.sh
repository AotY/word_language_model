#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4


python generate.py \
    --data ./data/wikitext-2/ \
    --checkpoint ./models/model.pt \
    --outf ./data/generated.txt
    --words 35 \
    --seed 7 \
    --cuda \
    --log-interval 100 \
    --temperature 1.0

/


