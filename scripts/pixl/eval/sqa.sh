#!/bin/bash

SLM=phi_35
model_name=PiXLLaVAPhi35-v2-3b-merged
VIT=siglip
CONV_MODE=phi3

MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python3 -m pixl.eval.model_vqa_science \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A_roi.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$model_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE

# model_name=llava-v1.5-13b

python3 pixl/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$model_name.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$model_name-output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$model_name-result.json

