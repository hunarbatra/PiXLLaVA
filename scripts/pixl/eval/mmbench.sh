#!/bin/bash

SPLIT="mmbench_dev_20230712_roi"

SLM=phi_35
model_name=PiXLLaVAPhi35-3b-merged
VIT=siglip
CONV_MODE=phi3

MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python -m pixl.eval.model_vqa_mmbench \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$model_name.jsonl \
    --image-folder ./playground/data/eval/mmbench \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

# model_name=llava-v1.5-13b
# SPLIT="mmbench_dev_20230712"

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $model_name