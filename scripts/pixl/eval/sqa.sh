#!/bin/bash

#model_name=PIXLGemma-v0-2b-finetune
model_name=PiXLLaVAPhi2-v1-3b-finetune
SLM=phi_2
VIT=siglip
# MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

SLM=phi_2
model_name=PiXLLaVAPhi2-v2-3b-finetune
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python3 -m pixl.eval.model_vqa_science \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$model_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode v0

python3 pixl/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$model_name.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$model_name-output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$model_name-result.json

