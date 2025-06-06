#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

#model_name=PIXLGemma-v0-2b-finetune
model_name=PIXLPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python -m pixl.eval.model_vqa_mmbench \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$model_name.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $$model_name
