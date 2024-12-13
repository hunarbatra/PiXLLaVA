#!/bin/bash

SLM=phi_35
model_name=PiXLLaVAPhi35-v2-3b-merged
VIT=siglip
CONV_MODE=phi3

MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python -m pixl.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr_roi.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

# model_name=llava-v1.5-13b

python -m pixl.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$model_name.jsonl
