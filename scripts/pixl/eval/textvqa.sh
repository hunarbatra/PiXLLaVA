#!/bin/bash

#model_name=PIXLGemma-v0-2b-finetune
model_name=PIXLPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python -m pixl.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /data/team/zhumj/data/finetune/data/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode v0

python -m pixl.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$model_name.jsonl
