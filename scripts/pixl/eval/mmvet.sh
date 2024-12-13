#!/bin/bash


SLM=phi_35
model_name=PiXLLaVAPhi35-v2-3b-merged
VIT=siglip
CONV_MODE=phi3

MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python -m pixl.eval.model_vqa \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet_roi.jsonl \
    --image-folder ./playground/data/eval/mm-vet/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$model_name.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$model_name.json

