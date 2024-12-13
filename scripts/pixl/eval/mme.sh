#!/bin/bash

SLM=phi_35
model_name=PiXLLaVAPhi35-v2-3b-merged
VIT=siglip
CONV_MODE=phi3

MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

CUDA_VISIBLE_DEVICES=1 python -m pixl.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/MME/llava_mme_roi.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark \
    --answers-file ./playground/data/eval/MME/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

# model_name=llava-v1.5-13b

cd ./playground/data/eval/MME
python convert_answer_to_mme.py --experiment $model_name
cd eval_tool
python calculation.py --results_dir answers/$model_name
cd ../../../../..
