#!/bin/bash

SLM=phi_35
model_name=PiXLLaVAPhi35-v2-3b-merged
VIT=siglip
CONV_MODE=phi3

MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name


python -m pixl.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions_roi.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE

# model_name=llava-v1.5-13b

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$model_name.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$model_name.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$model_name.jsonl