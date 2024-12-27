#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


SPLIT="llava_gqa_testdev_balanced_roi"
GQADIR="./playground/data/eval/gqa/data"

#model_name=PIXLGemma-v0-2b-finetune
model_name=PiXLLaVAPhi35-3b-merged
SLM=phi_35
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

CKPT=$MODEL_NAME

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m pixl.eval.model_vqa_loader \
        --model-path ./ckpts/checkpoints-$VIT/$SLM/$model_name \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# CKPT=llava-v1.5-13b
# model_name=llava-v1.5-13b
# CHUNKS=${#GPULIST[@]}

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python3 ./scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced
