#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=PIXLLaVAPhi35-3b-merged

#model_name=PIXLGemma-v0-2b-finetune
model_name=PiXLLaVAPhi35-3b-merged
SLM=phi_35
VIT=siglip
CONV_MODE=phi3
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m pixl.eval.model_vqa_loader \
        --model-path $MODELDIR \
        --question-file ./playground/data/eval/seed_bench/llava-seed-bench_roi-spatial.jsonl \
        --image-folder ./playground/data/eval/seed_bench \
        --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m pixl.eval.model_vqa_loader \
#         --model-path $MODELDIR \
#         --question-file ./playground/data/eval/seed_bench/llava-seed-bench_roi-location.jsonl \
#         --image-folder ./playground/data/eval/seed_bench \
#         --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode $CONV_MODE &
# done

wait

output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# CKPT=llava-v1.5-13b
# model_name=llava-v1.5-13b
# CHUNKS=${#GPULIST[@]}

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ./playground/data/eval/seed_bench/answers_upload/$model_name.jsonl

