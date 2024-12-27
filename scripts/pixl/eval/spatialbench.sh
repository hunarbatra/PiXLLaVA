SLM=phi_35
model_name=PiXLLaVAPhi35-3b-merged
VIT=siglip
CONV_MODE=phi3

# VIT=siglip
# CONV_MODE=phi2
# model_name=Mipha-3B_phi2
# SLM=phi_2

MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

python -m pixl.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file ./playground/data/eval/spatial_bench/spatial_bench_roi.jsonl \
    --image-folder ./playground/data/eval/spatial_bench \
    --answers-file ./playground/data/eval/spatial_bench/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE \
    --no-pixl

# model_name=llava-v1.5-13b

python pixl/eval/eval_spatial_bench.py \
    --question-file ./playground/data/eval/spatial_bench/spatial_bench_roi.jsonl \
    --result-file ./playground/data/eval/spatial_bench/answers/$model_name.jsonl