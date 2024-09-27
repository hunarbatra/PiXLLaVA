#!/bin/bash

## vision_encoder
#vision_encoder=openai/clip-vit-large-patch14-336
# vision_encoder=google/siglip-so400m-patch14-384
vision_encoder=./ckpts/siglip-so400m-patch14-384

## gemma
# model_dir=./ckpts/checkpoints-siglip/gemma_2b/MiphaGemma-v0-2b-pretrain
# outputdir=./ckpts/checkpoints-siglip/gemma_2b/MiphaGemma-v0-2b-finetune

## phi2
model_name=PiXLLaVAPhi2-v0-3b
model_dir=./ckpts/checkpoints-siglip/phi_2/${model_name}-pretrain
outputdir=./ckpts/checkpoints-siglip/phi_2/${model_name}-finetune

# create output dir
mkdir -p $outputdir

## Note: to run on certain devices do: deepspeed --master_port XXX --include localhost:0,2 ...

deepspeed --master_port 29600 mipha/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $model_dir \
    --version v0 \
    --data_path ./path/to/data/llava_v1_5_mix665k.json \
    --image_folder ./path/to/data/llava-finetune/data \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $outputdir \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

cp $vision_encoder/preprocessor_config.json  $outputdir
