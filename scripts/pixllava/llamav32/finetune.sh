#!/bin/bash

export NCCL_DEBUG=WARN
export DEEPSPEED_LOG_LEVEL=DEBUG

## vision_encoder
vision_encoder=./ckpts/siglip-so400m-patch14-384

## llama-3.2-3b
model_name=PiXLLaVALlama32-v2-3b
model_dir=./ckpts/checkpoints-siglip/llama32_3b/${model_name}-pretrain
outputdir=./ckpts/checkpoints-siglip/llama32_3b/${model_name}-finetune

# create output dir
mkdir -p $outputdir

cp $vision_encoder/preprocessor_config.json $outputdir

## Note: to run on certain devices do: deepspeed --master_port XXX --include localhost:0,2 ...

# gradient accumulation steps = 4
# per_device_train_batch_size = 4
# num_gpus = 8
# total global batch size = 4 * 4 * 8 = 128 

# --lora_enable True --lora_r 128 --lora_alpha 256 \

deepspeed --master_port 29600 pixl/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --model_name_or_path $model_dir \
    --version llama3 \
    --data_path ./data/llava-finetune/llava_v1_5_mix665k_roi.json \
    --image_folder ./data/llava-finetune/images \
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
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --lazy_preprocess True \
    --report_to wandb \
    --push_to_hub False \
    --dataloader_pin_memory True 
