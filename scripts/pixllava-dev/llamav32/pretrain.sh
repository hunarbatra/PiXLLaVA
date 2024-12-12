#!/bin/bash

## vision_encoder
vision_encoder=./ckpts/siglip-so400m-patch14-384

## llama-3.2-3b
model_name=PiXLLaVALlama32-3b-pretrain
model_dir=./ckpts/checkpoints-siglip/base_checkpoints/pixllava_llama32_3b
outputdir=./ckpts/checkpoints-siglip/llama32_3b/$model_name

# create outputdir
mkdir -p $outputdir

cp $vision_encoder/preprocessor_config.json $outputdir

## Note: to run on certain devices do: deepspeed --master_port XXX --include localhost:0,2 ...

# gradient accumulation steps = 1
# per_device_train_batch_size = 32
# num_gpus = 8
# total global batch size = 32 * 1 * 8 = 256

deepspeed --master_port 29600 --include localhost:0,2 pixl/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $model_dir \
    --version plain \
    --data_path ./data/llava-pretrain/blip_laion_cc_sbu_558k_roi.json \
    --image_folder ./data/llava-pretrain/images \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --freeze_backbone True \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $outputdir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
