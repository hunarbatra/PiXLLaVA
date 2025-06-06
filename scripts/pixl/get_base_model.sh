#!/bin/bash

## vision_encoder
vision_encoder=./ckpts/siglip-so400m-patch14-384

## gemma
# base_model_dir=/path/to/google/gemma-2b
# outputdir=./ckpts/checkpoints-siglip/base_checkpoints/pixl_gemma

## phi2
# base_model_dir=/path/to/susnato/phi-2
base_model_dir=./ckpts/phi-2
outputdir=./ckpts/checkpoints-siglip/base_checkpoints/pixllava_phi_2

# create outputdir 
mkdir -p $outputdir

python pixl/train/convert_model2base_pixl.py \
    --model_name_or_path $base_model_dir \
    --version plain \
    --data_path ./data/llava-pretrain/blip_laion_cc_sbu_558k_detr.json \
    --image_folder ./data/llava-pretrain/images \
    --vision_tower $vision_encoder \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $outputdir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --warmup_ratio 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

cp $vision_encoder/preprocessor_config.json  $outputdir