#!/bin/bash

# install packages and load in editable mode
pip install packaging wheel torch
pip install -e .
pip install flash-attn --no-cache-dir --no-build-isolation

# download vision encoder
python download_data.py download_siglip
# python download_data.py download_phi2

# download RAM plus model for preprocessing/inference
python download_data.py download_ram_plus
pip install git+https://github.com/hunarbatra/recognize-anything.git

# download YOLO World model for preprocessing/inference
python download_data.py download_yolo_world
cd ckpts/YOLO-World
pip install -e .
pip install -r requirements.txt
mim install "mmcv==2.0.1"
cd ../..

# download data
python download_data.py pretrain_data 
python download_data.py finetune_data 

# download base model
python download_data.py phi35
# get base model
bash scripts/pixllava/phiv35/get_base_model.sh
# pretrain
bash scripts/pixllava/phiv35/pretrain.sh
# finetune
bash scripts/pixllava/phiv35/finetune.sh

# Steps for:
# train PIXLLaVA with LLaMA-3.1-8B-Instruct
python3 download_data.py llama31_8b
bash scripts/pixllava/llamav31/get_base_model.sh
bash scripts/pixllava/llamav31/pretrain.sh
bash scripts/pixllava/llamav31/finetune.sh

# train PIXLLaVA with LLaMA-3.2-3B-Instruct
python3 download_data.py llama32_3b
bash scripts/pixllava/llamav32/get_base_model.sh
bash scripts/pixllava/llamav32/pretrain.sh
bash scripts/pixllava/llamav32/finetune.sh

# train PIXLLaVA with LLaMA-2-7B-Instruct
python3 download_data.py llama2_7b
bash scripts/pixllava/llamav2/get_base_model.sh
bash scripts/pixllava/llamav2/pretrain.sh
bash scripts/pixllava/llamav2/finetune.sh

# train PIXLLaVA with Phi-3.5-mini-instruct
python3 download_data.py phi35
bash scripts/pixllava/phiv35/get_base_model.sh
bash scripts/pixllava/phiv35/pretrain.sh
bash scripts/pixllava/phiv35/finetune.sh

# train PIXLLaVA with Phi-3-mini-instruct
python3 download_data.py phi3
bash scripts/pixllava/phiv3/get_base_model.sh
bash scripts/pixllava/phiv3/pretrain.sh
bash scripts/pixllava/phiv3/finetune.sh

# train PIXLLaVA with Phi-2
python3 download_data.py phi2
bash scripts/pixllava/phiv2/get_base_model.sh
bash scripts/pixllava/phiv2/pretrain.sh
bash scripts/pixllava/phiv2/finetune.sh
