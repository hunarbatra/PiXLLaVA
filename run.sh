#!/bin/bash

# install packages and load in editable mode
pip install -e .
pip uninstall torch torchvision flash-attn -y
pip install flash-attn --no-cache-dir --no-build-isolation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# download data
python download_data.py pretrain_data --json_only=True
python download_data.py finetune_data --json_only=True

# get base model
bash scripts/pixllava/get_base_model.sh

# pretrain
bash scripts/pixllava/pretrain.sh

# finetune
bash scripts/pixllava/finetune.sh