#!/bin/bash

# install packages and load in editable mode
pip install -e .

# download data
python download_data.py pretrain_data
python download_data.py finetune_data # takes 1-2 hours

# get base model
bash scripts/pixllava/get_base_model.sh

# pretrain
bash scripts/pixllava/pretrain.sh

# finetune
bash scripts/pixllava/finetune.sh