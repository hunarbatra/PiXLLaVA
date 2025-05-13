#!/bin/bash

# install packages and load in editable mode
pip install packaging wheel torch
pip install -e .
pip install flash-attn --no-cache-dir --no-build-isolation

# download vision encoder
python download_data.py download_siglip
# python download_data.py download_phi2

# download RAM plus model for preprocessing/inference
python download_data.py ram_plus
pip install git+https://github.com/hunarbatra/recognize-anything.git

# download YOLO World model for preprocessing/inference
python download_data.py yolo_world
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
python3 models/merge_weights.py merge_lora --model_path='ckpts/checkpoints-siglip/llama_1/PiXLLaVALlama31-8b' --upload_to_hf=False

# train PIXLLaVA with LLaMA-3.2-3B-Instruct
python3 download_data.py llama32_3b
bash scripts/pixllava/llamav32/get_base_model.sh
bash scripts/pixllava/llamav32/pretrain.sh
bash scripts/pixllava/llamav32/finetune.sh
python3 models/merge_weights.py merge_lora --model_path='ckpts/checkpoints-siglip/llama_3/PiXLLaVALlama32-3b' --upload_to_hf=False

# train PIXLLaVA with LLaMA-2-7B-Instruct
python3 download_data.py llama2_7b
bash scripts/pixllava/llamav2/get_base_model.sh
bash scripts/pixllava/llamav2/pretrain.sh
bash scripts/pixllava/llamav2/finetune.sh
python3 models/merge_weights.py merge_lora --model_path='ckpts/checkpoints-siglip/llama_2/PiXLLaVALlama2-7b' --upload_to_hf=False

# train PIXLLaVA with Phi-3.5-mini-instruct
python3 download_data.py phi35
bash scripts/pixllava/phiv35/get_base_model.sh
bash scripts/pixllava/phiv35/pretrain.sh
bash scripts/pixllava/phiv35/finetune.sh
python3 models/merge_weights.py merge_lora --model_path='ckpts/checkpoints-siglip/phi_35/PiXLLaVAPhi35-3b' --upload_to_hf=False

# train PIXLLaVA with Phi-3-mini-instruct
python3 download_data.py phi3
bash scripts/pixllava/phiv3/get_base_model.sh
bash scripts/pixllava/phiv3/pretrain.sh
bash scripts/pixllava/phiv3/finetune.sh
python3 models/merge_weights.py merge_lora --model_path='ckpts/checkpoints-siglip/phi_3/PiXLLaVAPhi3-3b' --upload_to_hf=False

# train PIXLLaVA with Phi-2
python3 download_data.py phi2
bash scripts/pixllava/phiv2/get_base_model.sh
bash scripts/pixllava/phiv2/pretrain.sh
bash scripts/pixllava/phiv2/finetune.sh
python3 models/merge_weights.py merge_lora --model_path='ckpts/checkpoints-siglip/phi_2/PiXLLaVAPhi2-3b' --upload_to_hf=False



################## RUNPOD ###################
git clone https://github.com/hunarbatra/PiXLLaVA

%%writefile .env
WANDB_API_KEY='**************'
HF_TOKEN='************'

pip install -e .
pip install flash-attn --no-cache-dir --no-build-isolation

python download_data.py pretrain_data 
python download_data.py siglip

python3 download_data.py llama31_8b
bash scripts/pixllava/llamav31/get_base_model.sh
bash scripts/pixllava/llamav31/pretrain.sh


##################### EVALS ########################
# download eval.zip into playground/data/eval: https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view

# download data
python download_data.py eval_data --download_all=True
# OR download the eval data individually
python download_data.py eval_data --download_all=False --download_dataset='vqav2'
python download_data.py eval_data --download_all=False --download_dataset='gqa'
python download_data.py eval_data --download_all=False --download_dataset='viswiz'
python download_data.py eval_data --download_all=False --download_dataset='scienceqa'
python download_data.py eval_data --download_all=False --download_dataset='textvqa'
python download_data.py eval_data --download_all=False --download_dataset='pope'
python download_data.py eval_data --download_all=False --download_dataset='mme'
python download_data.py eval_data --download_all=False --download_dataset='mmbench_cn'
python download_data.py eval_data --download_all=False --download_dataset='mmvet'
python download_data.py eval_data --download_all=False --download_dataset='seed_bench'

############# Inference ###############

# eval PIXLLaVA with Phi-3.5-mini-instruct eg
bash scripts/pixl/eval/sqa.sh phi_35 PiXLLaVA35-v2-3b-merged siglip phi35