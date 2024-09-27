# PiXLLaVA: 

## Add .env file:
```
WANDB_API_KEY=YOUR_WANDB_API_KEY
HF_TOKEN=YOUR_HUGGING_FACE_TOKEN # to upload trained models to HuggingFace Hub
```

### Steps to run:
```
# install packages and load in editable mode
pip install -e .

# download data (96 GB)
python download_data.py pretrain_data
python download_data.py finetune_data # takes 1-2 hours

# init base model
bash scripts/pixllava/get_base_model.sh

# pretrain
bash scripts/pixllava/pretrain.sh

# finetune
bash scripts/pixllava/finetune.sh
```

Or run the script `bash run.sh` to run all the scripts.