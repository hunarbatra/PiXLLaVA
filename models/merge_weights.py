import os
import fire
import torch
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from huggingface_hub import HfApi
from dotenv import load_dotenv
from safetensors.torch import load_file
from peft import PeftModel

from pixl.model import PIXLPhiForCausalLM, PIXLGemmaForCausalLM, PIXLPhi3ForCausalLM, PIXLlamaForCausalLM, PIXLPhi15ForCausalLM

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]


def load_model(
    model_path: str = './ckpts/checkpoints-siglip/phi_35/PiXLLaVAPhi35-3b',
    pretrain_path: str = './ckpts/checkpoints-siglip/phi_35/PiXLLaVAPhi35-3b-pretrain',
    kwargs: dict = {},
    device: str = "cuda",
):
    if "phi3" in model_path.lower() or "phi_3" in model_path.lower() or "phi-3" in model_path.lower():
        print(f'Loading the pretrained model from {pretrain_path}')
        pretrain_model = PIXLPhi3ForCausalLM.from_pretrained(
                pretrain_path, 
                use_safetensors=True,
                **kwargs
            ).to(device)
        
    elif "phi2" in model_path.lower() or "phi-2" in model_path.lower() or "phi_2" in model_path.lower():
        print(f'Loading pretrained model from {pretrain_path}')
        
        pretrain_model = PIXLPhiForCausalLM.from_pretrained(
                pretrain_path, 
                use_safetensors=True,
                **kwargs
            ).to(device)
        
    elif "phi15" in model_path.lower() or "phi-15" in model_path.lower() or "phi_15" in model_path.lower():
        print(f'Loading pretrained model from {pretrain_path}')
        
        pretrain_model = PIXLPhi15ForCausalLM.from_pretrained(
                pretrain_path, 
                use_safetensors=True,
                **kwargs
            ).to(device)
        
    elif "gemma" in model_path.lower():
        print(f'Loading pretrained model from {pretrain_path}')
        
        pretrain_model = PIXLGemmaForCausalLM.from_pretrained(
                pretrain_path, 
                use_safetensors=True,
                **kwargs
            ).to(device)
        
    elif "llama" in model_path.lower():
        print(f'Loading pretrained model from {pretrain_path}')
        
        pretrain_model = PIXLlamaForCausalLM.from_pretrained(
                pretrain_path, 
                use_safetensors=True,
                **kwargs
            ).to(device)
        
    else:
        raise ValueError(f'Unknown model path: {model_path}')
    
    return pretrain_model

def merge_weights_lora(
    model_path: str = './ckpts/checkpoints-siglip/phi_35/PiXLLaVAPhi35-3b',
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map: str = "cuda",
    device: str = "cuda",
    upload_to_hf: bool = False,
    flash_attn: bool = False,
):
    kwargs = {"device_map": device_map}
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16
        
    if flash_attn:
        kwargs['attn_implementation'] = "flash_attention_2"
    
    pretrain_path = f'{model_path}-pretrain'
    finetune_path = f'{model_path}-finetune'
    merged_path = f'{model_path}-merged'
        
    # Load the pretrained model
    pretrain_model = load_model(model_path, pretrain_path, kwargs, device)
    
    # Load the fine-tuned model with the adapter
    print(f"Loading LoRA fine-tuned model from {finetune_path}")
    model = PeftModel.from_pretrained(pretrain_model, finetune_path, device_map=device_map)

    # Merge LoRA weights into the base model
    print("Merging LoRA weights into the base model")
    model = model.merge_and_unload()
    print("Convert to FP16")
    model.to(torch.float16)
            
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_path, use_fast=False)

    # Save the merged model
    print(f"Saving the merged model to {merged_path}")
    model.save_pretrained(merged_path, safe_serialization=False)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to {merged_path}")
    
    # move the preprocessor_config.json file from pretrain_path/finetune_path to merged path
    if os.path.exists(os.path.join(pretrain_path, 'preprocessor_config.json')):
        shutil.copy(os.path.join(pretrain_path, 'preprocessor_config.json'), merged_path)
    elif os.path.exists(os.path.join(finetune_path, 'preprocessor_config.json')):
        shutil.copy(os.path.join(finetune_path, 'preprocessor_config.json'), merged_path)
    
    if upload_to_hf:
        upload_weights(merged_path)
        
        
def merge_weights(
    model_path: str = './ckpts/checkpoints-siglip/phi_35/PiXLLaVAPhi35-3b',
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map: str = "cuda",
    device: str = "cuda",
    upload_to_hf: bool = False,
    flash_attn: bool = False,
):
    kwargs = {"device_map": device_map}
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16
        
    if flash_attn:
        kwargs['attn_implementation'] = "flash_attention_2"
    
    pretrain_path = f'{model_path}-pretrain'
    finetune_path = f'{model_path}-finetune'
    merged_path = f'{model_path}-merged'
    
    # Load the pretrained model
    pretrain_model = load_model(model_path, pretrain_path, kwargs, device)
    
    # Load the finetuned model weights
    adapter_file = f"{finetune_path}/adapter_model.safetensors"
    if not torch.load(adapter_file, map_location=device_map):
        print("Falling back to PyTorch model. Adapter Model weights not found!")
        adapter_file = f"{finetune_path}/pytorch_model.bin"
    print(f'Loading finetuned model weights weights from {adapter_file}')
    finetuned_weights = torch.load(adapter_file, map_location=device_map)
    
    # Merge the finetuned model weights into the pretrained model
    print('Merging weights...')
    pretrain_model.load_state_dict(finetuned_weights, strict=False)
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_path, use_fast=False)
    
    # Save the merged model
    pretrain_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    
    print(f'Merged model saved to {merged_path}')
    
    if upload_to_hf:
        upload_weights(merged_path)
        
def upload_weights(
    merged_path: str = './ckpts/checkpoints-siglip/phi_35/PiXLLaVAPhi35-3b-merged'
):  
    print(f'Uploading merged model to Hugging Face Hub...')
    api = HfApi(token=HF_TOKEN)
    repo_name = merged_path.split('/')[-1]
    username = HF_USERNAME
    repo_id = f"{username}/{repo_name}"
    
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=merged_path, 
        repo_id=repo_id, 
        repo_type="model"
    )
    
    print('Uploaded merged model to Hugging Face Hub.')


if __name__ == "__main__":
    fire.Fire({
        "merge_lora": merge_weights_lora,
        "upload": upload_weights,
        "merge": merge_weights,
    })
    