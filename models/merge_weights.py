import os
import fire
import glob
import torch
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from huggingface_hub import HfApi
from dotenv import load_dotenv
from peft import PeftModel

from pixl.model import PIXLPhiForCausalLM, PIXLGemmaForCausalLM, PIXLPhi3ForCausalLM, PIXLlamaForCausalLM, PIXLPhi15ForCausalLM

from pixl.model.builder import load_pretrained_model
from pixl.mm_utils import get_model_name_from_path

load_dotenv()


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

def print_model_params_grads(
    model_path: str = './ckpts/checkpoints-siglip/phi_35/PiXLLaVAPhi35-3b-merged',
    device: str = "cuda",
):
    kwargs = {"device_map": device, "torch_dtype": torch.float16}
    
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    
    print(f'loading from {model_path}')
    print(f'model name: {model_name}')
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_path, model_name, flash_attn=True)
    
    for name, param in model.named_parameters():
        print(f'Name: {name}, Requires Grad: {param.requires_grad}')

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
    try: 
        print(f"Loading LoRA fine-tuned model and tokenizer from final checkpoint:{finetune_path}")
        model = PeftModel.from_pretrained(model=pretrain_model, model_id=finetune_path, device_map=device_map, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(finetune_path, use_fast=False, local_files_only=True)
    except:
        checkpoints = glob.glob(f'{finetune_path}/checkpoint-*')
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoints found in {finetune_path}")

        # get the latest checkpoint
        latest_ft_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Loading LoRA fine-tuned model and tokenizer from finetuned checkpoint: {latest_ft_checkpoint}")
        
        model = PeftModel.from_pretrained(model=pretrain_model, model_id=latest_ft_checkpoint, device_map=device_map, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(latest_ft_checkpoint, use_fast=False, local_files_only=True)
        
    # Merge LoRA weights into the base model
    print("Merging LoRA weights into the base model")
    model = model.merge_and_unload()
    print("Convert to FP16")
    model.to(torch.float16)

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
    merged_path: str = './ckpts/checkpoints-siglip/phi_35/PiXLLaVAPhi35-3b-merged-210225'
):  
    print(f'Uploading merged model to Hugging Face Hub...')
    
    HF_TOKEN = os.environ["HF_TOKEN"]
    HF_USERNAME = os.environ["HF_USERNAME"]

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
        "debug": print_model_params_grads,
    })
    