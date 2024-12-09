import os
import argparse

from huggingface_hub import HfApi
from dotenv import load_dotenv

from pixl.model.builder import load_pretrained_model
from pixl.mm_utils import get_model_name_from_path


load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
    
    print(f"Merged model saved to {args.save_model_path}")
    
    if args.upload:
        print('Uploading merged model to Hugging Face Hub...')
        api = HfApi(token=HF_TOKEN)
        repo_name = args.save_model_path.split('/')[-1]
        username = HF_USERNAME
        repo_id = f"{username}/{repo_name}"
        print(f'saving merged model to {repo_id}')
        
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=args.save_model_path, 
            repo_id=repo_id, 
            repo_type="model"
        )
        
        print('Uploaded merged model to Hugging Face Hub.')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--upload", type=bool, default=False)

    args = parser.parse_args()

    merge_lora(args)
