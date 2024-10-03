import os
import json
import fire
import math
import torch
import pandas as pd
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from tqdm.auto import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from huggingface_hub import HfApi


load_dotenv()

hf_token = os.getenv("HF_TOKEN")


class DetrData:
    def __init__(self, device):
        self.device = device
        self.detr_model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )
        self.detr_processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )
        self.detr_model.to(self.device)
        self.detr_model.eval()  # Set model to evaluation mode
        for param in self.detr_model.parameters():
            param.requires_grad = False  # Freeze model parameters

        # # Prepare the model with accelerator
        # self.detr_model = self.accelerator.prepare(self.detr_model)

    def process_batch(self, images):
        # Remove any None images from the list
        none_idx = []
        if any(image is None for image in images):
            none_idx = [i for i, image in enumerate(images) if image is None]
            images = [image for i, image in enumerate(images) if i not in none_idx]

        # Prepare inputs for DETR
        detr_inputs = self.detr_processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            detr_outputs = self.detr_model(**detr_inputs)

        # Post-process DETR outputs to get bounding boxes
        target_sizes = torch.tensor([img.size[::-1] for img in images]).to(self.device)
        results = self.detr_processor.post_process_object_detection(detr_outputs, target_sizes=target_sizes)

        images_bboxes = []
        for result in results:
            if len(result['scores']) > 0:
                sorted_indices = torch.argsort(result['scores'], descending=True)
                sorted_boxes = result['boxes'][sorted_indices].tolist()
                images_bboxes.append(sorted_boxes)
            else:
                images_bboxes.append([])

        # Re-insert empty bounding boxes for None images
        for idx in none_idx:
            images_bboxes.insert(idx, [])

        return images_bboxes


def preprocess_detr_data(
    data_type='pretrain', 
    custom_file_path='', 
    custom_img_path='',
    batch_size=64, 
    resume=False, 
    gpu_index=0, 
    total_gpus=1,
    custom_start_idx=None,
    custom_start_file_idx=None,
):
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine the data path based on the data type
    if data_type == 'pretrain':
        data_path = 'data/llava-pretrain/blip_laion_cc_sbu_558k.json'
    elif data_type == 'finetune':
        data_path = 'data/llava-finetune/llava_v1_5_mix665k.json'
    else:
        if len(custom_file_path):
            if not os.path.exists(custom_file_path):
                raise FileNotFoundError(f'Custom file path {custom_file_path} does not exist')
            else:
                data_path = custom_file_path
        else:
            raise ValueError(
                'Invalid data_type: {} or custom_file_path not defined. '
                'Please provide either "pretrain", "finetune", or a "custom_file_path".'.format(data_type)
            )
    
    images_root = f'data/llava-{data_type}/images' if not custom_img_path else custom_img_path

    # Load the DataFrame
    process_df = pd.read_json(data_path)
    
    json_data_path = data_path.replace('.json', f'_detr_{gpu_index}.json')
    
    if custom_start_idx is not None:
        process_df = process_df.iloc[custom_start_idx:].reset_index(drop=True)
        print(f'Custom start index: {custom_start_idx}')
        
    if custom_start_file_idx is not None:
        json_data_path = data_path.replace('.json', f'_detr_{custom_start_file_idx}.json')
        print(f'Custom start file index: {custom_start_file_idx}')
        
    if 'bboxes' not in process_df.columns:
        process_df['bboxes'] = []
        
    # split the data across GPUs
    total_rows = len(process_df)
    rows_per_gpu = math.ceil(total_rows / total_gpus)
    print(f"Total rows: {total_rows}, rows per GPU: {rows_per_gpu}")
    start = gpu_index * rows_per_gpu
    end = min(start + rows_per_gpu, total_rows)
    subset_df = process_df.iloc[start:end].reset_index(drop=True)
    
    print(f"GPU {gpu_index} handling rows {start} to {end} (total for GPU: {len(subset_df)}) | total in dataset: {total_rows}")
    
    # Determine the starting index for resuming
    start_idx = 0
    if resume:
        mask = (subset_df['image'].isnull()) & (subset_df['bboxes'].isnull())
        if mask.any():
            start_idx = mask.idxmax()
        else:
            print(f"All images have been processed for GPU {gpu_index}. Nothing to resume.")
            return
        print(f'Resuming from row index for GPU {gpu_index}: {start_idx}')
        # Select the subset of the DataFrame to process
        subset_df = subset_df.iloc[start_idx:].reset_index(drop=True)

    detr_model = DetrData(device)

    # Iterate over batches with progress bar
    with tqdm(total=len(subset_df), desc=f"GPU {gpu_index} processing") as pbar:
        # Iterate over the subset of the DataFrame in batches
        for i in range(0, len(subset_df), batch_size):
            batch_df = subset_df.iloc[i:i+batch_size].copy()
            images = []
            for _, row in batch_df.iterrows():
                if row['image']:
                    image_path = os.path.join(images_root, row['image'])
                else: 
                    None
                if image_path:
                    if os.path.exists(image_path):
                        try:
                            image = Image.open(image_path).convert("RGB")
                        except Exception as e:
                            print(f"Error loading image {image_path}: {e}")
                            image = None
                    else:
                        raise FileNotFoundError(f'Image file {image_path} does not exist')
                else:
                    image = None
                images.append(image)
            
            try:
                bboxes = detr_model.process_batch(images)
            except Exception as e:
                print(f"Error processing batch through DETR {i//batch_size}: {e}")
                bboxes = [[] for _ in images]  # Assign empty lists if processing fails

            batch_df['bboxes'] = bboxes
            
            # update the original DataFrame with the bounding boxes
            subset_df.iloc[i:i+batch_size] = batch_df
                        
            # save the data so far to json_path
            subset_df.to_json(json_data_path, orient='records', lines=False)
            
            pbar.update(len(batch_df))
    

def merge_detr_json_files(data_type='pretrain', custom_file_path='', total_files=1):
    if data_type == 'pretrain':
        data_path = 'data/llava-pretrain/blip_laion_cc_sbu_558k.json'
    elif data_type == 'finetune':
        data_path = 'data/llava-finetune/llava_v1_5_mix665k.json'
    else:
        if len(custom_file_path):
            if not os.path.exists(custom_file_path) or not custom_file_path.endswith('.json'):
                raise FileNotFoundError(f'Custom file path {custom_file_path} does not exist or is not a JSON file')
            else:
                data_path = custom_file_path
        else:
            raise ValueError(f'Invalid data_type: {data_type} or custom_file_path not defined. Please provide either "pretrain", "finetune", or a "custom_file_path".')
    
    for i in range(total_files):
        json_file_path = data_path.replace('.json', f'_detr_{i}.json')
        if not os.path.exists(json_file_path):
            print(f"Warning: JSON file {json_file_path} does not exist. Skipping.")
            continue
        
        temp_df = pd.read_json(json_file_path)
        
        if i == 0:
            df = temp_df
        else:
            df = pd.concat([df, temp_df])
        
    # save the merged DataFrame to the final JSON file
    json_data_path = data_path.replace('.json', f'_detr.json')
    df.to_json(json_data_path, orient='records', lines=False)
    
    
def view_json_data(data_type='pretrain', custom_file_path='', gpu_index=0, total_gpus=1):
    if data_type == 'pretrain':
        data_path = 'data/llava-pretrain/blip_laion_cc_sbu_558k.json'
    elif data_type == 'finetune':
        data_path = 'data/llava-finetune/llava_v1_5_mix665k.json'
    else:
        if len(custom_file_path):
            if not os.path.exists(custom_file_path) or not custom_file_path.endswith('.json'):
                raise FileNotFoundError(f'Custom file path {custom_file_path} does not exist or is not a JSON file')
            else:
                data_path = custom_file_path
        else:
            raise ValueError(f'Invalid data_type: {data_type} or custom_file_path not defined. Please provide either "pretrain", "finetune", or a "custom_file_path".')
    
    json_file_path = data_path.replace('.json', f'_detr_{gpu_index}.json')
    if not os.path.exists(json_file_path):
        print(f"Warning: JSON file {json_file_path} does not exist. Skipping.")
        return
    
    df = pd.read_json(json_file_path)
    print(df.head(5))
    
    
def upload_to_hf(data_type='pretrain', custom_file_path='', repo_id='hunarbatra/LLaVA-pretrain558k'):
    if data_type == 'pretrain':
        data_path = 'data/llava-pretrain/blip_laion_cc_sbu_558k_detr.json'
    elif data_type == 'finetune':
        data_path = 'data/llava-finetune/llava_v1_5_mix665k_detr.json'
    else:
        if len(custom_file_path):
            if not os.path.exists(custom_file_path) or not custom_file_path.endswith('.json'):
                raise FileNotFoundError(f'Custom file path {custom_file_path} does not exist or is not a JSON file')
            else:
                data_path = custom_file_path
        else:
            raise ValueError(f'Invalid data_type: {data_type} or custom_file_path not defined. Please provide either "pretrain", "finetune", or a "custom_file_path".')
        
    api = HfApi()
    
    repo_path = data_path.split('/')[-1]
    
    api.upload_file(
        path_or_fileobj=data_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    print(f'Uploaded {data_path} to {repo_id} on Hugging Face Hub')
    

if __name__ == '__main__':
    fire.Fire({
        'detr': preprocess_detr_data,
        'merge': merge_detr_json_files,
        'view': view_json_data,
        'upload': upload_to_hf,
    })
