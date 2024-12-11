import os
import shutil
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
from datetime import datetime

from models.detr_resnet import DetrModel
from models.roi_selection import ROISelection


load_dotenv()


def remove_yolo_folders(base_path='./'):
    current_year = str(datetime.now().year)
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith(current_year):
            print(f"Removing folder: {item_path}")
            shutil.rmtree(item_path)

def preprocess_roi_data(
    data_type='pretrain', 
    custom_file_path='', 
    custom_img_path='',
    batch_size=64, 
    resume=False, 
    gpu_index=0, 
    total_gpus=1,
    custom_start_idx=None,
    custom_start_file_idx=None,
    clear_bboxes=False,
    custom_device=None,
    llm='openai'
):
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    if custom_device:
        device = torch.device(f"cuda:{custom_device}")
    print(f"Using device: {device}")

    # Determine the data path based on the data type
    if data_type == 'pretrain':
        data_path = 'data/llava-pretrain/blip_laion_cc_sbu_558k_detr.json'
    elif data_type == 'finetune':
        data_path = 'data/llava-finetune/llava_v1_5_mix665k_detr.json'
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
    print(f'Processing data from {data_path}...')
    
    images_root = f'data/llava-{data_type}/images' if not custom_img_path else custom_img_path

    # Load the DataFrame
    process_df = pd.read_json(data_path)
    
    json_data_path = data_path.replace('.json', f'_roi_{gpu_index}.json')
    
    if custom_start_idx is not None:
        process_df = process_df.iloc[custom_start_idx:].reset_index(drop=True)
        print(f'Custom start index: {custom_start_idx}')
        
    if custom_start_file_idx is not None:
        json_data_path = data_path.replace('.json', f'_roi_{custom_start_file_idx}.json')
        print(f'Custom start file index: {custom_start_file_idx}')
    
    if clear_bboxes and not resume:
        process_df = process_df.drop(columns=['bboxes'])
    if 'ram_tags' not in process_df.columns:
        process_df['ram_tags'] = ['' for _ in range(len(process_df))]
    if 'select_tags' not in process_df.columns:
        process_df['select_tags'] = ['' for _ in range(len(process_df))]
    if 'bboxes' not in process_df.columns:
        process_df['bboxes'] = [[] for _ in range(len(process_df))]
        
    roi_pipeline = ROISelection(device=device, llm=llm)
    
    # split the data across GPUs
    total_rows = len(process_df)
    rows_per_gpu = math.ceil(total_rows / total_gpus)
    print(f"Total rows: {total_rows}, rows per GPU: {rows_per_gpu}")
    start = gpu_index * rows_per_gpu
    end = min(start + rows_per_gpu, total_rows)
    subset_df = process_df.iloc[start:end].reset_index(drop=True)
    
    print(f"GPU {gpu_index} handling rows {start} to {end} (total for GPU: {len(subset_df)}) | total in dataset: {total_rows} | device: {device}")
    print(f'Using LLM: {llm}')
    
    # Determine the starting index for resuming
    start_idx = 0
    if resume:
        mask = (subset_df['select_tags'] == '')
        if mask.any():
            start_idx = mask.idxmax()
        else:
            print(f"All images have been processed for GPU {gpu_index}. Nothing to resume.")
            return
        print(f'Resuming from row index for GPU {gpu_index}: {start_idx}')
        # Select the subset of the DataFrame to process
        subset_df = subset_df.iloc[start_idx:].reset_index(drop=True)

    # Iterate over batches with progress bar
    with tqdm(total=len(subset_df), desc=f"GPU {gpu_index} processing") as pbar:
        # Iterate over the subset of the DataFrame in batches
        for i in range(0, len(subset_df), batch_size):
            batch_df = subset_df.iloc[i:i+batch_size].copy()
            
            # check if all rows in batch_df have 'ram_tags' len > 0, and if it does, then skip it
            if all(len(row['select_tags']) > 0 for _, row in batch_df.iterrows()):
                print(f'All rows in batch {i} have select_tags, skipping...')
                pbar.update(len(batch_df))
                continue
            
            images = []
            prompts = []
            text_only_idx = []
            for batch_row_idx, row in batch_df.iterrows():
                if row['image']:
                    image_path = os.path.join(images_root, row['image'])
                    if os.path.exists(image_path):
                        images.append(image_path)
                        prompts.append(row['conversations'][0]['value'])
                    else:
                        raise FileNotFoundError(f'Image file {image_path} does not exist')
                else: 
                    text_only_idx.append(batch_row_idx)
                    print(f'Text only input, skipping row {i}')
                    continue
                
            try:
                if batch_size != len(text_only_idx):
                    ram_tags, select_tags, bboxes = roi_pipeline(images, prompts)
                    if len(text_only_idx):
                        print(f'Mixed Training Batch with Multimodal and Text inputs detected.')
                else:
                    print(f'Text only inputs detected in the batch')
            except Exception as e:
                if batch_size != len(text_only_idx):
                    print(f"Error processing batch through ROI selection {i//batch_size}: {e}")
                    ram_tags = ['' for _ in images]  # Assign empty lists if processing fails
                    select_tags = ['' for _ in images]
                    bboxes = [[] for _ in images]
            
            if len(text_only_idx) and batch_size != len(text_only_idx): # if we have a mixed batch
                for text_idx in text_only_idx:
                    ram_tags.insert(text_idx, '')
                    select_tags.insert(text_idx, '')
                    bboxes.insert(text_idx, [])

            if batch_size != len(text_only_idx): # for text only cases don't save the data
                batch_df['ram_tags'] = ram_tags
                batch_df['select_tags'] = select_tags
                batch_df['bboxes'] = bboxes
                
                # update the original DataFrame with the bounding boxes
                subset_df.iloc[i:i+batch_size] = batch_df
                            
                # save the data so far with the batch update to json_path
                subset_df.to_json(json_data_path, orient='records', lines=False)
            
            pbar.update(len(batch_df))
            
    remove_yolo_folders() # remove YOLO-World folders

    
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
        data_path = 'data/llava-pretrain/blip_laion_cc_sbu_558k_detr.json'
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

    detr_model = DetrModel(device)

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
                    image_path = None
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
    
def merge_roi_json_files(data_type='pretrain', custom_file_path='', total_files=1):
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
        
    for i in range(total_files):
        json_file_path = data_path.replace('.json', f'_roi_{i}.json')
        if not os.path.exists(json_file_path):
            print(f"Warning: JSON file {json_file_path} does not exist. Skipping.")
            continue
        
        temp_df = pd.read_json(json_file_path)
        
        if i == 0:
            df = temp_df
        else:
            df = pd.concat([df, temp_df])
            
        # remove the temporary file
        # os.remove(json_file_path)
        
    # save the merged DataFrame to the final JSON file
    json_data_path = data_path.replace('.json', f'_roi.json')
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
        data_path = 'data/llava-pretrain/blip_laion_cc_sbu_558k_detr_roi.json'
    elif data_type == 'finetune':
        data_path = 'data/llava-finetune/llava_v1_5_mix665k_detr_roi.json'
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
    
def preprocess_eval_science_qa(device='cuda', llm='openai'):
    data_path = 'playground/data/eval/scienceqa/llava_test_CQM-A.json'
    
    data = json.load(open(data_path, "r"))
    
    remove_yolo_folders() # clean
    roi_pipeline = ROISelection(device=device, llm=llm)
    
    for i, sample in enumerate(tqdm(data, desc=f"Processing data")):
        if 'image' in sample:
            image_path = os.path.join('playground/data/eval/scienceqa/images/test', sample['image'])
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist. Please download the images. Skipping.")
                continue
            
            prompt = sample['conversations'][0]['value']
            
            # check if the image is already processed
            if 'ram_tags' in sample and 'select_tags' in sample and 'bboxes' in sample:
                print(f'Image {image_path} already processed. Skipping.')
                continue
            
            ram_tags, select_tags, bboxes = roi_pipeline(image_path, prompt)
            sample['ram_tags'] = ram_tags
            sample['select_tags'] = select_tags
            # extract current bbox - we're processing single samples here - and convert from float32 to float
            bboxes = [[float(coord) for coord in box] for box in bboxes[0]]
            sample['bboxes'] = bboxes
            
            data[i] = sample
        else:
            print(f'Text only sample detected. Skipping.')
            sample['ram_tags'] = ''
            sample['select_tags'] = ''
            sample['bboxes'] = []
            
            data[i] = sample
            
        # save the data to json_path
        if i % 10 == 0:
            json_data_path = data_path.replace('.json', f'_roi.json')
            with open(json_data_path, 'w') as f:
                json.dump(data, f)
        
    print(f'Preprocessed data saved to {data_path}')
    remove_yolo_folders() # remove YOLO-World folders
    
def preprocess_eval_gqa(device='cuda', llm='openai'):
    data_path = 'playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl'
    json_data_path = data_path.replace('.jsonl', f'_roi.jsonl')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    
    remove_yolo_folders() # clean
    roi_pipeline = ROISelection(device=device, llm=llm)
    
    for i, sample in enumerate(tqdm(data, desc=f"Processing data")):
        if 'image' in sample:
            image_path = os.path.join('playground/data/eval/gqa/data/images', sample['image'])
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist. Please download the images. Skipping.")
                continue
            
            prompt = sample["text"]
            
            # check if the image is already processed
            if 'ram_tags' in sample and 'select_tags' in sample and 'bboxes' in sample:
                print(f'Image {image_path} already processed. Skipping.')
                continue
            
            ram_tags, select_tags, bboxes = roi_pipeline(image_path, prompt)
            sample['ram_tags'] = ram_tags
            sample['select_tags'] = select_tags
            # extract current bbox - we're processing single samples here - and convert from float32 to float
            bboxes = [[float(coord) for coord in box] for box in bboxes[0]]
            sample['bboxes'] = bboxes
            
            data[i] = sample
        else:
            print(f'Text only sample detected. Skipping.')
            sample['ram_tags'] = ''
            sample['select_tags'] = ''
            sample['bboxes'] = []
            
            data[i] = sample
        
        # save the data to json_path
        if i % 10 == 0:
            with open(json_data_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
                
    print(f'Preprocessed data saved to {json_data_path}')
    remove_yolo_folders() # remove YOLO-World folders
    
def preprocess_eval_mmvet(device='cuda', llm='openai'):
    data_path = 'playground/data/eval/mm-vet/llava-mm-vet.jsonl'
    json_data_path = data_path.replace('.jsonl', f'_roi.jsonl')
    
    # data = json.load(open(data_path, "r"))
    # Open the file and read it line by line
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    
    remove_yolo_folders() # clean
    roi_pipeline = ROISelection(device=device, llm=llm)
    
    for i, sample in enumerate(tqdm(data, desc=f"Processing data")):
        if 'image' in sample:
            image_path = os.path.join('playground/data/eval/mm-vet/mm-vet/images', sample['image'])
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist. Please download the images. Skipping.")
                continue
            
            prompt = sample["text"]
            
            # check if the image is already processed
            if 'ram_tags' in sample and 'select_tags' in sample and 'bboxes' in sample:
                print(f'Image {image_path} already processed. Skipping.')
                continue
            
            ram_tags, select_tags, bboxes = roi_pipeline(image_path, prompt)
            sample['ram_tags'] = ram_tags
            sample['select_tags'] = select_tags
            # extract current bbox - we're processing single samples here - and convert from float32 to float
            bboxes = [[float(coord) for coord in box] for box in bboxes[0]]
            sample['bboxes'] = bboxes
            
            data[i] = sample
        else:
            print(f'Text only sample detected. Skipping.')
            sample['ram_tags'] = ''
            sample['select_tags'] = ''
            sample['bboxes'] = []
            
            data[i] = sample
        
        # save the data to json_path
        if i % 10 == 0:
            # with open(json_data_path, 'w') as f:
            #     json.dump(data, f)
            with open(json_data_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
                
    print(f'Preprocessed data saved to {json_data_path}')
    remove_yolo_folders() # remove YOLO-World folders
    
def preprocess_eval_pope(device='cuda', llm='openai'):
    data_path = 'playground/data/eval/pope/llava_pope_test.jsonl'
    json_data_path = data_path.replace('.jsonl', f'_roi.jsonl')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
        
    remove_yolo_folders()
    roi_pipeline = ROISelection(device=device, llm=llm)
    
    for i, sample in enumerate(tqdm(data, desc=f"Processing data")):
        if 'image' in sample:
            image_path = os.path.join('playground/data/eval/pope/images', sample['image'])
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist. Please download the images. Skipping.")
                continue
            
            prompt = sample['text']
            
            # check if the image is already processed
            if 'ram_tags' in sample and 'select_tags' in sample and 'bboxes' in sample:
                print(f'Image {image_path} already processed. Skipping.')
                continue
            
            ram_tags, select_tags, bboxes = roi_pipeline(image_path, prompt)
            sample['ram_tags'] = ram_tags
            sample['select_tags'] = select_tags
            # extract current bbox - we're processing single samples here - and convert from float32 to float
            bboxes = [[float(coord) for coord in box] for box in bboxes[0]]
            sample['bboxes'] = bboxes
            
            data[i] = sample
        else:
            print(f'Text only sample detected. Skipping.')
            sample['ram_tags'] = ''
            sample['select_tags'] = ''
            sample['bboxes'] = []
            
            data[i] = sample
            
        # save the data to json_path
        if i % 10 == 0:
            with open(json_data_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
                    
    print(f'Preprocessed data saved to {json_data_path}')
    remove_yolo_folders() # remove YOLO-World folders
    
def preprocess_eval_textvqa(device='cuda', llm='openai'):
    data_path = 'playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl'
    json_data_path = data_path.replace('.jsonl', f'_roi.jsonl')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
        
    remove_yolo_folders()
    roi_pipeline = ROISelection(device=device, llm=llm)
    
    for i, sample in enumerate(tqdm(data, desc=f"Processing data")):
        if 'image' in sample:
            image_path = os.path.join('playground/data/eval/textvqa/train_images', sample['image'])
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist. Please download the images. Skipping.")
                continue
            
            prompt = sample['text']
            
            # check if the image is already processed
            if 'ram_tags' in sample and 'select_tags' in sample and 'bboxes' in sample:
                print(f'Image {image_path} already processed. Skipping.')
                continue
            
            ram_tags, select_tags, bboxes = roi_pipeline(image_path, prompt)
            sample['ram_tags'] = ram_tags
            sample['select_tags'] = select_tags
            # extract current bbox - we're processing single samples here - and convert from float32 to float
            bboxes = [[float(coord) for coord in box] for box in bboxes[0]]
            sample['bboxes'] = bboxes
            
            data[i] = sample
        else:
            print(f'Text only sample detected. Skipping.')
            sample['ram_tags'] = ''
            sample['select_tags'] = ''
            sample['bboxes'] = []
            
            data[i] = sample
            
        # save the data to json_path
        if i % 10 == 0:
            with open(json_data_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
                    
    print(f'Preprocessed data saved to {json_data_path}')
    remove_yolo_folders() # remove YOLO-World folders
    
def preprocess_eval_mme(device='cuda', llm='openai'):
    data_path = 'playground/data/eval/MME/llava_mme.jsonl'
    json_data_path = data_path.replace('.jsonl', f'_roi.jsonl')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
        
    remove_yolo_folders()
    roi_pipeline = ROISelection(device=device, llm=llm)
    
    for i, sample in enumerate(tqdm(data, desc=f"Processing data")):
        if 'image' in sample:
            image_path = os.path.join('playground/data/eval/MME/MME_Benchmark_release_version/MME_Benchmark', sample['image'])
            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist. Please download the images. Skipping.")
                continue
            
            prompt = sample['text']
            
            # check if the image is already processed
            if 'ram_tags' in sample and 'select_tags' in sample and 'bboxes' in sample:
                print(f'Image {image_path} already processed. Skipping.')
                continue
            
            ram_tags, select_tags, bboxes = roi_pipeline(image_path, prompt)
            sample['ram_tags'] = ram_tags
            sample['select_tags'] = select_tags
            # extract current bbox - we're processing single samples here - and convert from float32 to float
            bboxes = [[float(coord) for coord in box] for box in bboxes[0]]
            sample['bboxes'] = bboxes
            
            data[i] = sample
        else:
            print(f'Text only sample detected. Skipping.')
            sample['ram_tags'] = ''
            sample['select_tags'] = ''
            sample['bboxes'] = []
            
            data[i] = sample
            
        # save the data to json_path
        if i % 10 == 0:
            with open(json_data_path, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
                    
    print(f'Preprocessed data saved to {json_data_path}')
    remove_yolo_folders() # remove YOLO-World folders

if __name__ == '__main__':
    fire.Fire({
        'detr': preprocess_detr_data,
        'merge': merge_detr_json_files,
        'view': view_json_data,
        'upload': upload_to_hf,
        'roi': preprocess_roi_data,
        'roi_merge': merge_roi_json_files,
        'scienceqa': preprocess_eval_science_qa,
        'gqa': preprocess_eval_gqa,
        'mmvet': preprocess_eval_mmvet,
        'pope': preprocess_eval_pope,
        'textvqa': preprocess_eval_textvqa,
        'mme': preprocess_eval_mme,
    })
