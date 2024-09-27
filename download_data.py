import os
import io
import fire
import requests
import zipfile
import shutil
import json
import urllib.request as ureq

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForZeroShotImageClassification
from datasets import load_dataset
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def download_file(url, local_filename, headers=None):
    """
    Downloads a file from a URL to a local path with a progress bar.
    """
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc=os.path.basename(local_filename),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

def download_and_extract_zip(url, extract_to, temp_zip_path='temp.zip'):
    """
    Downloads a zip file and extracts its contents.
    """
    print(f'Downloading {os.path.basename(temp_zip_path)}...')
    download_file(url, temp_zip_path)
    print(f'Extracting {os.path.basename(temp_zip_path)}...')
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(temp_zip_path)
    print(f'{os.path.basename(temp_zip_path)} extracted to {extract_to}.')
    
def download_pretrain_dataset():
    # dataset = load_dataset("liuhaotian/LLaVA-Pretrain")
    # dataset.save_to_disk("data/")
    
    if os.path.exists('data/llava-pretrain'):
        print('Pretrain dataset already exists. Skipping download.')
        return
    
    os.makedirs('data/llava-pretrain', exist_ok=True)
    # URLs of the files to download
    json_url = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json'
    images_zip_url = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip'

    # Local file paths
    json_file_path = 'data/llava-pretrain/blip_laion_cc_sbu_558k.json'
    images_zip_path = 'data/llava-pretrain/images.zip'
    images_extract_path = 'data/llava-pretrain/images'
    
    headers = None  # or headers = {'Authorization': 'Bearer YOUR_HF_ACCESS_TOKEN'}

    print('Downloading JSON file...')
    if not os.path.exists(json_file_path):
        download_file(json_url, json_file_path, headers=headers)
    print('JSON file downloaded to:', json_file_path)

    print('Downloading images.zip file...')
    if not os.path.exists(images_extract_path) and not os.path.exists(images_zip_path):
        download_file(images_zip_url, images_zip_path, headers=headers)
        with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_extract_path)
        print('images.zip file extracted to:', images_extract_path)
        os.remove(images_zip_path)
    elif os.path.exists(images_extract_path):
        print('images/ folder has already been extracted to:', images_extract_path)
    elif os.path.exists(images_zip_path) and not os.path.exists(images_extract_path):
        print('Extracting images.zip file...')
        with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
            zip_ref.extractall(images_extract_path)
        print('images.zip file extracted to:', images_extract_path)
        os.remove(images_zip_path)
        
    image_count = 0
    for root, dirs, files in os.walk(images_extract_path):
        for file in files:
            if file.endswith('.jpg'):
                image_count += 1
    print(f'Found {image_count} images in {images_extract_path}')
        
    if os.path.exists(images_zip_path):
        os.remove(images_zip_path)
            
    print('images/ downloaded to:', images_extract_path)
    
    
def download_image(k, image_url, output_file):
    try:
        response = requests.get(image_url, stream=True, timeout=30)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img.convert('RGB').save(output_file, 'JPEG')
        return k, None
    except Exception as e:
        return k, f"Error downloading image {k} from {image_url}: {e}"

# Download OCR-VQA images in parallel
def download_ocr_vqa_images(data, ocr_vqa_images_dir):
    print('Downloading OCR-VQA images...')
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for k in data.keys():
            output_file = os.path.join(ocr_vqa_images_dir, f'{k}.jpg')
            image_url = data[k]['imageURL']
            futures.append(executor.submit(download_image, k, image_url, output_file))

        for future in tqdm(futures, desc='OCR-VQA Images'):
            k, error = future.result()
            if error:
                print(error)

def download_finetune_dataset():
    # check if dataset already exists
    if os.path.exists('data/llava-finetune'):
        print('Finetune dataset already exists. Skipping download.')
        return
    
    os.makedirs('data/llava-finetune', exist_ok=True)
    images_root = 'data/llava-finetune/images'
    os.makedirs(images_root, exist_ok=True)

    # URLs of the files to download
    json_url = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json'
    json_file_path = 'data/llava-finetune/llava_v1_5_mix665k.json'

    headers = None  # or headers = {'Authorization': 'Bearer YOUR_HF_ACCESS_TOKEN'}

    # Download the fine-tuning JSON file
    print('Downloading fine-tuning JSON file...')
    if not os.path.exists(json_file_path):
        download_file(json_url, json_file_path, headers=headers)
    print('JSON file downloaded to:', json_file_path)

    # Download and organize datasets
    print('\nStarting dataset downloads and extraction...\n')

    # 1. COCO Dataset
    coco_url = 'http://images.cocodataset.org/zips/train2017.zip'
    coco_extract_to = os.path.join(images_root, 'coco')
    os.makedirs(coco_extract_to, exist_ok=True)
    if not os.path.exists(coco_extract_to) or len(os.listdir(coco_extract_to)) == 0:
        download_and_extract_zip(coco_url, coco_extract_to)

    # 2. GQA Dataset
    gqa_url = 'https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip'
    gqa_extract_to = os.path.join(images_root, 'gqa')
    os.makedirs(gqa_extract_to, exist_ok=True)
    if not os.path.exists(gqa_extract_to) or len(os.listdir(gqa_extract_to)) == 0:
        download_and_extract_zip(gqa_url, gqa_extract_to)

    # # 3. OCR-VQA Dataset - 207572 images
    ocr_vqa_json_path = 'data/llava-finetune/dataset.json' 
    ocr_vqa_images_dir = os.path.join(images_root, 'ocr_vqa/images')
    os.makedirs(ocr_vqa_images_dir, exist_ok=True)
    if not os.path.exists(ocr_vqa_images_dir) or len(os.listdir(ocr_vqa_images_dir)) < 207572:
        print('Loading OCR-VQA dataset JSON from local file...')
        with open(ocr_vqa_json_path, 'r') as fp:
            data = json.load(fp)

        # Parallel downloading of images
        print(f'Downloading OCR-VQA images to {ocr_vqa_images_dir}...')
        download_ocr_vqa_images(data, ocr_vqa_images_dir)
        
    # 4. TextVQA Dataset
    textvqa_url = 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip'
    textvqa_extract_to = os.path.join(images_root, 'textvqa')
    os.makedirs(textvqa_extract_to, exist_ok=True)
    if not os.path.exists(textvqa_extract_to) or len(os.listdir(textvqa_extract_to)) == 0:
        download_and_extract_zip(textvqa_url, textvqa_extract_to)

    # 5. Visual Genome Dataset
    vg_urls = [
        'https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip',
        'https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip'
    ]
    vg_extract_to = os.path.join(images_root, 'vg')
    os.makedirs(vg_extract_to, exist_ok=True)
    for vg_url in vg_urls:
        if not os.path.exists(vg_extract_to) or len(os.listdir(vg_extract_to)) < 2:
            download_and_extract_zip(vg_url, vg_extract_to)

    print('\nAll datasets have been downloaded and organized successfully.')

def move_and_clean_model_directory(model_path):
    # Define the folders to delete
    folders_to_delete = ['.no_exist', 'blobs', 'refs', 'snapshots']

    # Move files from snapshots/* to the model path root
    snapshots_path = os.path.join(model_path, 'snapshots')
    
    # If snapshots exist, move the contents and remove the snapshots directory
    if os.path.exists(snapshots_path):
        for subdir in os.listdir(snapshots_path):
            subdir_path = os.path.join(snapshots_path, subdir)
            if os.path.isdir(subdir_path):
                for file_name in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file_name)
                    shutil.move(file_path, model_path)  # Move file to root model path

    # Delete unnecessary directories
    for folder in folders_to_delete:
        folder_path = os.path.join(model_path, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Delete the directory and its contents
    
    
if __name__ == '__main__':
    fire.Fire({
        'pretrain_data': download_pretrain_dataset, # eg usage: python setup.py pretrain_data
        'finetune_data': download_finetune_dataset,
    })
    