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
from concurrent.futures import ThreadPoolExecutor, as_completed

from hf_transfer import download
    
    
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

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
    
def download_and_extract_zip_hf_transfer(url, extract_to, zip_filename='temp.zip'):
    """
    Downloads a zip file using hf_transfer and extracts its contents.
    """
    # Define the local path for the zip file
    zip_path = os.path.join(extract_to, zip_filename)

    os.makedirs(extract_to, exist_ok=True)
    
    # Download the file using hf_transfer
    print(f'Downloading {os.path.basename(url)} with hf_transfer...')
    download(url=url, filename=zip_path, chunk_size=10_485_760, max_files=10)  # hf_transfer handles faster downloading
    
    # Extract the zip file
    print(f'Extracting {os.path.basename(zip_path)}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Remove the zip file after extraction
    os.remove(zip_path)
    print(f'{os.path.basename(zip_path)} extracted to {extract_to}.')
    
def download_pretrain_dataset():
    # dataset = load_dataset("liuhaotian/LLaVA-Pretrain")
    # dataset.save_to_disk("data/")
    
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
        download_and_extract_zip_hf_transfer(
            images_zip_url,
            images_extract_path,
        )
        
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
    os.makedirs('data/llava-finetune', exist_ok=True)
    images_root = 'data/llava-finetune/images'
    os.makedirs(images_root, exist_ok=True)

    # URLs of the files to download
    json_url = 'https://huggingface.co/datasets/hunarbatra/LLaVA-finetune665k/resolve/main/llava_v1_5_mix665k.json'
    json_file_path = 'data/llava-finetune/llava_v1_5_mix665k.json'

    headers = {'Authorization': f'Bearer {os.getenv("HF_TOKEN")}'}

    def download_json():
        print('Downloading fine-tuning JSON file...')
        if not os.path.exists(json_file_path):
            download_file(json_url, json_file_path, headers=headers)
        print('JSON file downloaded to:', json_file_path)

    download_tasks = []

    # 1. COCO Dataset
    def download_coco():
        print('Starting Coco download')
        coco_url = 'https://huggingface.co/datasets/BoyangZ/coco_train_2017/resolve/main/train2017.zip'
        coco_extract_to = os.path.join(images_root, 'coco')
        os.makedirs(coco_extract_to, exist_ok=True)
        if not os.path.exists(coco_extract_to) or len(os.listdir(coco_extract_to)) == 0:
            print('Downloading and extracting COCO dataset...')
            download_and_extract_zip_hf_transfer(coco_url, coco_extract_to, zip_filename="coco_temp.zip")
            print('COCO dataset downloaded and extracted.')

    download_tasks.append(download_coco)

    # 2. GQA Dataset
    def download_gqa():
        print('Starting GQA download')
        gqa_url = 'https://huggingface.co/datasets/BoyangZ/GQA_llava/resolve/main/images.zip'
        gqa_extract_to = os.path.join(images_root, 'gqa')
        os.makedirs(gqa_extract_to, exist_ok=True)
        if not os.path.exists(gqa_extract_to) or len(os.listdir(gqa_extract_to)) == 0:
            print('Downloading and extracting GQA dataset...')
            download_and_extract_zip_hf_transfer(gqa_url, gqa_extract_to, zip_filename="cqa_temp.zip")
            print('GQA dataset downloaded and extracted.')

    download_tasks.append(download_gqa)

    # 3. OCR-VQA Dataset
    def download_ocr_vqa():
        print('Starting OCR download')
        ocr_vqa_url = 'https://huggingface.co/datasets/BoyangZ/OCR_VQA/resolve/main/ocr_vqa_images_llava_v15.zip'
        ocr_vqa_extract_to = os.path.join(images_root, 'ocr_vqa')
        os.makedirs(ocr_vqa_extract_to, exist_ok=True)
        if not os.path.exists(ocr_vqa_extract_to) or len(os.listdir(ocr_vqa_extract_to)) == 0:
            print('Downloading and extracting OCR-VQA dataset...')
            download_and_extract_zip_hf_transfer(ocr_vqa_url, ocr_vqa_extract_to, zip_filename="ocr_temp.zip")
            print('OCR-VQA dataset downloaded and extracted.')

    download_tasks.append(download_ocr_vqa)

    # 4. TextVQA Dataset
    def download_textvqa():
        print('Starting TextVQA download')
        textvqa_url = 'https://huggingface.co/datasets/BoyangZ/text_vqa_train_val_images/resolve/main/train_val_images.zip'
        textvqa_extract_to = os.path.join(images_root, 'textvqa')
        os.makedirs(textvqa_extract_to, exist_ok=True)
        if not os.path.exists(textvqa_extract_to) or len(os.listdir(textvqa_extract_to)) == 0:
            print('Downloading and extracting TextVQA dataset...')
            download_and_extract_zip_hf_transfer(textvqa_url, textvqa_extract_to, zip_filename="textvqa_temp.zip")
            print('TextVQA dataset downloaded and extracted.')

    download_tasks.append(download_textvqa)

    # 5. Visual Genome Dataset
    def download_vg():
        print('Starting VG download')
        vg_urls = [
            'https://huggingface.co/datasets/BoyangZ/VisualGenome_VG_100K_1_and_2/resolve/main/images.zip',
            'https://huggingface.co/datasets/BoyangZ/VisualGenome_VG_100K_1_and_2/resolve/main/images2.zip'
        ]
        vg_extract_to = os.path.join(images_root, 'vg')
        os.makedirs(vg_extract_to, exist_ok=True)
        for vg_url in vg_urls:
            if not os.path.exists(vg_extract_to) or len(os.listdir(vg_extract_to)) < 2:
                print(f'Downloading and extracting Visual Genome dataset from {vg_url}...')
                download_and_extract_zip_hf_transfer(vg_url, vg_extract_to, zip_filename="vg_temp.zip")
                print(f'Visual Genome dataset from {vg_url} downloaded and extracted.')

    download_tasks.append(download_vg)

    with ThreadPoolExecutor(max_workers=10) as executor:
        json_future = executor.submit(download_json)

        dataset_futures = {executor.submit(task): task.__name__ for task in download_tasks}

        json_future.result()

        for future in as_completed(dataset_futures):
            task_name = dataset_futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'Error in {task_name}: {e}')

    print('\nAll datasets have been downloaded and organized successfully.')
    
def coco_data_test():
    # Example usage for the COCO dataset
    coco_url = 'https://huggingface.co/datasets/BoyangZ/coco_train_2017/resolve/main/train2017.zip'
    images_root = 'data/llava-finetune/images'
    coco_extract_to = os.path.join(images_root, 'coco')

    os.makedirs(coco_extract_to, exist_ok=True)
    # if not os.path.exists(coco_extract_to) or len(os.listdir(coco_extract_to)) == 0:
    download_and_extract_zip_hf_transfer(coco_url, coco_extract_to, zip_filename='train2017.zip')

    
    
if __name__ == '__main__':
    fire.Fire({
        'pretrain_data': download_pretrain_dataset, # eg usage: python setup.py pretrain_data
        'finetune_data': download_finetune_dataset,
        'coco_data_test': coco_data_test,
    })
    
