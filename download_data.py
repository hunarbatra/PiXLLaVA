import os
import io
import fire
import requests
import zipfile
import shutil
import json
import urllib.request as ureq
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForZeroShotImageClassification
from datasets import load_dataset
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from hf_transfer import download
from huggingface_hub import hf_hub_download, HfApi, login
    
    
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
HF_KEY = os.getenv('HF_TOKEN')

login(token=HF_KEY)


def download_model_files(repo_id, local_dir, space=False):
    os.makedirs(local_dir, exist_ok=True)
    api = HfApi(token=HF_KEY)
    file_kwargs = {'repo_id': repo_id}
    if space:
        file_kwargs['repo_type'] = 'space'
    files = api.list_repo_files(**file_kwargs)

    for file in files:
        print(f'Downloading {file}...')
        kwargs = {}
        kwargs['repo_id'] = repo_id
        kwargs['filename'] = file
        kwargs['local_dir'] = local_dir
        kwargs['local_dir_use_symlinks'] = False
        if space:
            kwargs['repo_type'] = 'space'

        hf_hub_download(**kwargs)

    print(f'Downloaded {len(files)} files from {repo_id}.')
    
def download_siglip():
    repo_id = "google/siglip-so400m-patch14-384"
    local_dir = "ckpts/" + repo_id.split("/")[-1]
    
    download_model_files(repo_id, local_dir)
    
def download_clip():
    repo_id = "openai/clip-vit-large-patch14-336"
    local_dir = "ckpts/" + "clip-336"
    
    download_model_files(repo_id, local_dir)
    
def download_phi2():
    # repo_id = "susnato/phi-2"
    repo_id = "microsoft/phi-2"
    local_dir = "ckpts/" + repo_id.split("/")[-1]
    
    download_model_files(repo_id, local_dir)
    
def download_phi35():
    repo_id = "microsoft/Phi-3.5-mini-instruct"
    local_dir = "ckpts/phi-35"
    
    download_model_files(repo_id, local_dir)
    
def download_phi3():
    repo_id = "microsoft/Phi-3-mini-4k-instruct"
    local_dir = "ckpts/phi-3"
    
    download_model_files(repo_id, local_dir)
    
def download_mipha3b():
    repo_id = "zhumj34/Mipha-3B"
    local_dir = "ckpts/" + repo_id.split("/")[-1]
    
    download_model_files(repo_id, local_dir)
    
def download_paligemma2_3b():
    repo_id = "google/paligemma2-3b-pt-224"
    local_dir = "ckpts/" + repo_id.split("/")[-1]
    
    download_model_files(repo_id, local_dir)
    
def download_llavaphi3():
    repo_id = "MBZUAI/LLaVA-Phi-3-mini-4k-instruct"
    local_dir = "ckpts/checkpoints-siglip/phi_35/LLaVA-Phi3"
    
    download_model_files(repo_id, local_dir)
    
def download_llavaphi3_ft():
    repo_id = "MBZUAI/LLaVA-Phi-3-mini-4k-instruct-FT"
    local_dir = "ckpts/" + repo_id.split("/")[-1]
    
    download_model_files(repo_id, local_dir)
    
def download_llama3_8b():
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    local_dir = "ckpts/llama3_8b"
    
    download_model_files(repo_id, local_dir)

def download_llama3_1_8b():
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    local_dir = "ckpts/llama31_8b"
    
    download_model_files(repo_id, local_dir)
    
def download_llama3_2_3b():
    repo_id = "meta-llama/Llama-3.2-3B-Instruct"
    local_dir = "ckpts/llama32_3b"
    
    download_model_files(repo_id, local_dir)
    
def download_llama2_7b():
    repo_id = "meta-llama/Llama-2-7b-chat-hf"
    local_dir = "ckpts/llama2_7b"
    
    download_model_files(repo_id, local_dir)
    
def download_vicuna_llama2_7b():
    repo_id = "lmsys/vicuna-7b-v1.5"
    local_dir = "ckpts/vicuna_7b"
    
    download_model_files(repo_id, local_dir)
    
def download_ram_plus():
    repo_id = "xinyu1205/recognize-anything-plus-model"
    local_dir = "ckpts/ram-plus"
    
    download_model_files(repo_id, local_dir)
    
def download_yolo_world():
    # repo_id = "stevengrove/YOLO-World"
    repo_id = "hunarbatra/YOLO-World"
    local_dir = "ckpts/" + repo_id.split('/')[-1]

    download_model_files(repo_id, local_dir, space=True)

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

def download_and_extract_zip(url, extract_to, zip_filename='temp.zip'):
    """
    Downloads a zip file and extracts its contents.
    """
    print(f'Downloading {os.path.basename(zip_filename)}...')
    download_file(url, zip_filename)
    print(f'Extracting {os.path.basename(zip_filename)}...')
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_filename)
    print(f'{os.path.basename(zip_filename)} extracted to {extract_to}.')
    
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
    
def download_pretrain_dataset(json_only=False):
    # dataset = load_dataset("liuhaotian/LLaVA-Pretrain")
    # dataset.save_to_disk("data/")
    
    os.makedirs('data/llava-pretrain', exist_ok=True)
    # URLs of the files to download
    # json_url = 'https://huggingface.co/datasets/hunarbatra/llava-pretrain-558k-object-coords/resolve/main/blip_laion_cc_sbu_558k_detr.json'
    json_url = 'https://huggingface.co/datasets/hunarbatra/PiXLLaVA-pretrain-558k-roi/resolve/main/blip_laion_cc_sbu_558k_detr_roi.json'
    images_zip_url = 'https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip'

    # Local file paths
    json_file_path = 'data/llava-pretrain/blip_laion_cc_sbu_558k_roi.json'
    images_zip_path = 'data/llava-pretrain/images.zip'
    images_extract_path = 'data/llava-pretrain/images'
    
    headers = None  # or headers = {'Authorization': 'Bearer YOUR_HF_ACCESS_TOKEN'}

    print('Downloading JSON file...')
    if not os.path.exists(json_file_path):
        download_file(json_url, json_file_path, headers=headers)
        print('JSON file downloaded to:', json_file_path)
    else:
        print(f'JSON file already exists at {json_file_path}, skipping download.')
    
    if json_only:
        return

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

def download_finetune_dataset(json_only=False, download_all=True, download_dataset=''):
    os.makedirs('data/llava-finetune', exist_ok=True)
    images_root = 'data/llava-finetune/images'
    os.makedirs(images_root, exist_ok=True)

    # URLs of the files to download
    # json_url = 'https://huggingface.co/datasets/hunarbatra/llava-finetune-665k-object-coords/resolve/main/llava_v1_5_mix665k_detr.json'
    # json_url = 'https://huggingface.co/datasets/hunarbatra/PiXLLaVA-finetune-665k-roi/resolve/main/llava_v1_5_mix665k_detr_roi.json'
    json_url = 'https://huggingface.co/datasets/hunarbatra/PiXLLaVA-finetune-665k-roi/resolve/main/llava_v1_5_mix665k_roi.json'
    json_file_path = 'data/llava-finetune/llava_v1_5_mix665k_roi.json'

    # headers = {'Authorization': f'Bearer {os.getenv("HF_TOKEN")}'}

    def download_json():
        print('Downloading fine-tuning JSON file...')
        if not os.path.exists(json_file_path):
            # download_file(json_url, json_file_path, headers=headers)
            download_file(json_url, json_file_path)
            print('JSON file downloaded to:', json_file_path)
        else:
            print(f'JSON file already exists at {json_file_path}, skipping download.')
        
    download_json()
    
    if json_only:
        return

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

    def individual_dataset_download(dataset_name):
        print(f'Downloading single dataset: {dataset_name}...')
        if dataset_name == 'coco':
            download_coco()
        elif dataset_name == 'gqa':
            download_gqa()
        elif dataset_name == 'ocr_vqa':
            download_ocr_vqa()
        elif dataset_name == 'textvqa':
            download_textvqa()
        elif dataset_name == 'vg':
            download_vg()
        else:
            raise ValueError(f'Invalid dataset name: {dataset_name}')
    
    if not download_all and download_dataset:
        individual_dataset_download(download_dataset)
    elif download_all:
        download_tasks = []
        download_tasks.append(download_coco)
        download_tasks.append(download_gqa)
        download_tasks.append(download_ocr_vqa)
        download_tasks.append(download_textvqa) 
        download_tasks.append(download_vg)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            dataset_futures = {executor.submit(task): task.__name__ for task in download_tasks}

            for future in as_completed(dataset_futures):
                task_name = dataset_futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f'Error in {task_name}: {e}')

        print('\nAll datasets have been downloaded and organized successfully.')
    
    
def download_eval_dataset(download_all=True, download_dataset=''):
    root_path = 'playground/data/eval'
    os.makedirs(root_path, exist_ok=True)
    
    download_tasks = []
    
    # 1. VQAv2 Dataset
    def download_vqav2():
        vqav2_url = 'http://images.cocodataset.org/zips/test2015.zip'
        vqav2_extract_to = os.path.join(root_path, 'vqav2')
        os.makedirs(vqav2_extract_to, exist_ok=True)
        
        print('Downloading and extracting VQAv2 dataset...')
        download_and_extract_zip(vqav2_url, vqav2_extract_to, zip_filename='test2015.zip')
        
        print('VQAv2 dataset downloaded and extracted.')
    
    # 2. GQA Dataset
    def download_gqa():
        gqa_url = 'https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip'
        gqa_extract_to = os.path.join(root_path, 'gqa/data')
        os.makedirs(gqa_extract_to, exist_ok=True)
        
        # download images
        print('Downloading and extracting GQA dataset...')
        download_and_extract_zip(gqa_url, gqa_extract_to, zip_filename='images.zip')
        
        # download questions
        gqa_questions_url = 'https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip'
        print('Downloading GQA questions...')
        download_and_extract_zip(gqa_questions_url, gqa_extract_to, zip_filename='questions1.2.zip')
        
        # download scene graphs
        gqa_scene_graphs_url = 'https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip'
        print('Downloading GQA scene graphs...')
        download_and_extract_zip(gqa_scene_graphs_url, gqa_extract_to, zip_filename='sceneGraphs.zip')
        
        # download evaluation script
        gqa_eval_script_url = 'https://nlp.stanford.edu/data/gqa/eval.zip'
        print('Downloading GQA evaluation script...')
        download_and_extract_zip(gqa_eval_script_url, gqa_extract_to, zip_filename='eval.zip')
        
        print('GQA dataset downloaded and extracted.')
    
    # 3. VizWiz Dataset
    def download_vizwiz():
        test_zip_url = 'https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip'
        test_json_url = 'https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip'
        
        vizwiz_extract_to = os.path.join(root_path, 'vizwiz')
        os.makedirs(vizwiz_extract_to, exist_ok=True)
        
        # download test.json
        print('Downloading VizWiz test.json...')
        download_and_extract_zip(test_json_url, os.path.join(vizwiz_extract_to, 'Annotations.zip'))
        
        # download test.zip
        print('Downloading VizWiz test.zip...')
        download_and_extract_zip(test_zip_url, vizwiz_extract_to, zip_filename='test.zip')
        
        print('VizWiz dataset downloaded and extracted.')
    
    # 4. ScienceQA Dataset
    def download_scienceqa():
        pid_splits_url = 'https://raw.githubusercontent.com/lupantech/ScienceQA/refs/heads/main/data/scienceqa/pid_splits.json'
        problems_url = 'https://raw.githubusercontent.com/lupantech/ScienceQA/refs/heads/main/data/scienceqa/problems.json'
        
        sqa_extract_to = os.path.join(root_path, 'scienceqa')
        os.makedirs(sqa_extract_to, exist_ok=True)
        
        # download pid_splits.json
        print('Downloading ScienceQA pid_splits.json...')
        download_file(pid_splits_url, os.path.join(sqa_extract_to, 'pid_splits.json'))
        
        # download problems.json
        print('Downloading ScienceQA problems.json...')
        download_file(problems_url, os.path.join(sqa_extract_to, 'problems.json'))
        
        sqa_images_extract_to = os.path.join(sqa_extract_to, 'images')
        os.makedirs(sqa_images_extract_to, exist_ok=True)
        
        images_url = 'https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip'
        
        # download test.zip
        print('Downloading ScienceQA test.zip...')
        download_and_extract_zip(images_url, sqa_images_extract_to, zip_filename='test.zip')
        
        print('ScienceQA dataset downloaded and extracted.')
    
    # 5. TextVQA Dataset
    def download_textvqa():
        json_url = 'https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json'
        
        textvqa_extract_to = os.path.join(root_path, 'textvqa')
        os.makedirs(textvqa_extract_to, exist_ok=True)
        
        # download test.json
        print('Downloading TextVQA test.json...')
        download_file(json_url, os.path.join(textvqa_extract_to, 'TextVQA_0.5.1_val.json'))
        
        images_url = 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip'
        
        # download train_val_images.zip
        print('Downloading TextVQA train_val_images.zip...')
        download_and_extract_zip(images_url, textvqa_extract_to, zip_filename='train_val_images.zip')
        
        print('TextVQA dataset downloaded and extracted.')
    
    # 6. POPE Dataset
    def download_pope():
        dataset_id = "lmms-lab/POPE"
        subset = "Full"
        
        dataset = load_dataset(dataset_id, subset)
        
        image_dir = os.path.join(root_path, 'pope', 'images')
        os.makedirs(image_dir, exist_ok=True)
        
        def save_images(split, split_name):
            print(f"Saving images for {split_name} split...")
            for i, record in enumerate(split):
                image = record['image']  
                image_source = record['image_source'] 
                file_name = f"{image_source}.jpg"
                file_path = os.path.join(image_dir, file_name)

                try:
                    if isinstance(image, Image.Image):  # If the image is a PIL Image
                        image.save(file_path)
                    else:
                        print(f"Skipping record {i}: Image format not recognized.")
                except Exception as e:
                    print(f"Error saving image for record {i}: {e}")

            print(f"Finished saving images for {split_name} split.")
            
        save_images(dataset['adversarial'], 'adversarial')
        save_images(dataset['popular'], 'popular')
        save_images(dataset['random'], 'random')
        
        json_url_1 = 'https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json'
        json_url_2 = 'https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json'
        json_url_3 = 'https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json'
        
        pope_extract_to = os.path.join(root_path, 'pope/coco')
        os.makedirs(pope_extract_to, exist_ok=True)
        
        # download coco_pope_adversarial.json
        print('Downloading POPE coco_pope_adversarial.json...')
        download_file(json_url_1, os.path.join(pope_extract_to, 'coco_pope_adversarial.json'))
        
        # download coco_pope_popular.json
        print('Downloading POPE coco_pope_popular.json...')
        download_file(json_url_2, os.path.join(pope_extract_to, 'coco_pope_popular.json'))
        
        # download coco_pope_random.json
        print('Downloading POPE coco_pope_random.json...')
        download_file(json_url_3, os.path.join(pope_extract_to, 'coco_pope_random.json'))
        
        print('Pope dataset downloaded and extracted.')
    
    # 7. MME Dataset 
    def download_mme():
        images_url = 'https://huggingface.co/datasets/darkyarding/MME/resolve/main/MME_Benchmark_release_version.zip'
        images_root = os.path.join(root_path, 'MME')
        
        os.makedirs(images_root, exist_ok=True)
        download_and_extract_zip_hf_transfer(
            images_url,
            images_root,
        )
        
        eval_tool_url = 'https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/raw/Evaluation/tools/eval_tool.zip'
        eval_tool_root = os.path.join(root_path, 'MME')
        
        download_and_extract_zip(
            eval_tool_url,
            eval_tool_root,
        )
        
        print(f'MME dataset and eval tool have been downloaded and extracted to {root_path}/MME')
    
    # 8. MMBench-CN Dataset 
    def download_mmbench_cn():
        tsv_url = 'https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv'
        
        mmb_cn_extract_to = os.path.join(root_path, 'mmbench_cn')
        
        # download mmbench_dev_en_20231003.tsv
        print('Downloading MMBench-CN tsv file...')
        download_file(tsv_url, os.path.join(mmb_cn_extract_to, 'mmbench_dev_en_20231003.tsv'))
        
        print('MMBench-CN dataset downloaded and extracted.')
        
    # 11. MMBench Dataset
    def download_mmbench():
        tsv_url = 'https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv'
        
        mmb_extract_to = os.path.join(root_path, 'mmbench')
        
        print('Downloading MMBench tsv file...')
        download_file(tsv_url, os.path.join(mmb_extract_to, 'mmbench_dev_20230712.tsv'))
        
        print('MMBench dataset downloaded and extracted.')
    
    # 9. MMVet Dataset
    def download_mmvet():
        json_url = 'https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip'

        mmv_extract_to = os.path.join(root_path, 'mm-vet')
        os.makedirs(mmv_extract_to, exist_ok=True)
        
        # download mm-vet.zip
        print('Downloading MMVet mm-vet.zip...')
        download_and_extract_zip(json_url, mmv_extract_to, zip_filename='mm-vet.zip')
        
        print('MMVet dataset downloaded and extracted.')
        
    # 10. SEED-Bench Dataset 
    def download_seed_bench():
        images_url = 'https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench-image.zip'
        images_root = os.path.join(root_path, 'seed_bench')
        
        os.makedirs(images_root, exist_ok=True)
        download_and_extract_zip_hf_transfer(
            images_url,
            images_root,
        )
        
        answers_url = 'https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench.json'
        answers_save_path = os.path.join(root_path, 'seed_bench/SEED-Bench.json')
        download_file(answers_url, answers_save_path)
        
        print(f'SEED-Bench dataset has been downloaded and extracted to {root_path}/seed_bench')
        
    # 12. LLaVA-Bench Dataset
    def download_llava_bench():
        dataset_name = 'liuhaotian/llava-bench-in-the-wild'
        dataset = load_dataset(dataset_name)

        image_dir = os.path.join(root_path, 'llava-bench-in-the-wild/images')
        os.makedirs(image_dir, exist_ok=True)

        for i, record in enumerate(dataset['train']):
            image = record['image'].convert('RGB')
            file_idx = f'{i+1}'
            file_idx = file_idx.zfill(3)
            file_path = os.path.join(image_dir, f"{file_idx}.jpg")

            if isinstance(image, Image.Image):
                image.save(file_path)
            else:
                print(f"Skipping record {i}: Image format not recognized.") 
                
        q_json_url = 'https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/questions.jsonl'
        q_json_save_path = os.path.join(root_path, 'llava-bench-in-the-wild/questions.jsonl')
        download_file(q_json_url, q_json_save_path)
        
        context_json_url = 'https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/context.jsonl'
        context_json_save_path = os.path.join(root_path, 'llava-bench-in-the-wild/context.jsonl')
        download_file(context_json_url, context_json_save_path)
        
        answers_gpt4_json_url = 'https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/answers_gpt4.jsonl'
        answers_gpt4_json_path = os.path.join(root_path, 'llava-bench-in-the-wild/answers_gpt4.jsonl')
        download_file(answers_gpt4_json_url, answers_gpt4_json_path)
        
        print(f'LLaVA bench (in the wild) dataset has been downloaded and extracted to {root_path}/llava-bench-in-the-wild')
        
    def download_vstar_bench():
        dataset_name = 'craigwu/vstar_bench'
        dataset = load_dataset(dataset_name)
        
        base_images_path = "https://huggingface.co/datasets/craigwu/vstar_bench/resolve/main/"
        root_path = "playground/data/eval/vstar_bench"
        base_save_path = os.path.join(root_path, "images")
        images_paths = []
        
        os.makedirs(base_save_path, exist_ok=True)
        
        for i, record in enumerate(dataset['test']):
            img_path = record['image']
            
            if img_path.split('.')[-1] == 'json':
                continue
            
            req_path = base_images_path + img_path
            save_path = f"{base_save_path}/{img_path}"
            
            save_root = os.path.dirname(save_path)
            os.makedirs(save_root, exist_ok=True)
            
            download_file(req_path, save_path)
            
        json_url = "https://huggingface.co/datasets/craigwu/vstar_bench/resolve/main/test_questions.jsonl"
        json_path = os.path.join(root_path, "test_questions.jsonl")
        
        download_file(json_url, json_path)
        
        print(f'Vstar bench dataset has been downloaded and extracted to {root_path}')
        
    def download_mmvp():
        root_path = 'playground/data/eval/mmvp'
        os.makedirs(root_path, exist_ok=True)
        
        images_path = os.path.join(root_path, 'images')
        os.makedirs(images_path, exist_ok=True)
        base_img_req_url = 'https://huggingface.co/datasets/MMVP/MMVP/resolve/main/MMVP%20Images/'
        
        for i in range(1, 301):
            req_path = f"{base_img_req_url}{i}.jpg"
            save_path = f"{images_path}/{i}.jpg"
            
            # download_file(req_path, save_path)
            
        csv_url = 'https://huggingface.co/datasets/MMVP/MMVP/resolve/main/Questions.csv'
        csv_path = os.path.join(root_path, 'Questions.csv')
        download_file(csv_url, csv_path)
        
        df = pd.read_csv(csv_path)
        df['image'] = [f"{i+1}.jpg" for i in range(len(df))]
        
        for i, row in df.iterrows():
            options = row['Options']
            options = options.replace('(a)', '(A)')
            options = options.replace('(b)', '(B)')

            options = options.split('(')
            updated_options = ""
            for o in options:
                if o == '':
                    continue
                updated_options += f"({o.strip()}\n"

            correct_ans = row['Correct Answer']
            correct_ans = correct_ans.replace('(a)', 'A')
            correct_ans = correct_ans.replace('(b)', 'B')

            question = row['Question']
            question = question + "\n" + updated_options + "Answer with the option's letter from the given choices directly."

            # Update the DataFrame directly
            df.at[i, 'Correct Answer'] = correct_ans
            df.at[i, 'Question'] = question
            df.at[i, 'Options'] = updated_options

        df.rename(columns={'Index': 'question_id', 'Question': 'text', "Correct Answer": "label"}, inplace=True)
        df.to_json(os.path.join(root_path, 'questions.jsonl'), orient='records', lines=True)
        
        print(f'MMVP dataset has been downloaded and extracted to {root_path}')
        
    def download_spatial_bench():
        base_url = "https://huggingface.co/datasets/RussRobin/SpatialBench/resolve/main"
        root_path = "playground/data/eval/spatial_bench"
        
        os.makedirs(root_path, exist_ok=True)
        
        tasks = ['counting', 'existence', 'positional', 'reach', 'size']
        
        all_data = []
        
        headers = {'Authorization': f'Bearer {os.getenv("HF_TOKEN")}'}
        
        file_count = 0
        
        for task in tasks:
            json_url = f"{base_url}/{task}.json"
            json_path = os.path.join(root_path, f"{task}.json")
            download_file(json_url, json_path, headers=headers)
            
            data = json.load(open(json_path, "r"))
            
            for i, sample in enumerate(data):
                image_path = sample['image']
                img_save_path = os.path.join(root_path, image_path)
        
                img_req_url = f"{base_url}/{image_path}"
                
                img_root_dir = os.path.dirname(img_save_path)
                os.makedirs(img_root_dir, exist_ok=True)
                
                if not os.path.exists(img_save_path):
                    print(f"Downloading image {i+1} of {len(data)} for {task} task...")
                    download_file(img_req_url, img_save_path, headers=headers)
                
                file_count += 1
                sample['category'] = task
                sample['text'] = sample['question']
                sample['question_id'] = file_count
                del sample['question']
                
                if task == 'existence':
                    sample['text'] = sample['text'] + "\nAnswer with Yes or No directly."
                
                all_data.append(sample)
                
            print(f"Downloaded {len(data)} samples for {task} task.")
            
        with open(os.path.join(root_path, "spatial_bench.jsonl"), "w") as f:
            for sample in all_data:
                f.write(json.dumps(sample) + "\n")
                
        print(f"All data has been saved to {os.path.join(root_path, 'spatial_bench.jsonl')}")
            
    
    def individual_dataset_download(dataset_name):
        if dataset_name == 'vqav2':
            download_vqav2()
        elif dataset_name == 'gqa':
            download_gqa()
        elif dataset_name == 'vizwiz':
            download_vizwiz()
        elif dataset_name == 'scienceqa':
            download_scienceqa()
        elif dataset_name == 'textvqa':
            download_textvqa()
        elif dataset_name == 'pope':
            download_pope()
        elif dataset_name == 'mme':
            download_mme()
        elif dataset_name == 'mmbench':
            download_mmbench()
        elif dataset_name == 'mmbench_cn':
            download_mmbench_cn()
        elif dataset_name == 'mmvet':
            download_mmvet()
        elif dataset_name == 'seed_bench':
            download_seed_bench()
        elif dataset_name == 'llava_bench':
            download_llava_bench()
        elif dataset_name == 'vstar_bench':
            download_vstar_bench()
        elif dataset_name == 'mmvp':
            download_mmvp()
        elif dataset_name == 'spatial_bench':
            download_spatial_bench()
        else:
            raise ValueError(f'Invalid dataset name: {dataset_name}. Please select from the following options: vqav2, gqa, vizwiz, scienceqa, textvqa, pope, mme, mmbench, mmbench_cn, mmvet, seed_bench, llava_bench')
        
    if not download_all and download_dataset:
        individual_dataset_download(download_dataset)
    elif download_all:
        download_tasks.append(download_vqav2)
        download_tasks.append(download_gqa)
        download_tasks.append(download_vizwiz)
        download_tasks.append(download_scienceqa)
        download_tasks.append(download_textvqa)
        download_tasks.append(download_pope)
        download_tasks.append(download_mme)
        download_tasks.append(download_mmbench_cn)
        download_tasks.append(download_mmvet)
        download_tasks.append(download_seed_bench)
        download_tasks.append(download_llava_bench)
        download_tasks.append(download_vstar_bench)
    
        with ThreadPoolExecutor(max_workers=10) as executor:
            dataset_futures = {executor.submit(task): task.__name__ for task in download_tasks}

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
        'pretrain_data': download_pretrain_dataset, # eg usage: python download_data.py pretrain_data
        'finetune_data': download_finetune_dataset,
        'eval_data': download_eval_dataset,
        'coco_data_test': coco_data_test,
        'siglip': download_siglip,
        'clip': download_clip,
        'phi2': download_phi2,
        'mipha3b': download_mipha3b,
        'llavaphi3': download_llavaphi3,
        'llavaphi3_ft': download_llavaphi3_ft,
        'ram_plus': download_ram_plus,
        'yolo_world': download_yolo_world,
        'phi35': download_phi35,
        'phi3': download_phi3,
        'llama31_8b': download_llama3_1_8b,
        'llama32_3b': download_llama3_2_3b,
        'llama3_8b': download_llama3_8b,
        'llama2_7b': download_llama2_7b,
        'vicuna_7b': download_vicuna_llama2_7b,
        'paligemma2_3b': download_paligemma2_3b,
    })
    