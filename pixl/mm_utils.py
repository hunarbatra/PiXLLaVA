from PIL import Image
from io import BytesIO
import base64
import os
import shutil
from datetime import datetime

import torch
from transformers import StoppingCriteria
from pixl.constants import IMAGE_TOKEN_INDEX
from PIL import Image

from models.roi_selection import ROISelection


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

def remove_yolo_folders(base_path='./'):
    current_year = str(datetime.now().year)
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith(current_year):
            print(f"Removing folder: {item_path}")
            shutil.rmtree(item_path)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_images(image, bboxes, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    image_crops = []
    bboxes_list = []
        
    # include the original image bbox
    bboxes_list.append([0.5, 0.5, 1.0, 1.0])
    
    image = Image.open(image).convert('RGB')
    
    # pad image to square if needed: default for 'image_aspect_ratio' is 'square'
    if model_cfg.image_aspect_ratio == 'pad':
        background_color = [int(x * 255) for x in model_cfg.image_processor.image_mean]
        image = expand2square(image, background_color)
        
    # create image crops
    image_crops = [image]
    w, h = image.size
    max_crops = 5
    
    for box in bboxes: 
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2 / w # get x center and normalize to image width
        y_center = (y1 + y2) / 2 / h # get y center and normalize to image height
        width = (x2 - x1) / w # get width and normalize to image width
        height = (y2 - y1) / h # get height and normalize to image height
        bbox = [x_center, y_center, width, height]
        bboxes_list.append(bbox)
        
        # crop the image
        crop_box = [int(x1), int(y1), int(x2), int(y2)]
        image_crop = image.crop(crop_box)
        image_crops.append(image_crop)
        
        if len(image_crops) == max_crops + 1:
            break
        
    images_tensors = image_processor(images=image_crops, return_tensors='pt')['pixel_values'] # shape: [len(image_crops), 3, 384, 384]
    bboxes_tensor = torch.tensor(bboxes_list) # shape: [len(image_crops), 4]
        
    return images_tensors, bboxes_tensor
        
def process_images_llava(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
