import torch
import numpy as np
import os

from ram.models import ram_plus
from ram import get_transform
from ram import inference_ram as inference

from PIL import Image


ram_path = 'ckpts/ram-plus/ram_plus_swin_large_14m.pth'

class RAMPlus:
    def __init__(self, image_size=384, device='cuda:0'):
        super().__init__()
        self.ram_transform = get_transform(image_size=image_size)
        ram_model = ram_plus(pretrained=ram_path, image_size=image_size, vit='swin_l')
        ram_model.cuda(device)
        ram_model.eval()
        self.ram_model = ram_model
        self.device = device

    def forward_batch(self, images_batch):
        images_pil = [Image.open(img) for img in images_batch]

        with torch.no_grad():
            images = [self.ram_transform(image).cuda(self.device) for image in images_pil]
            images = torch.stack(images)
            res = self.ram_model.generate_batch_tag(images, self.ram_model)

            res = res[0] # english outputs
            all_tags = []
            for r in res:
                tags = r.replace(' |', ',').lower().strip().split(',')
                tags = [tag.strip() for tag in tags]
                all_tags.append(tags)

        return all_tags

    def forward(self, image_file):
        if not isinstance(image_file, Image.Image):
            image = Image.open(image_file)
        else:
            image = image_file
            
        with torch.no_grad():
            image = self.ram_transform(image).unsqueeze(0).cuda(self.device)
            res = inference(image, self.ram_model)
            
        tags = res[0].replace(' |', ',')
        tags = tags.lower()
        tags = tags.strip()
        tags = tags.split(',')
        tags = [tag.strip() for tag in tags]

        return tags

    def __call__(self, image_file):
        if isinstance(image_file, list):
            return self.forward_batch(image_file)
        else:
            return self.forward(image_file)
        