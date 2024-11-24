import torch
import numpy as np
import os

from ram.models import ram_plus
from ram import get_transform
from ram import inference_ram as inference

from PIL import Image


ram_path = 'ckpts/ram-plus/ram_plus_swin_large_14m.pth'

class RAMPlus:
    def __init__(self, ram_path=ram_path, image_size=384, device='cuda:0'):
        self.ram_transform = get_transform(image_size=image_size)
        ram_model = ram_plus(pretrained=ram_path, image_size=image_size, vit='swin_l').to(device)
        ram_model.cuda()
        ram_model.eval()
        self.ram_model = ram_model

    def forward(self, image_file):
        image = Image.open(image_file)
        with torch.no_grad():
            image = self.ram_transform(image).unsqueeze(0).cuda()
            res = inference(image, self.ram_model)
        tags = res[0].replace(' |', ',')
        tags = tags.lower()
        tags = tags.strip()
        tags = tags.split(',')
        tags = [tag.strip() for tag in tags]

        return tags

    def __call__(self, image_file):
        return self.forward(image_file)
