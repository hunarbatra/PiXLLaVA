import os
import torch
import asyncio

from PIL import Image

from models.yolo_world import YOLOWorldModel
from models.ram_plus import RAMPlus
from models.tag_filtering import process_tags_batch


class ROISelection:
    def __init__(self, image_size=384, device='cuda:0', llm='openai'):
        self.yolo_world_model = YOLOWorldModel(device=device)
        self.ram_model = RAMPlus(image_size=image_size, device=device)
        self.llm = llm
        
    def forward(self, images: list[str] | str | Image.Image, prompts: list[str] | str, max_num_boxes:int=20, score_thr:float=0.05, nms_thr:float=0.5): # handles both, batches, and single images
        # Run RAM++ model to get image tags
        torch.cuda.empty_cache()
        
        tags = self.ram_model(images) 
        if len(tags) and not isinstance(tags[0], list): # handle the case for a single image string
            tags = [tags]
        ram_tags = [', '.join(tag_list) for tag_list in tags] # get tags string for LLM processing

        if not isinstance(prompts, list):
            prompts = [prompts]
            
        # Run gpt-4o-mini or claude-3.5-haiku-20241022 for filtering tags list
        llm_batch = [{'tags': tag, 'prompt': p} for tag, p in zip(ram_tags, prompts)]
        tags = process_tags_batch(llm_batch, model=self.llm) # filtered tags

        # filtered_tags = []
        # for t, p in zip(ram_tags, prompts):
        #     tags = filter_tags(t, p)
        #     filtered_tags.append(tags)
        #     print(f'filtered_tags: {tags}')
        # tags = filtered_tags

        tags_list = [tag_str.split(', ') for tag_str in tags if tag_str is not None]
        max_length = max(len(tag_seq) for tag_seq in tags_list)

        padded_tags = [
            tag_seq + ['<pad>'] * (max_length - len(tag_seq))
            for tag_seq in tags_list
        ]
        padded_str = [', '.join(pad_seq) for pad_seq in padded_tags]

        if not isinstance(images, list):
            images = [images]
            
        # run YOLO-World model for detecting ROI bboxes for selected tags
        yolo_detections, coords = self.yolo_world_model(padded_str, images)
        
        # coords = []
        # for t, img in zip(tags, images):
        #     yolo_detections, bboxes = self.yolo_world_model(t, img)
        #     print(f'bboxes: {bboxes}')
        #     coords.append(bboxes)
        
        return ram_tags, tags, coords
    
    def __call__(self, images: list[str] | str, prompts: list[str] | str, max_num_boxes:int=20, score_thr:float=0.05, nms_thr:float=0.5):
        return self.forward(images, prompts, max_num_boxes, score_thr, nms_thr)
