import torch

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


class DetrModel:
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