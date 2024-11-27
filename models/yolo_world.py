import sys
import os

import supervision as sv
import numpy as np
import torch
import PIL.Image
import cv2

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmyolo.registry import RUNNERS

from torch.cuda.amp import autocast
from torchvision.ops import nms


class YOLOWorldModel:
    def __init__(
            self,
            config_path='ckpts/YOLO-World/configs/pretrain/yolo_world_xl_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py',
            model_weights_path='ckpts/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain.pth',
            device='cuda:0'
        ):
        super().__init__()
        self.cfg = Config.fromfile(config_path)
        self.cfg.work_dir = "."
        self.cfg.load_from = model_weights_path

        if 'runner_type' not in self.cfg:
            self.runner = Runner.from_cfg(self.cfg)
        else:
            self.runner = RUNNERS.build(self.cfg)

        self.runner.call_hook('before_run')
        self.runner.load_or_resume()
        pipeline = self.cfg.test_dataloader.dataset.pipeline
        self.runner.pipeline = Compose(pipeline)
        self.runner.model.eval()

        self.runner.model.to(device)

        # Annotators for visualization
        self.bounding_box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        
        self.device = device

    def filter_detections(self, output, pred_instances, max_num_boxes, score_thr, nms_thr):
        # Apply Non-Maximum Suppression (NMS) and filter detections
        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

        if len(pred_instances.scores) > max_num_boxes:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]
        output.pred_instances = pred_instances

        pred_instances = pred_instances.cpu().numpy()
        detections = sv.Detections(
            xyxy=pred_instances['bboxes'],
            class_id=pred_instances['labels'],
            confidence=pred_instances['scores']
        )

        # Sort bounding box coordinates by confidence scores
        sorted_indices = np.argsort(-detections.confidence)
        sorted_bboxes = detections.xyxy[sorted_indices]

        coords = [list(bbox) for bbox in sorted_bboxes]

        return detections, coords

    def forward_batch(self, classes_batch, img_path_batch, max_num_boxes=20, score_thr=0.05, nms_thr=0.5):
        batch_data = []

        for img_id, (classes, img_path) in enumerate(zip(classes_batch, img_path_batch)):
            texts = [[t.strip()] for t in classes.split(",")] + [[" "]]
            data_info = self.runner.pipeline(dict(img_id=img_id, img_path=img_path, texts=texts))

            batch_data.append({
                "inputs": data_info["inputs"],
                "data_samples": data_info["data_samples"],
                "texts": texts,
            }),

        inputs_batch = torch.stack([data["inputs"] for data in batch_data]).to(self.device)
        data_samples_batch = [data["data_samples"] for data in batch_data]

        data_batch = dict(
            inputs=inputs_batch,
            data_samples=data_samples_batch,
        )

        with autocast(enabled=False), torch.no_grad():
            outputs = self.runner.model.test_step(data_batch)

        detections_batch = []
        bboxes_batch = []
        for idx, output in enumerate(outputs):
            texts = batch_data[idx]["texts"]
            self.runner.model.class_names = texts
            pred_instances = output.pred_instances

            detections, coords = self.filter_detections(output, pred_instances, max_num_boxes, score_thr, nms_thr)

            detections_batch.append(detections)
            bboxes_batch.append(coords)
            
        return detections_batch, bboxes_batch

    def forward(self, classes, img_path, max_num_boxes=20, score_thr=0.05, nms_thr=0.5):
        texts = [[t.strip()] for t in classes.split(",")] + [[" "]]
        print(f'texts: {texts}')
        data_info = self.runner.pipeline(dict(img_id=0, img_path=img_path, texts=texts))

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0).to(self.device),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]
            self.runner.model.class_names = texts
            pred_instances = output.pred_instances

        detections, coords = self.filter_detections(output, pred_instances, max_num_boxes, score_thr, nms_thr)

        return detections, coords

    def visualize(self, img_path, detections):
        labels = [
            f"{class_id} {confidence:0.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        image = cv2.imread(img_path)
        svimage = np.array(image)
        svimage = self.bounding_box_annotator.annotate(svimage, detections)
        svimage = self.label_annotator.annotate(svimage, detections, labels)

        sv.plot_image(svimage, (10, 10))

    def __call__(self, classes, img_path, max_num_boxes=20, score_thr=0.05, nms_thr=0.5):
        if isinstance(classes, list) and isinstance(img_path, list):
            return self.forward_batch(classes, img_path, max_num_boxes, score_thr, nms_thr)
        else:
            return self.forward(classes, img_path, max_num_boxes, score_thr, nms_thr)
