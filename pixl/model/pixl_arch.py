# Adapted from https://github.com/haotian-liu/LLaVA / Copyright 2023 Haotian Liu


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .multimodal_encoder.clip_encoder import CLIPVisionTower
from .multimodal_encoder.siglip_encoder import SiglipVisionTower
from .multimodal_encoder.dinov2_encoder import Dinov2VisionTower
from .multimodal_projector.builder import build_vision_projector
from .language_model.configuration_pixl import PIXLVisionConfig, ProjectorConfig
from pixl.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN


class PIXLMetaModel:
    def __init__(self, config):
        super(PIXLMetaModel, self).__init__(config)
        if "clip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
            self.vision_tower = CLIPVisionTower(
                PIXLVisionConfig(**config.vision_config["vision_tower"])
            )
        elif "siglip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
            self.vision_tower = SiglipVisionTower(
                PIXLVisionConfig(**config.vision_config["vision_tower"])
            )
        elif "dinov2" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
            self.vision_tower = Dinov2VisionTower(
                PIXLVisionConfig(**config.vision_config["vision_tower"])
            )
        else:
            raise ValueError(
                "Vision model name or path should contain either 'clip' or 'siglip'"
            )

        self.mm_projector = build_vision_projector(
            ProjectorConfig(**config.vision_config["mm_projector"])
        )
        
        self.hidden_size = config.vision_config['mm_projector']['mm_hidden_size'] # 1152
        self.proj_hidden_size = config.vision_config['mm_projector']['hidden_size'] # 2560
        self.num_heads = config.vision_config['vision_tower']['num_attention_heads']  # 16
        
        # ensure hidden_size is divisible by num_heads
        assert self.hidden_size % self.num_heads == 0, f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
        
        self.layer_norm_eps = config.vision_config['vision_tower'].get('layer_norm_eps', 1e-6)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps) # 1152
        self.proj_layer_norm = nn.LayerNorm(self.proj_hidden_size, eps=self.layer_norm_eps) # 2560
        
        self.bbox_embedder = nn.Sequential(
            nn.Linear(4, self.proj_hidden_size), # [4, 2560]: d_{bbox} -> d_{LLM_hidden_dim}
            self.proj_layer_norm # [2560] - add post layer norm to normalize the bbox coords across samples
        )
        
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True  # uses (batch, seq, feature) for input and output tensors if set to true
        )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def meanPooling2D(self, x):
        batch_size, num_positions, hidden_size = x.size()
        spatial_dim = int(num_positions ** 0.5) # 27x27 for 729 -> 13x13 for 169
        
        # reshape to [batch_size, 13, 13, 2560 (hidden_size)]
        x_reshaped = x.view(batch_size, spatial_dim, spatial_dim, hidden_size)
        
        # apply 2D pooling with kernel_size=2 and stride=2 to halve the spatial dimensions
        # mean pooling will reduce [27, 27] to [13, 13]
        pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        x_pooled = pooling(x_reshaped.permute(0, 3, 1, 2)) # permute to [batch_size, hidden_size, 13, 13] to get [n, channels, h, w] for 2D Avg Pooling
        
        # reshape back to [batch_size, 169, hidden_size] after pooling
        new_spatial_dim = x_pooled.size(2)
        x_pooled = x_pooled.permute(0, 2, 3, 1).view(batch_size, new_spatial_dim * new_spatial_dim, hidden_size)
        
        return x_pooled
    
    def attentionMeanPooling2D(self, x):
        # Adapted from Molmo's implementation: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/molmo.py
        
        batch_size, num_positions, hidden_size = x.size()
        spatial_dim = int(num_positions ** 0.5)  # For 729 positions, spatial_dim = 27

        # Reshape to [batch_size, spatial_dim, spatial_dim, hidden_size]
        x = x.view(batch_size, spatial_dim, spatial_dim, hidden_size)

        # Pad if spatial dimensions are odd to make them even (required for 2x2 pooling)
        if spatial_dim % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1, 0, 1))  # Pad height and width -- adds 1 padding to the bottom and right of the spatial dumensions, making spatial dim even (28) to ensure downsampling with 2x2 pooling without data loss
            spatial_dim += 1  # Increase spatial_dim since we've padded

        # Rearrange to group patches into 2x2 regions
        # Resulting shape: [batch_size, h, w, 4, hidden_size], where h and w are halved
        x = rearrange(x, 'b (h dh) (w dw) c -> b h w (dh dw) c', dh=2, dw=2)

        # Flatten the spatial dimensions and combine batch size for efficient computation
        num_groups = (spatial_dim // 2) * (spatial_dim // 2)
        x = x.view(batch_size * num_groups, 4, hidden_size)  # [batch_size * num_groups, 4, hidden_size]

        # Compute the query as the mean over the patches in each group - each group has 2x2 patches 
        # we downsample 2x2 patch to 1x1 reducing height and width by half
        query = x.mean(dim=1, keepdim=True)  # [batch_size * num_groups, 1, hidden_size]

        # Apply multi-head attention
        # Note: nn.MultiheadAttention expects inputs in shape [batch_size, seq_len, embed_dim] when batch_first=True
        attn_output, attn_weights = self.attention_pool(query, x, x)
        # attn_output shape: [batch_size * num_groups, 1, hidden_size]

        # Reshape back to [batch_size, h, w, hidden_size]
        h_w = spatial_dim // 2
        attn_output = attn_output.view(batch_size, h_w, h_w, hidden_size)

        # apply layer normalization
        attn_output = self.layer_norm(attn_output) # let's not add this now to closely match molmo's implementation

        # Flatten spatial dimensions to match the expected output shape
        output = attn_output.view(batch_size, -1, hidden_size)  # [batch_size, new_num_positions=196, hidden_size]
        
        return output
    

class PIXLMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, bbox_coords_per_sample):
        # images: now a List of stacked tensors per sample, each element in the list of shape [num_images, channels, height, width]
        # bbox_coords_per_sample: now a List of stacked tensors per sample, each of shape [num_images-1, 4] (where the first image doesn't have bbox coords)
        
        # flatten images and encode them
        flat_images = torch.cat(images, dim=0) # shape: [total_num_images, channels, height, width]
        flat_image_features = self.get_model().get_vision_tower()(flat_images) # shape: [total_num_images, 729, 1152]
        
        # Apply pooling operation to projected features to get 27x27 patches down to 196 patches i.e 1/4 pooling strategy to halve the spatial dimensions (width and height)
        flat_pooled_features = self.get_model().attentionMeanPooling2D(flat_image_features) # shape of each element in the list: [total_num_images, 196, 1152] down from [total_num_images, 729, 1152]

        # Project image features to LLM hidden size (d_llm)
        flat_projected_features = self.get_model().mm_projector(flat_pooled_features) # shape of each element in the list: [total_num_images, 729, 2560]
        
        # Split projected pooled features back into per-sample lists
        projected_features_per_sample = []
        idx = 0
        for img_tensor in images:
            num_images = img_tensor.shape[0]
            projected_features_per_sample.append(flat_projected_features[idx:idx + num_images])
            idx += num_images
            
        # Add scaled bbox embeddings to object crops (excluding the original image) to the LLM projected and pooled features to add in 2D positional information
        for i, (proj_feats, bbox_coords) in enumerate(zip(projected_features_per_sample, bbox_coords_per_sample)):
            if bbox_coords.shape[0] > 1: # there are object crops
                # apply bbox embedding
                bbox_embeddings = self.get_model().bbox_embedder(bbox_coords[1:].to(proj_feats.device)) # shape: [num_crops, 2560] - project d_bbox -> d_llm
                bbox_embeddings = bbox_embeddings.unsqueeze(1) # shape: [num_crops, 1, 2560]
                bbox_embeddings = bbox_embeddings.expand(-1, proj_feats.shape[1], -1) # shape: [num_crops, len(pooled_feats), 2560] i.e [num_crops, 169, 2560]
                proj_feats[1:] += bbox_embeddings # proj_feats shape is: [num_images, 169, 2560], add bbox embeddings projected to d_llm to the projected object crop features
                
        return projected_features_per_sample # list of tensors per sample, each element in the list is a tensor of shape [num_images, 169, 2560] i.e stacked tensors per sample

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, images, bbox_coords_per_sample
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
            
        image_features = self.encode_images(images, bbox_coords_per_sample)
        
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx] # [num_images, 169, 2560]
                num_images = cur_image_features.shape[0]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_new_input_embeds = []
                cur_new_input_embeds.append(cur_input_embeds_1)
                cur_new_input_embeds.extend([img_feat[0:0] for img_feat in cur_image_features])
                cur_new_input_embeds.append(cur_input_embeds_2)
                cur_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    # add embeddings for the text before the image token
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                    # cur_new_input_embeds.append(cur_image_features) # [num_images, 1, hidden_dim]
                    # add embeddings for the image token
                    # append all the image features to the input embeds
                    cur_new_input_embeds.extend([img_feat for img_feat in cur_image_features])
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2])
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start]) # add up the labels upto the image start token
                        # add up image ignore index tokens to not include these in loss calculation
                        cur_new_labels.extend([
                            torch.full((img_feat.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype) 
                            for img_feat in cur_image_features
                        ])
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1]) 
                        cur_labels = cur_labels[image_token_start + 2:] # add up the labels after the image end token
                else:
                    # add embeddings for the text before the image token
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    # cur_new_input_embeds.append(cur_image_features) # [num_images, 1, hidden_dim]
                    # add embeddings for the image token
                    # append all the image features to the input embeds
                    cur_new_input_embeds.extend([img_feat for img_feat in cur_image_features])
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start]) # add up the labels upto the image start token
                        cur_new_labels.extend([
                            torch.full((img_feat.shape[0],), IGNORE_INDEX, device=labels.device,
                                       dtype=labels.dtype) 
                            for img_feat in cur_image_features
                        ]) # add up image ignore index tokens to not include these in loss calculation
                        cur_labels = cur_labels[image_token_start + 1:] # add up the labels after the image start token 
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False): # text input ids
                    # get input ids for the input text (i.e after the image token)
                    cur_input_ids = cur_input_ids[image_token_start + 2:] # 2 to deal with img start and end tokens
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1:] # get input ids for the input text (i.e after the image token)
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0] # get the indices of the image tokens
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end',
                                                                                  False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach()) # add the embeddings for the text input
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids)) # add the embeddings for the text input
                if labels is not None:
                    cur_new_labels.append(cur_labels) # add the labels for the text input
                    
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds] # move all embeds to device
            # Before concatenating, ensure lengths match
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0) # concat up all the embeds in the list - multiple images will easily work here since we appended each image feature to the input embeds separately in the cur_new_input_embeds list
            
            new_input_embeds.append(cur_new_input_embeds)
            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            # if any of the elements shape in inputs is not the same as the first element shape, then we need to pad the inputs to the same shape
            max_len = max(x.shape[0] for x in new_input_embeds) # get the max length of the input embeds

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0) # pad the input embeds to the same length with zeros
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0) # stack the padded input embeds

            if labels is not None: # if labels exist, pad them to the same length with IGNORE_INDEX
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0) # pad the labels to the same length with IGNORE_INDEX
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0) # stack the padded labels

            if attention_mask is not None: # if attention_mask exist, pad it to the same length with 1's (left) and with 0's (right)
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
