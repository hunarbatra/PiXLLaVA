# Adapted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adapted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li


import os
from dotenv import load_dotenv


load_dotenv()

os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY')
# os.environ["WANDB_MODE"] = "offline"
HF_TOKEN = os.getenv('HF_TOKEN')

import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import sys
import torch
import torch.nn as nn

import transformers

from mipha.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from mipha.train.mipha_trainer import MiphaTrainer

from mipha import conversation as conversation_lib
from mipha.model import *
from mipha.mm_utils import tokenizer_image_token
from transformers import CLIPVisionConfig, SiglipVisionConfig, Dinov2Config, \
    CLIPImageProcessor, SiglipImageProcessor, BitImageProcessor, \
    DetrImageProcessor, DetrForObjectDetection, \
    AutoTokenizer

from PIL import Image
from huggingface_hub import HfApi, Repository, HfFolder


HfFolder.save_token(HF_TOKEN)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")

    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)
    remove_unused_columns: bool = field(default=False)

    freeze_vision_tower: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)

    # freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    non_lora_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    rank0_print(lora_module_names)
    return list(lora_module_names)


def find_all_slm_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    rank0_print(lora_module_names)
    return list(lora_module_names)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        data_args: DataArguments
) -> Dict:
    '''
        prepends the <image> token before the sentence
    '''
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_v0(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(
            conv.sep2)  # in phi-2, pad_token_id == eos_token_id

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # +1 for <|endoftext|>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma_1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1 - 1  # +1 for <eos>, -1 for <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2  # -1 for ' ', -1 for <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) + 1 - 1  # +1 for <eos>, -1 for <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # -1 for ' ', -1 for <bos>

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt().strip())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask targets
    sep = conv.sep + conv.roles[1] + "\n"              # <start_of_turn>model\n
    round_sep ="\n" + conv.sep + conv.roles[0] + "\n"  # \n<start_of_turn>user\n
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(round_sep)
        cur_len = 1
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if i != 0:
                rou = round_sep + rou
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  #  -1 for <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # -1 for <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # -1 for <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # -1 for <bos>

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_plain(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    # print(sources)
    # time.sleep(5)
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    # print(conversations)
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    elif conversation_lib.default_conversation.version.startswith("gemma"):
        return preprocess_gemma_1(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version.startswith("v0"):
        return preprocess_v0(sources, tokenizer, has_image=has_image)
    else:
        raise ValueError(f"Invalid version: {conversation_lib.default_conversation.version}")
    # add end signal and concatenate together


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            # Do not preprocess the image here; pass the PIL image to the data collator
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            
            if self.data_args.image_aspect_ratio == 'pad':
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
            # else if self.data_args.image_aspect_ratio == 'square': then we apply the image_processor which happens by default further in the data collator

                image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            #     image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # else: # self.data_args.image_aspect_ratio == 'square' - default case
            #     # preprocesses image to 384x384 size for the img encoder
            #     image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
        else: # if text only input:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data, add up the PIL image to data_dict
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            try:
                crop_size = self.data_args.image_processor.crop_size
            except:
                crop_size = self.data_args.image_processor.size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        # elif self.data_args.is_multimodal:
        #     # image does not exist in the data, but the model is multimodal
        #     try:
        #         crop_size = self.data_args.image_processor.crop_size
        #     except:
        #         crop_size = self.data_args.image_processor.size
        #     data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, detr_model: DetrForObjectDetection, detr_processor: DetrImageProcessor):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.detr_model = detr_model
        self.detr_processor = detr_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        #print(f"self.tokenizer.pad_token_id: {self.tokenizer.pad_token_id}")
        
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # temp_pad_token_id = 51000
        # pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
            # padding_value=temp_pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            # attention_mask=input_ids.ne(temp_pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances] # PIL Images or torch.zeroes(3, 384, 384) if text only input
            
            images_pil = [img for img in images if type(img) == Image.Image]
            images_tensor_zeros_idx = [idx for idx, img in enumerate(images) if type(img) != Image.Image]
            
            num_images_per_sample = []
            images_per_sample = [] 
            bboxes_per_sample = []
            
            # process images through DETR in batch
            detr_inputs = self.detr_processor(images=images_pil, return_tensors="pt").to(self.detr_model.device)
            
            with torch.no_grad():
                detr_outputs = self.detr_model(**detr_inputs)
                
            # get bounding boxes and scores from detr outputs
            target_sizes = torch.tensor([img.size[::-1] for img in images_pil]) # prepare target sizes for all images
            results = self.detr_processor.post_process_object_detection(detr_outputs, target_sizes=target_sizes) # detection threshold=0.7
            
            for i, (image, result) in enumerate(zip(images_pil, results)):
                curr_img_crops = [image] # original image
                # compute original image bbox coords
                w, h = image.size
                curr_bboxes = [[0.5, 0.5, 1.0, 1.0]] # [[w/2/w, h/2/h, w/w, h/h]] -> [x_center, y_center, width, height] bbox coords for the original img with normalization - these bbox coords do not end up being used for the original image
                
                if len(result['scores']) > 0:
                    sorted_indices = torch.argsort(result['scores'], descending=True)
                    sorted_scores = result['scores'][sorted_indices]
                    sorted_labels = result['labels'][sorted_indices]
                    sorted_boxes = result['boxes'][sorted_indices]
                    
                    max_crops = 9
            
                    for score, label, box in zip(sorted_scores, sorted_labels, sorted_boxes):
                        # convert bbox coords to [x_center, y_center, w, h], and normalize to [0, 1] based on the original image size
                        box = box.tolist()
                        x1, y1, x2, y2 = box
                        
                        # convert to normalized [x_center, y_center, width, height]
                        x_center = (x1 + x2) / 2 / w
                        y_center = (y1 + y2) / 2 / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        bbox = [x_center, y_center, width, height]
                        curr_bboxes.append(bbox)
                        
                        # crop the image
                        crop_box = [int(x1), int(y1), int(x2), int(y2)]
                        image_crop = image.crop(crop_box)
                        curr_img_crops.append(image_crop)
                        
                        if len(curr_img_crops) == max_crops + 1: 
                            # print(f"Truncating image crops to {max_crops} crops. Remaining crops with lower scores not being processed: {len(curr_img_crops) - max_crops}")
                            break
                    
                images_per_sample.append(curr_img_crops)
                bboxes_per_sample.append(torch.tensor(curr_bboxes))
                num_images_per_sample.append(len(curr_img_crops)) # main image + crops count
                
            # print(f"Average number of images per sample for the current Batch: {sum(num_images_per_sample) / len(num_images_per_sample)}")
                
            # flatten the lists
            flat_images = [img for imgs in images_per_sample for img in imgs]
            
            # process images using the image processor in batch
            processor = self.data_args.image_processor
            images_tensor = processor(images=flat_images, return_tensors='pt')['pixel_values'] # shape: [len(flat_images), 3, 384, 384] (where w = 384, h = 384)
            
            # build back the images hierarchy with num_images_per_sample
            idx = 0
            images_per_sample_tensors = []
            for num_images in num_images_per_sample:
                images_per_sample_tensors.append(images_tensor[idx:idx + num_images])
                idx += num_images
                
            # add up the torch_zeros_idx to the images_per_sample at torch_zeros_idx index with the value in images[torch_zeros_idx]
            added_zeros_tensor_ctr = 0
            for idx_to_add in images_tensor_zeros_idx:
                images_per_sample.insert(idx_to_add+added_zeros_tensor_ctr, images[idx_to_add])
                added_zeros_tensor_ctr += 1
                
            # if all(x is not None and x.shape == images[0].shape for x in images):
            #     batch['images'] = torch.stack(images)
            # else:
            #     batch['images'] = images
            
            batch['images'] = images_per_sample_tensors # list of image preprocessed tensors per sample
            batch['bbox_coords'] = bboxes_per_sample # list of normalised bbox coords tensors per sample 
            
            # print(f"bboxes_per_sample_tensors: {bboxes_per_sample}")

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, detr_model: DetrForObjectDetection, detr_processor: DetrImageProcessor) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, 
        data_args=data_args, 
        detr_model=detr_model, 
        detr_processor=detr_processor
    )
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def init_detr_model():
    # Initialize the DETR model and processor
    # print("Initializing DETR model and processor...")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    
    # Set DETR model to evaluation mode and disable gradients
    detr_model.eval()
    for param in detr_model.parameters():
        param.requires_grad = False
    # print("DETR model and processor initialized.")
    
    return detr_model, detr_processor

def train():
    global local_rank
    
    # initialise DETR model and processor before the Main Model / DeepSpeed initialization to avoid DeepSpeed from wrapping the model iniitalization, which leads to size mismatch errors
    detr_model, detr_processor = init_detr_model()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))
    if "phi3" in model_args.model_name_or_path or "phi-3" in model_args.model_name_or_path or "phi_3" in model_args.model_name_or_path:
        config = MiphaPhi3Config.from_pretrained(model_args.model_name_or_path)
        model = MiphaPhi3ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            **bnb_model_from_pretrained_args
        )
    elif "phi2" in model_args.model_name_or_path or "phi-2" in model_args.model_name_or_path or "phi_2" in model_args.model_name_or_path:
        config = MiphaPhiConfig.from_pretrained(model_args.model_name_or_path)
        model = MiphaPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            # attn_implementation="flash_attention_2",
            **bnb_model_from_pretrained_args
        )
    elif "gemma" in model_args.model_name_or_path:
        config = MiphaGemmaConfig.from_pretrained(model_args.model_name_or_path)
        model = MiphaGemmaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            **bnb_model_from_pretrained_args
        )
    else:
        raise ValueError(f"Unknown model: {model_args.model_name_or_path}")
    rank0_print(model)

    model.config.use_cache = False

    model_args.freeze_backbone = training_args.freeze_backbone
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    else:
        model.model.requires_grad_(True)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_slm_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        print(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right"
    )
    if "phi3" in model_args.model_name_or_path or "phi-3" in model_args.model_name_or_path or "phi_3" in model_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
    elif 'phi2' in model_args.model_name_or_path or 'phi-2' in model_args.model_name_or_path or "phi_2" in model_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["phi-2_v0"]
    rank0_print("default_conversation :")
    rank0_print(conversation_lib.default_conversation)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    if "clip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        data_args.image_processor = CLIPImageProcessor.from_pretrained(model_args.model_name_or_path)
    elif "siglip" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        data_args.image_processor = SiglipImageProcessor.from_pretrained(model_args.model_name_or_path)
    elif "dinov2" in config.vision_config["vision_tower"]["vision_model_name_or_path"]:
        data_args.image_processor = BitImageProcessor.from_pretrained(model_args.model_name_or_path)
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio # square
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    # whether to tune the projector layer (vision to llm embedding space)
    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter
    if not model_args.tune_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
    else:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # whether to train the vision encoder model
    model.config.freeze_vision_tower = model_args.freeze_vision_tower = training_args.freeze_vision_tower
    if model_args.freeze_vision_tower:
        for p in model.get_model().vision_tower.parameters():
            p.requires_grad = False
    else:
        for p in model.get_model().vision_tower.parameters():
            p.requires_grad = True

    def calculate_trainable_parameters_percentage(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        percentage = (trainable_params / total_params) * 100
        return ("trainable_params", trainable_params, "total_params", total_params, "percentage", percentage)

    # rank0_print(model)
    # rank0_print(calculate_trainable_parameters_percentage(model))
    # sys.exit(-1)

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.non_lora_lr = training_args.non_lora_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    # print("Building data module with regional crops...")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, detr_model=detr_model, detr_processor=detr_processor)
    # print("Data module built.")

    # Push the model to HuggingFace Hub
    repo_name = training_args.output_dir.split('/')[-1]
    training_args.hub_model_id = repo_name
    training_args.push_to_hub = True
    training_args.hub_private_repo = True
    
    trainer = MiphaTrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
