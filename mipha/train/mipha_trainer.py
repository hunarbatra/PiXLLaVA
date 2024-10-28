import os
import torch
import random

from torch.utils.data import Sampler, BatchSampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional

from torch.utils.data import DataLoader


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks

def get_modality_length_grouped_indices(lengths, batch_size, generator=None):
    # Separate multimodal and text-only samples
    mm_indices = [i for i, l in enumerate(lengths) if l > 0]
    lang_indices = [i for i, l in enumerate(lengths) if l < 0]

    # Shuffle indices if generator is provided, to ensure randomness in sampling order
    if generator:
        g = torch.Generator()
        g.manual_seed(generator.initial_seed())
        random.shuffle(mm_indices, random=g.random)
        random.shuffle(lang_indices, random=g.random)
    else:
        random.shuffle(mm_indices)
        random.shuffle(lang_indices)

    # Create batches of original batch size for each modality
    mm_batches = [mm_indices[i:i + batch_size] for i in range(0, len(mm_indices), batch_size)]
    lang_batches = [lang_indices[i:i + batch_size] for i in range(0, len(lang_indices), batch_size)]
    
    # Combine all multimodal batches followed by all text-only batches
    combined_batches = mm_batches + lang_batches
    print(f'batch size: {batch_size}')
    print(f'len(combined_batches): {len(combined_batches)}')
    print(f'len(combined_batches[0]): {len(combined_batches[0])}')

    return combined_batches


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # Regular length-based grouping without modality consideration
    indices = torch.randperm(len(lengths), generator=generator).tolist()
    megabatch_size = world_size * batch_size
    megabatches = [indices[i:i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, 
                self.batch_size, 
                self.world_size, 
                generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
    
class LengthGroupedBatchSampler(BatchSampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int, 
        lengths: List[int],
        group_by_modality: bool = False,
        generator=None,
    ):
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.group_by_modality = group_by_modality
        self.generator = generator

    def __iter__(self):
        if self.group_by_modality:
            combined_batches = get_modality_length_grouped_indices(
                self.lengths,
                self.batch_size,
                generator=self.generator
            )
            # Keep multimodal and text-only batches strictly separate
            return iter(combined_batches)
        else:
            indices = torch.randperm(len(self.lengths), generator=self.generator).tolist()
            batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
            return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
    

class MiphaTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Custom batch sampler that respects modality grouping
        train_batch_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedBatchSampler(
                batch_size=self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
                # generator=self._get_generator()
            )
        else:
            return super()._get_train_sampler()