# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午3:03
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

import zcls.util.distributed as du

from .pk_sampler import PKSampler


def build_sampler(cfg, dataset, is_train=True):
    if is_train:
        # targets is a list where the i_th element corresponds to the label of i_th dataset element.
        # This is required for PKSampler to randomly sample from exactly p classes. You will need to
        # construct targets while building your dataset. Some datasets (such as ImageFolder) have a
        # targets attribute with the same format.
        targets = dataset.targets.tolist()

        sampler = PKSampler(targets, cfg.SIMILARITY.LABELS_PER_BATCH, cfg.SIMILARITY.SAMPLES_PER_LABEL)
    else:
        world_size = du.get_world_size()
        num_gpus = cfg.NUM_GPUS
        rank = du.get_rank()

        if num_gpus <= 1:
            if cfg.DATALOADER.RANDOM_SAMPLE:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        else:
            shuffle = cfg.DATALOADER.RANDOM_SAMPLE
            sampler = DistributedSampler(dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle)

    return sampler
