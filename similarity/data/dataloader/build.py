# -*- coding: utf-8 -*-

"""
@date: 2021/3/31 上午11:25
@file: build.py
@author: zj
@description: 
"""

from torch.utils.data import DataLoader
from zcls.data.datasets.mp_dataset import MPDataset

from ..samplers.build import build_sampler


def build_dataloader(cfg, dataset, is_train=True):
    # batch_size = cfg.DATALOADER.TRAIN_BATCH_SIZE if is_train else cfg.DATALOADER.TEST_BATCH_SIZE
    if is_train:
        batch_size = cfg.SIMILARITY.LABELS_PER_BATCH * cfg.SIMILARITY.SAMPLES_PER_LABEL
    else:
        batch_size = cfg.DATALOADER.TEST_BATCH_SIZE

    sampler = None if isinstance(dataset, MPDataset) else build_sampler(cfg, dataset, is_train)
    data_loader = DataLoader(dataset,
                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                             sampler=sampler,
                             batch_size=batch_size,
                             drop_last=is_train,
                             # [When to set pin_memory to true?](https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723)
                             pin_memory=True)

    return data_loader
