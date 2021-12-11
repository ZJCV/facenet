# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午2:55
@file: build.py
@author: zj
@description: 
"""

from zcls.data.transforms.build import build_transform
from zcls.data.datasets.mp_dataset import MPDataset

from .datasets.build import build_dataset
from .dataloader.build import build_dataloader
from .datasets.pk_dataset import PKDataset


def build_data(cfg, is_train=True, **kwargs):
    transform, target_transform = build_transform(cfg, is_train=is_train)
    dataset = build_dataset(cfg, transform=transform, target_transform=target_transform, is_train=is_train, **kwargs)

    return build_dataloader(cfg, dataset, is_train=is_train)


def shuffle_dataset(loader, cur_epoch, is_shuffle=False):
    """"
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
        is_shuffle (bool): need to shuffle the data
    """
    if not is_shuffle:
        return

    dataset = loader.dataset
    if isinstance(dataset, (PKDataset, MPDataset)):
        dataset.set_epoch(cur_epoch)
