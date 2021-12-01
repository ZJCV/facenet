# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午2:55
@file: build.py
@author: zj
@description: 
"""

from .datasets.build import build_dataset
from zcls.data.transforms.build import build_transform
from .dataloader.build import build_dataloader


def build_data(cfg, is_train=True, **kwargs):
    transform, target_transform = build_transform(cfg, is_train=is_train)
    dataset = build_dataset(cfg, transform=transform, target_transform=target_transform, is_train=is_train, **kwargs)

    return build_dataloader(cfg, dataset, is_train=is_train)
