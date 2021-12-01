# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午3:09
@file: build.py
@author: zj
@description: 
"""

from .fashionmnist import FashionMNIST


def build_dataset(cfg, transform=None, target_transform=None, is_train=True, **kwargs):
    dataset_name = cfg.DATASET.NAME
    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT
    top_k = cfg.DATASET.TOP_K

    if dataset_name == 'FashionMNIST':
        dataset = FashionMNIST(data_root, train=is_train, transform=transform, target_transform=target_transform,
                               top_k=top_k)
    else:
        raise ValueError(f"the dataset {dataset_name} does not exist")

    return dataset
