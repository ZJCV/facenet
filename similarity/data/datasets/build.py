# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午3:09
@file: build.py
@author: zj
@description: 
"""

from .fashionmnist import FashionMNIST
from .pk_dataset import PKDataset


def build_dataset(cfg, transform=None, target_transform=None, is_train=True, **kwargs):
    dataset_name = cfg.DATASET.NAME
    data_root = cfg.DATASET.TRAIN_ROOT if is_train else cfg.DATASET.TEST_ROOT
    top_k = cfg.DATASET.TOP_K
    keep_rgb = cfg.DATASET.KEEP_RGB

    if dataset_name == 'FashionMNIST':
        dataset = FashionMNIST(data_root, train=is_train, transform=transform, target_transform=target_transform,
                               top_k=top_k)
    elif dataset_name == 'PKDataset':
        shuffle = cfg.DATALOADER.RANDOM_SAMPLE if is_train else False
        num_gpus = cfg.NUM_GPUS
        rank_id = kwargs.get('rank_id', 0)
        epoch = kwargs.get('epoch', 0)

        labels_per_batch = cfg.SIMILARITY.LABELS_PER_BATCH
        sample_per_label = cfg.SIMILARITY.SAMPLES_PER_LABEL
        num_workers = cfg.DATALOADER.NUM_WORKERS

        dataset = PKDataset(labels_per_batch, sample_per_label, num_workers,
                            data_root, transform=transform, target_transform=target_transform, top_k=top_k,
                            keep_rgb=keep_rgb, shuffle=shuffle, num_gpus=num_gpus, rank_id=rank_id, epoch=epoch,
                            drop_last=is_train)
    else:
        raise ValueError(f"the dataset {dataset_name} does not exist")

    return dataset
