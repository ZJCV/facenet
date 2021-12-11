# -*- coding: utf-8 -*-

"""
@date: 2021/12/2 上午11:10
@file: pk_dataset.py
@author: zj
@description: 
"""

from abc import ABC

import os
import random

import numpy as np

import torch
from torch.utils.data import IterableDataset

from zcls.data.datasets.mp_dataset import get_base_info, build_sampler, get_subset_data, shuffle_dataset
from zcls.config.key_word import KEY_DATASET, KEY_CLASSES
from zcls.data.datasets.util import default_loader

from ..samplers.pk_sampler import PKSampler
from .evaluator.verification_evaluator import VerificationEvaluator


class PKDataset(IterableDataset, ABC):

    def __init__(self, labels_per_batch, sample_per_label, num_workers,
                 root, transform=None, target_transform=None, top_k=(1, 5), keep_rgb: bool = False,
                 shuffle: bool = False, num_gpus: int = 1, rank_id: int = 0, epoch: int = 0, drop_last: bool = False):
        self.labels_per_batch = labels_per_batch
        self.sample_per_label = sample_per_label
        self.num_workers = num_workers
        self.is_train = drop_last

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.keep_rgb = keep_rgb
        self.shuffle = shuffle

        self.rank = rank_id
        self.num_replicas = num_gpus

        self.data_path = os.path.join(self.root, KEY_DATASET)
        self.cls_path = os.path.join(self.root, KEY_CLASSES)
        self.classes, self.length = get_base_info(self.cls_path, self.data_path)
        self.sampler = build_sampler(self, num_gpus=self.num_replicas, random_sample=self.shuffle,
                                     rank_id=self.rank, drop_last=drop_last)
        self.set_epoch(epoch)

        self._update_evaluator(top_k)

    def parse_file(self, img_list, label_list):
        for img_path, target in zip(img_list, label_list):
            image = default_loader(img_path, rgb=self.keep_rgb)
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

            yield image, target

    def get_indices(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # in a worker process
            worker_id = worker_info.id
            # split workload
            indices = self.indices[worker_id:self.indices_length:worker_info.num_workers]
        else:
            # single-process data loading, return the full iterator
            indices = self.indices

        return indices

    def __iter__(self):
        indices = self.get_indices()

        print('0')
        img_list, label_list = get_subset_data(self.data_path, indices)
        assert len(img_list) == len(label_list)

        if self.is_train:
            sampler = PKSampler(label_list, self.labels_per_batch, self.sample_per_label)
            sub_indices = list(sampler)

            # 采集剩余的数据下标
            remain_indices = list(set(list(range(len(label_list)))) - set(sub_indices))
            # 打乱操作
            random.shuffle(remain_indices)
            # 全部数据进行训练
            sub_indices.extend(remain_indices)
            assert len(sub_indices) == len(label_list)

            sub_img_list = np.array(img_list)[sub_indices]
            sub_label_list = np.array(label_list)[sub_indices]

            return iter(self.parse_file(sub_img_list, sub_label_list))
        else:
            return iter(self.parse_file(img_list, label_list))

    def __len__(self):
        return self.indices_length if self.num_replicas > 1 else self.length

    def _update_evaluator(self, top_k):
        self.evaluator = VerificationEvaluator()

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler.

        Args:
            epoch (int): Epoch number.
        """
        shuffle_dataset(self.sampler, epoch, self.shuffle)
        self.indices = list(self.sampler)
        self.indices_length = len(self.indices)