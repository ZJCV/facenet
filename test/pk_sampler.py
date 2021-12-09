# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 上午10:59
@file: pk_sampler.py
@author: zj
@description: 
"""

import numpy as np
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from similarity.data.samplers.pk_sampler import PKSampler

if __name__ == '__main__':
    groups = list()
    for i in range(10):
        groups.extend([i, ] * 100)
    print(groups)
    p = 8
    k = 8

    model = PKSampler(groups, p, k)
    print(model)

    idxs = list(model)
    batch_idxs = np.array(idxs).reshape(-1, p * k)
    print(idxs)
    print(batch_idxs.shape)

    for tmp_idxs in batch_idxs:
        # print(tmp_idxs)
        tmp_str = ''
        for idx in tmp_idxs:
            # print(groups[idx])
            tmp_str += str(groups[idx]) + ' '
        print(tmp_str)

    # # Ensure p, k constraints on batch
    # dataset = FakeData(size=1000, num_classes=100, image_size=(3, 1, 1), transform=transforms.ToTensor())
    # targets = [target for _, target in dataset]
    # sampler = PKSampler(targets, p, k)
    # loader = DataLoader(dataset, batch_size=p * k, sampler=sampler, num_workers=4)
    #
    # print(len(loader))
