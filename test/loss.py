# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午2:11
@file: loss.py
@author: zj
@description: 
"""

import torch
import numpy as np
from similarity.criterion.loss import _get_triplet_mask

if __name__ == '__main__':
    p = 8
    k = 8

    idxs = np.random.choice(np.arange(10), p, replace=False)
    print(idxs)

    targets = list()
    for i in idxs:
        targets.extend([i, ] * k)

    print(targets)

    mask = _get_triplet_mask(torch.from_numpy(np.array(targets)))
    print(mask)
