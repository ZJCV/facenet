# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午2:30
@file: __init__.py.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN
from zcls.config import get_cfg_defaults


def add_custom_config(_C):
    # Add your own customized config.
    _C.SIMILARITY = CN()
    # Number of unique labels/classes per batch
    _C.SIMILARITY.LABELS_PER_BATCH = 8
    # Number of samples per label in a batch
    _C.SIMILARITY.SAMPLES_PER_LABEL = 8
    # Triplet loss margin
    _C.SIMILARITY.MARGIN = 1.0
    # p value for the p-norm distance to calculate between each vector pair
    _C.SIMILARITY.P = 2.0
    # triplet loss mining way, supports
    # batch_all
    # batch_hard
    _C.SIMILARITY.MINING = "batch_all"

    return _C


cfg = add_custom_config(get_cfg_defaults())
