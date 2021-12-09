# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午2:41
@file: build.py
@author: zj
@description: 
"""

from zcls.model import registry

from .triplet_loss import TripletMarginLoss


def build_criterion(cfg, device):
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg).to(device=device)
