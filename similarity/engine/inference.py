# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午5:23
@file: inference.py
@author: zj
@description: 
"""

import os
import torch

from zcls.config.key_word import KEY_OUTPUT
from zcls.util import logging

logger = logging.get_logger(__name__)


def find_best_threshold(dists, targets, device):
    best_thresh = 0.01
    best_correct = 0
    for thresh in torch.arange(0.0, 1.51, 0.01):
        predictions = dists <= thresh.to(device)
        correct = torch.sum(predictions == targets.to(device)).item()
        if correct > best_correct:
            best_thresh = thresh
            best_correct = correct

    accuracy = 100.0 * best_correct / dists.size(0)

    return best_thresh, accuracy


@torch.inference_mode()
def do_evaluation(model, loader, device):
    model.eval()
    embeds, labels = [], []
    dists, targets = None, None

    # 计算所有图片的嵌入和对应标签
    for data in loader:
        samples, _labels = data[0].to(device), data[1]
        out = model(samples)[KEY_OUTPUT]
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    dists = torch.cdist(embeds, embeds)

    labels = labels.unsqueeze(0)
    targets = labels == labels.t()

    mask = torch.ones(dists.size()).triu() - torch.eye(dists.size(0))
    dists = dists[mask == 1]
    targets = targets[mask == 1]

    threshold, accuracy = find_best_threshold(dists, targets, device)

    logger.info(f"accuracy: {accuracy:.3f}%, threshold: {threshold:.2f}")
