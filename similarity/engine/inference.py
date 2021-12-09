# -*- coding: utf-8 -*-

"""
@date: 2021/12/1 下午5:23
@file: inference.py
@author: zj
@description: 
"""

import os
import torch

import numpy as np
from tqdm import tqdm

from zcls.util.distributed import all_gather, is_master_proc
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


@torch.no_grad()
def compute_on_dataset(images, targets, model, num_gpus):
    output_dict = model(images)

    logger.info('begin eval 2')
    values = list()
    values.append(output_dict[KEY_OUTPUT])
    labels = list()
    labels.append(targets)

    logger.info('begin eval 3')
    # Gather all the predictions across all the devices to perform ensemble.
    if num_gpus > 1:
        values = all_gather(values)
        labels = all_gather(labels)

    logger.info('begin eval 4')
    return values[0].detach().cpu(), labels[0].detach().cpu()
    # return values[0], labels[0]


@torch.inference_mode()
def do_evaluation(cfg, model, loader, device):
    model.eval()
    embeds, labels = [], []
    num_gpus = cfg.NUM_GPUS
    total_num = 0

    logger.info('begin eval ...')
    # 计算所有图片的嵌入和对应标签
    if is_master_proc():
        logger.info('begin eval 1')
        for data in tqdm(loader):
            samples, _labels = data[0].to(device), data[1].to(device)
            print(_labels)
            outs, targets = compute_on_dataset(samples, _labels, model, num_gpus)

            embeds.append(outs)
            labels.append(targets)
            print(targets)
    else:
        for data in loader:
            samples, _labels = data[0].to(device), data[1].to(device)
            outs, targets = compute_on_dataset(samples, _labels, model, num_gpus)

            embeds.append(outs)
            labels.append(targets)

            total_num += len(targets)

    print(labels)
    logger.info('begin eval 5')
    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    print(labels)
    print(len(labels))
    dists = torch.cdist(embeds, embeds)

    labels = labels.unsqueeze(0)
    targets = labels == labels.t()

    mask = torch.ones(dists.size()).triu() - torch.eye(dists.size()[0])
    dists = dists[mask == 1]
    targets = targets[mask == 1]

    cpu_device = torch.device('cpu')
    # threshold, accuracy = find_best_threshold(dists, targets, device)
    threshold, accuracy = find_best_threshold(dists, targets, cpu_device)

    # if is_master_proc():
    logger.info(f"accuracy: {accuracy:.3f}%, threshold: {threshold:.2f}")
