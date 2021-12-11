# -*- coding: utf-8 -*-

"""
@date: 2021/12/9 下午5:22
@file: verification_evaluator.py
@author: zj
@description: 
"""
import torch

from zcls.config.key_word import KEY_OUTPUT
from zcls.data.datasets.evaluator.base_evaluator import BaseEvaluator


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


class VerificationEvaluator(BaseEvaluator):

    def __init__(self, ):
        # super().__init__(classes, top_k)
        self.device = torch.device('cpu')

        self._init()

    def _init(self):
        # super()._init()
        self.total_outputs_list = list()
        self.total_targets_list = list()

    def evaluate_train(self, output_dict: dict, targets: torch.Tensor):
        # return super().evaluate_train(output_dict, targets)
        pass

    def evaluate_test(self, output_dict: dict, targets: torch.Tensor):
        # super().evaluate_test(output_dict, targets)
        assert isinstance(output_dict, dict) and KEY_OUTPUT in output_dict.keys()
        probs = output_dict[KEY_OUTPUT]
        outputs = probs.detach().to(device=self.device)
        targets = targets.detach().to(device=self.device)

        self.total_outputs_list.append(outputs)
        self.total_targets_list.append(targets)

    def get(self):
        # return super().get()
        embeds = torch.cat(self.total_outputs_list, dim=0)
        labels = torch.cat(self.total_targets_list, dim=0)

        dists = torch.cdist(embeds, embeds)

        labels = labels.unsqueeze(0)
        targets = labels == labels.t()

        mask = torch.ones(dists.size()).triu() - torch.eye(dists.size()[0])
        dists = dists[mask == 1]
        targets = targets[mask == 1]

        # threshold, accuracy = find_best_threshold(dists, targets, device)
        threshold, accuracy = find_best_threshold(dists, targets, self.device)

        result_str = f"accuracy: {accuracy:.3f}%, threshold: {threshold:.2f}"
        acc_dict = dict()
        acc_dict["accuracy"] = accuracy
        acc_dict["threshold"] = threshold
        return result_str, acc_dict

    def clean(self):
        # super().clean()
        self._init()
