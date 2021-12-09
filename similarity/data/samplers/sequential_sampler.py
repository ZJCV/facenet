# -*- coding: utf-8 -*-

"""
@date: 2021/12/9 上午10:58
@file: sequential_sampler.py
@author: zj
@description: 
"""

from typing import Iterator, TypeVar

T_co = TypeVar('T_co', covariant=True)

from torch.utils.data.sampler import Sampler

from .wrap_sized import WrappedSized


class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: WrappedSized

    def __init__(self, data_source: WrappedSized) -> None:
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.data_source.get_length()))

    def __len__(self) -> int:
        return self.data_source.get_length()
