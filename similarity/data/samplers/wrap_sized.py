# -*- coding: utf-8 -*-

"""
@date: 2021/12/9 上午11:10
@file: wrap_sized.py
@author: zj
@description: 
"""

from collections.abc import Sized


class WrappedSized(Sized):

    def __len__(self) -> int:
        pass

    def get_length(self) -> int:
        pass
