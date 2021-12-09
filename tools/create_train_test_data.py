# -*- coding: utf-8 -*-

"""
@date: 2021/12/2 下午3:06
@file: create_train_test_data.py
@author: zj
@description: 
"""

import os

from pathlib import Path
import numpy as np
from tqdm import tqdm

from zcls.config.key_word import KEY_SEP


def get_data(data_root):
    assert os.path.isdir(data_root)

    data_list = list()
    p = Path(data_root)
    for path in tqdm(p.rglob('*.jpg')):
        data_list.append(str(path))
    return data_list


def get_class(data_root):
    assert os.path.isdir(data_root)

    class_list = os.listdir(data_root)
    return class_list


def process(data_list, class_list):
    img_target_list = list()
    for img_path in data_list:
        img_dir = os.path.split(img_path)[0]
        cls_name = os.path.split(img_dir)[1]

        target = class_list.index(cls_name)
        img_target_list.append([img_path, target])

    return img_target_list


def save_to_csv(data_list, dst_path):
    assert not os.path.exists(dst_path)

    length = len(data_list)
    with open(dst_path, 'w') as f:
        for idx, item in tqdm(enumerate(data_list)):
            assert isinstance(item, list), item
            assert len(item) == 2, item
            img_path, target = item

            if idx < (length - 1):
                f.write(f'{img_path}{KEY_SEP}{target}\n')
            else:
                f.write(f'{img_path}{KEY_SEP}{target}')


def save_to_cls(yc_code_list, dst_path):
    assert not os.path.exists(dst_path)

    np.savetxt(dst_path, yc_code_list, fmt='%s', delimiter=' ')


if __name__ == '__main__':
    train_root = 'data/cifar/train'
    train_data_list = get_data(train_root)

    test_root = 'data/cifar/test'
    test_data_list = get_data(test_root)

    class_list = get_class(train_root)
    # 获取训练数据
    train_list = process(train_data_list, class_list)
    # 获取测试数据
    test_list = process(test_data_list, class_list)

    train_path = f'data/cifar/train_{len(train_list)}_1202.csv'
    save_to_csv(train_list, train_path)
    test_path = f'data/cifar/test_{len(test_list)}_1202.csv'
    save_to_csv(test_list, test_path)
    cls_path = f'data/cifar/cls_{len(class_list)}_1202.csv'
    save_to_cls(class_list, cls_path)
