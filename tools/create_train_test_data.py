# -*- coding: utf-8 -*-

"""
@date: 2021/12/2 下午3:06
@file: create_train_test_data.py
@author: zj
@description:

$ python tools/create_train_test_data.py -h
usage: create_train_test_data.py [-h] [--test] data_root save_root

positional arguments:
  data_root   Data ROOT. Default: None
  save_root   Save ROOT. Default: None

optional arguments:
  -h, --help  show this help message and exit
  --test      Separate training and test set

Enter the data root path and the data file saving path.
for data-root, assume that the data storage format is as follows :
data_root/
    cls_1/
        img_1.jpg
        img_2.jpg
        ...
    cls_2/
        ...
for save-root, used to save data profiles.
    if set --test, then the data will be saved in two parts and in two folders.
        save_root/
            train/
                data.csv
                cls.csv
            test/
                data.csv
                cls.csv
    if not set --test, then the data will be saved in two parts, like this
        save_root/
            data.csv
            cls.csv
note: if --test is used, then save_root can exist; Otherwise, it cannot exist.
"""

import os
import argparse

import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from zcls.config.key_word import KEY_SEP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', type=str, default=None, help='Data ROOT. Default: None')
    parser.add_argument('save_root', type=str, default=None, help='Save ROOT. Default: None')
    parser.add_argument('ratio', type=float, default=0.2, help='Training/Test set split ratio. Default: 0.2')
    parser.add_argument('--test', default=False, action='store_true', help='Separate training and test set')

    return parser.parse_args()


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


def save(save_root, data_list, cls_list):
    data_path = f'{save_root}/data.csv'
    save_to_csv(data_list, data_path)
    cls_path = f'{save_root}/cls.csv'
    save_to_cls(cls_list, cls_path)


def main(data_root, save_root, is_split_test, ratio):
    print('process ...')
    total_data_list = get_data(data_root)
    # get cls
    class_list = get_class(data_root)
    # get data
    train_list = process(total_data_list, class_list)

    print('save ...')
    if is_split_test:
        X_train, X_test, _, _ = train_test_split(train_list, list(range(len(train_list))),
                                                 test_size=ratio, random_state=0)
        train_root = os.path.join(save_root, 'train')
        os.makedirs(train_root)
        save(train_root, X_train, class_list)

        test_root = os.path.join(save_root, 'test')
        os.makedirs(test_root)
        save(test_root, X_test, class_list)
    else:
        save(save_root, train_list, class_list)


if __name__ == '__main__':
    args = parse_args()

    data_root = os.path.abspath(args.data_root)
    save_root = os.path.abspath(args.save_root)
    ratio = args.ratio
    is_split_test = args.test

    if not os.path.isdir(data_root):
        raise ValueError(f'{args.data_root} is not a dir')
    if not is_split_test and os.path.exists(save_root):
        raise ValueError(f'{args.save_root} has existed')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    main(data_root, save_root, is_split_test, ratio)
