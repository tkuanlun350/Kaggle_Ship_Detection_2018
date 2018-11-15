#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: coco.py

import numpy as np
import os
from termcolor import colored
from tabulate import tabulate

from tensorpack.utils import logger
from tensorpack.utils.rect import FloatBox
from tensorpack.utils.timer import timed_operation
from tensorpack.utils.argtools import log_once

import pickle
import skimage
from skimage import color
import glob
from tqdm import tqdm
import config
import cv2
import pandas as pd

class ResnetDetection(object):
    def __init__(self, basedir, dataset):
        self.basedir = os.path.join(basedir, dataset)
        csv_path = os.path.join(basedir, 'train_ship_segmentations_v2.csv')
        df = pd.read_csv(csv_path, engine="python")
        df = df.set_index('ImageId')
        self.df = df
    def load(self):
        exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
            '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
            'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
            'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg'] 
        if 'test' in self.basedir:
             filenames = glob.glob(self.basedir + '/*.jpg')
        else:
            mode = self.basedir.split("/")[-1]
            datadir = self.basedir.replace(mode, "")
            with open(datadir + "{}_resnet_v2.pkl".format(mode), "rb") as f:
                filenames = pickle.load(f)
                filenames = ["{}/train/{}".format(datadir, fname) for fname in filenames]
        ret = []
        for index, filename in tqdm(enumerate(filenames), total=len(filenames)):
            basename = os.path.basename(filename)
            if basename in exclude_list:
                continue
            _id = basename
            if 'test' in self.basedir:
                data = {}
                data['id'] = _id
                # data['multi_masks'] = self.getAnnotation(_id)
                data['image_data'] = filename
                ret.append(data)
            else:
                data = {}
                data['id'] = _id
                mask = self.df.loc[basename]['EncodedPixels']
                data['with_ship'] = 0 if (type(mask) == float) else 1
                data['image_data'] = filename
                ret.append(data)
        return ret
    @staticmethod
    def load_many(basedir,names):
        """
        Load and merges several instance files together.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            det = ResnetDetection(basedir, n)
            ret.extend(det.load())
        return ret
    
class Detection(object):
    def __init__(self, basedir, dataset):
        self.basedir = os.path.join(basedir, dataset)
        csv_path = os.path.join(basedir, 'train_ship_segmentations_v2.csv')
        if config.PSEUDO:
            pseudo_csv_path = './pseudo_label1.csv'
            df = pd.concat((pd.read_csv(f, engine="python") for f in [csv_path, pseudo_csv_path]))
            df = df.dropna(axis=0)
            df = df.set_index('ImageId')
        else:
            df = pd.read_csv(csv_path, engine="python")
            df = df.dropna(axis=0)
            df = df.set_index('ImageId')
        self.df = df
    
    def load(self, add_gt=True, add_mask=True):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask
        Returns:
            a list of dict, each has keys including:
                height, width, id, file_name,
                and (if add_gt is True) boxes, class, is_crowd
        """
        filenames = glob.glob(self.basedir + '/*.jpg')
        ret = []
        for index, filename in tqdm(enumerate(filenames), total=len(filenames)):
            basename = os.path.basename(filename)
            _id = basename
            if 'val' in self.basedir or 'test' in self.basedir:
                # small val
                data = {}
                data['id'] = _id
                data['image_data'] = filename
                ret.append(data)
            else:
                try:
                    rle_masks = self.df.loc[basename, 'EncodedPixels']
                except:
                    continue
                data = {}
                data['id'] = _id
                data['image_data'] = filename
                ret.append(data)
        return ret

    @staticmethod
    def load_many(basedir,names, add_gt=True, add_mask=False):
        """
        Load and merges several instance files together.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        ret = []
        for n in names:
            det = Detection(basedir, n)
            ret.extend(det.load(add_gt))
        return ret

if __name__ == '__main__':
    c = ResnetDetection(config.BASEDIR, 'train')
    gt = c.load()
    s = 0.0
    index = 0
    for g in gt:
        index += 1
        if g['with_ship'] == 1:
            s += 1
    print(s*1.0/index)
