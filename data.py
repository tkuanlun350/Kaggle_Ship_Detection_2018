#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: data.py

import cv2
import numpy as np
import copy
import random
import math
from utils.np_box_ops import iou as np_iou
from tensorpack.utils.argtools import memoized, log_once
from utils.np_box_ops import area as np_area
from tensorpack.dataflow import (
    MapData, imgaug, AugmentImageComponent, TestDataSpeed, MultiProcessMapData,
    MapDataComponent, DataFromList, PrefetchDataZMQ, BatchData)
import tensorpack.utils.viz as tpviz

#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
#import pycocotools.mask as cocomask

from airbus import Detection, ResnetDetection

from utils.generate_anchors import generate_anchors
from utils.box_ops import get_iou_callable
from common import (
    DataFromListOfDict, CustomResize,
    box_to_point8, point8_to_box, segmentation_to_mask)
import config
from imaug import (do_flip_transpose2, get_resnet_augmentor)
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma
)

class MalformedData(BaseException):
    pass

@memoized
def get_all_anchors_FPN(stride, sizes):
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = generate_anchors(
        stride,
        scales=np.array([sizes], dtype=np.float) / stride,
        ratios=np.array(config.ANCHOR_RATIOS, dtype=np.float))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)
    fpn_max_size = 32 * np.ceil(
        config.MAX_SIZE / 32
    )
    field_size = int(np.ceil(fpn_max_size / float(stride)))

    # field_size = config.MAX_SIZE // stride
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1
    return field_of_anchors

@memoized
def get_all_anchors(
        stride=config.ANCHOR_STRIDE,
        sizes=config.ANCHOR_SIZES,
        ratios=config.ANCHOR_RATIOS):
    """
    Get all anchors in the largest possible image, shifted, floatbox

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == MAX_SIZE//STRIDE, floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SCALE.

    """
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = generate_anchors(
        stride,
        scales=np.array(sizes, dtype=np.float) / stride,
        ratios=np.array(ratios, dtype=np.float))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

    field_size = config.MAX_SIZE // stride
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    #assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1
    return field_of_anchors


def get_anchor_labels(anchors, gt_boxes, crowd_boxes):
    """
    Label each anchor as fg/bg/ignore.
    Args:
        anchors: Ax4 float
        gt_boxes: Bx4 float
        crowd_boxes: Cx4 float

    Returns:
        anchor_labels: (A,) int. Each element is {-1, 0, 1}
        anchor_boxes: Ax4. Contains the target gt_box for each anchor when the anchor is fg.
    """
    # This function will modify labels and return the filtered inds
    def filter_box_label(labels, value, max_num):
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_num:
            disable_inds = np.random.choice(
                curr_inds, size=(len(curr_inds) - max_num),
                replace=False)
            labels[disable_inds] = -1    # ignore them
            curr_inds = np.where(labels == value)[0]
        return curr_inds

    #bbox_iou_float = get_iou_callable()

    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0  # empty images should have been filtered already
    #box_ious = bbox_iou_float(anchors, gt_boxes)  # NA x NB
    box_ious = np_iou(anchors, gt_boxes)  # NA x NB

    ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
    ious_max_per_anchor = box_ious.max(axis=1)
    ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
    # for each gt, find all those anchors (including ties) that has the max ious with it
    anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

    # Setting NA labels: 1--fg 0--bg -1--ignore
    anchor_labels = -np.ones((NA,), dtype='int32')   # NA,

    # the order of setting neg/pos labels matter
    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[ious_max_per_anchor >= config.POSITIVE_ANCHOR_THRES] = 1
    anchor_labels[ious_max_per_anchor < config.NEGATIVE_ANCHOR_THRES] = 0

    # First label all non-ignore candidate boxes which overlap crowd as ignore
    #if crowd_boxes.size > 0:
    #    cand_inds = np.where(anchor_labels >= 0)[0]
    #    cand_anchors = anchors[cand_inds]
    #    ious = bbox_iou_float(cand_anchors, crowd_boxes)
    #    overlap_with_crowd = cand_inds[ious.max(axis=1) > config.CROWD_OVERLAP_THRES]
    #    anchor_labels[overlap_with_crowd] = -1

    # Filter fg labels: ignore some fg if fg is too many
    target_num_fg = int(config.RPN_BATCH_PER_IM * config.RPN_FG_RATIO)
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)
    # Note that fg could be fewer than the target ratio

    # filter bg labels. num_bg is not allowed to be too many
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0 or len(fg_inds) == 0:
        # No valid bg/fg in this image, skip.
        # This can happen if, e.g. the image has large crowd.
        raise MalformedData("No valid foreground/background for RPN!")
    target_num_bg = config.RPN_BATCH_PER_IM - len(fg_inds)
    filter_box_label(anchor_labels, 0, target_num_bg)   # ignore return values

    # Set anchor boxes: the best gt_box for each fg anchor
    anchor_boxes = np.zeros((NA, 4), dtype='float32')
    fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
    anchor_boxes[fg_inds, :] = fg_boxes
    return anchor_labels, anchor_boxes

def get_rpn_anchor_input_FPN(im, boxes, is_crowd):
    def clip_boxes(boxes, shape):
        orig_shape = boxes.shape
        boxes = boxes.reshape([-1, 4])
        h, w = shape
        boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], w)
        boxes[:, 3] = np.minimum(boxes[:, 3], h)
        return boxes.reshape(orig_shape)

    def filter_box_inside(im, boxes):
        h, w = im.shape[:2]
        indices = np.where(
            (boxes[:, 0] >= 0) &
            (boxes[:, 1] >= 0) &
            (boxes[:, 2] <= w) &
            (boxes[:, 3] <= h))[0]
        return indices

    boxes = boxes.copy()
    fpn_boxes = []
    fpn_labels = []
    crowd_boxes = boxes[is_crowd == 1]
    non_crowd_boxes = boxes[is_crowd == 0]
    for stride, size in zip(config.FPN_STRIDES, config.FPN_SIZES):
        ALL_ANCHORS = get_all_anchors_FPN(stride, size)
        H, W = im.shape[:2]
        featureH, featureW = H // stride, W // stride
        # fHxfWxAx4
        featuremap_anchors = ALL_ANCHORS[:featureH, :featureW, :, :]
        featuremap_anchors_flatten = featuremap_anchors.reshape((-1, 4))
        """ no clip """
        featuremap_anchors_flatten = clip_boxes(featuremap_anchors_flatten, im.shape[:2])
        """ no clip """
        fpn_boxes.append(featuremap_anchors_flatten)
        #print("featuremap_{}".format(stride), featuremap_anchors_flatten.shape)
    all_featuremap_anchors_flatten = np.vstack(fpn_boxes)
    #print("all: ", all_featuremap_anchors_flatten.shape)
    # anchor of all featuremaps
    inside_ind = filter_box_inside(im, all_featuremap_anchors_flatten)
    inside_anchors = all_featuremap_anchors_flatten[inside_ind, :]
    anchor_labels, anchor_boxes = get_anchor_labels(inside_anchors, non_crowd_boxes, crowd_boxes)
    """
    featuremap_labels = -np.ones((featureH * featureW * config.NUM_ANCHOR, ), dtype='int32')
    featuremap_labels[inside_ind] = anchor_labels
    # featuremap_labels = featuremap_labels.reshape((featureH, featureW, config.NUM_ANCHOR))
    featuremap_boxes = np.zeros((featureH * featureW * config.NUM_ANCHOR, 4), dtype='float32')
    featuremap_boxes[inside_ind, :] = anchor_boxes
    # featuremap_boxes = featuremap_boxes.reshape((featureH, featureW, config.NUM_ANCHOR, 4))
    return featuremap_labels, featuremap_boxes
    """
    return anchor_labels, anchor_boxes

def get_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        The anchor labels and target boxes for each pixel in the featuremap.
        fm_labels: fHxfWxNA
        fm_boxes: fHxfWxNAx4
    """
    boxes = boxes.copy()

    ALL_ANCHORS = get_all_anchors()
    H, W = im.shape[:2]
    featureH, featureW = H // config.ANCHOR_STRIDE, W // config.ANCHOR_STRIDE

    def clip_boxes(boxes, shape):
        """
        Args:
            boxes: (...)x4, float
            shape: h, w
        """
        orig_shape = boxes.shape
        boxes = boxes.reshape([-1, 4])
        h, w = shape
        boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], w)
        boxes[:, 3] = np.minimum(boxes[:, 3], h)
        return boxes.reshape(orig_shape)

    def filter_box_inside(im, boxes):
        h, w = im.shape[:2]
        indices = np.where(
            (boxes[:, 0] >= 0) &
            (boxes[:, 1] >= 0) &
            (boxes[:, 2] <= w) &
            (boxes[:, 3] <= h))[0]
        return indices

    crowd_boxes = boxes[is_crowd == 1]
    non_crowd_boxes = boxes[is_crowd == 0]

    # fHxfWxAx4
    featuremap_anchors = ALL_ANCHORS[:featureH, :featureW, :, :]
    featuremap_anchors_flatten = featuremap_anchors.reshape((-1, 4))
    # only use anchors inside the image
    ### clip outside box and use them
    featuremap_anchors_flatten = clip_boxes(featuremap_anchors_flatten, im.shape[:2])
    ###
    inside_ind = filter_box_inside(im, featuremap_anchors_flatten)
    inside_anchors = featuremap_anchors_flatten[inside_ind, :]

    anchor_labels, anchor_boxes = get_anchor_labels(inside_anchors, non_crowd_boxes, crowd_boxes)

    # Fill them back to original size: fHxfWx1, fHxfWx4
    featuremap_labels = -np.ones((featureH * featureW * config.NUM_ANCHOR, ), dtype='int32')
    featuremap_labels[inside_ind] = anchor_labels
    featuremap_labels = featuremap_labels.reshape((featureH, featureW, config.NUM_ANCHOR))
    featuremap_boxes = np.zeros((featureH * featureW * config.NUM_ANCHOR, 4), dtype='float32')
    featuremap_boxes[inside_ind, :] = anchor_boxes
    featuremap_boxes = featuremap_boxes.reshape((featureH, featureW, config.NUM_ANCHOR, 4))
    return featuremap_labels, featuremap_boxes

def multi_mask_to_annotation(multi_mask):
    H,W      = multi_mask.shape[:2]
    box      = []
    label    = []
    instance = []
    is_crowd = []
    raw_label = np.unique(multi_mask)
    for i in raw_label:
        _la = i // 1000
        _ins_a = i % 1000
        if i == 0:
            # ignore background
            continue
        mask = (multi_mask==i)
        if mask.sum()>1:
            y,x = np.where(mask)
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            w = (x1-x0)+1
            h = (y1-y0)+1

            border = max(1, round(0.1*min(w,h)))
            x0 = x0-border
            x1 = x1+border
            y0 = y0-border
            y1 = y1+border

            #clip
            x0 = max(1,x0)
            y0 = max(1,y0)
            x1 = min(W-1,x1)
            y1 = min(H-1,y1)
            
            box.append([x0,y0,x1,y1])
            label.append(_la)
            instance.append(mask)
            is_crowd.append(0) # ignored cases

    box      = np.array(box, np.float32)
    label    = np.array(label, np.int32)
    instance = np.array(instance, np.float32)
    is_crowd = np.array(is_crowd, np.int32)


    return box, label, instance, is_crowd

def better_resize(img, size, max_size=config.MAX_SIZE):
    h, w = img.shape[:2]
    scale = size * 1.0 / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return newh, neww

def fix_resize_transform_range(image, semantic_label, sizes, u=0.5):
    H,W = image.shape[:2]
    #s = np.random.choice(sizes)
    s = random.randint(sizes[0], sizes[1])
    #s = sizes
    h, w = better_resize(image, s)
    if (H,W) != (h,w):
        image = cv2.resize(image, (w,h))

        semantic_label = cv2.resize(semantic_label, (w, h), interpolation=cv2.INTER_NEAREST)
        semantic_label = semantic_label.astype(np.int32)
    return image, semantic_label

def fix_resize_transform_scale(image, semantic_label, sizes, u=0.5):
    H,W = image.shape[:2]
    s = np.random.choice(sizes)
    #s = sizes
    h, w = better_resize(image, s)
    if (H,W) != (h,w):
        image = cv2.resize(image, (w,h))

        semantic_label = cv2.resize(semantic_label, (w, h), interpolation=cv2.INTER_NEAREST)
        semantic_label = semantic_label.astype(np.int32)
    return image, semantic_label

def random_crop_box_center(im, label, crop_height=800, crop_width=1024):
    crop_height = min(crop_height, im.shape[0])
    crop_width = min(crop_width, im.shape[1])
    
    instanceId = np.unique(label[label!=0])
    if len(instanceId) == 0:
        return im, label
    choiced_id = np.random.choice(instanceId)
    mask_true = np.where(label==choiced_id)
    seed = np.random.choice(range(len(mask_true[0])))
    coordinate = (mask_true[0][seed], mask_true[1][seed])
    ### get crop window ###
    start_y = max(0, coordinate[0] - crop_height//2)
    end_y = min(im.shape[0], coordinate[0] + crop_height//2)
    start_x = max(0, coordinate[1] - crop_width//2)
    end_x = min(im.shape[1], coordinate[1] + crop_width//2)
    ### random shift ###
    shift_limit = (-0.2, 0.2)
    dx = round(random.uniform(shift_limit[0],shift_limit[1])*crop_width )
    dy = round(random.uniform(shift_limit[0],shift_limit[1])*crop_height)
    start_x = max(start_x + dx, 0)
    end_x = min(end_x + dx, im.shape[1])
    start_y = max(start_y + dy, 0)
    end_y = min(end_y + dy, im.shape[0])
    ###
    #cropped_im = im[start_y:end_y, start_x:end_x, :]
    return im[start_y:end_y, start_x:end_x, :],  label[start_y:end_y, start_x:end_x]

def getAnnotation(df, imageId):
    def rle_decode(mask_rle, shape=(768, 768)):
        s = mask_rle.split()
        starts =  np.asarray(s[0::2], dtype=int)
        lengths = np.asarray(s[1::2], dtype=int)

        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T  # Needed to align to RLE direction

    def instances_to_multimask(instance):
        H,W = instance.shape[1:3]
        multi_mask = np.zeros((H,W),np.int32)
        num_masks = len(instance)
        for i in range(num_masks):
            multi_mask[instance[i]>0] = 1000 + i

        return multi_mask
    try:
        rle_masks = df.loc[imageId, 'EncodedPixels']
    except:
        return None
    if isinstance(rle_masks, str):
        rle_masks = [rle_masks]
    else:
        rle_masks = rle_masks.tolist()
    rle_masks = [rle_decode(m) for m in rle_masks]
    multimask = instances_to_multimask(np.array(rle_masks))
    return multimask

def strong_aug(p=.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        RandomBrightness(p=0.2,limit=0.2),
        RandomContrast(p=0.2,limit=0.2),
        RandomGamma(p=0.2)
    ], p=p)

def pad_to_factor(image, factor=32):
    height,width = image.shape[:2]
    h = math.ceil(height/factor)*factor
    w = math.ceil(width/factor)*factor

    image = cv2.copyMakeBorder(image, top=0, bottom=h-height, left=0, right=w-width,
                               borderType= cv2.BORDER_REFLECT101, value=[0,0,0] )

    return image

def get_train_dataflow(add_mask=True):
    """
    Return a training dataflow. Each datapoint is:
    image, fm_labels, fm_boxes, gt_boxes, gt_class [, masks]
    """
    imgs = Detection.load_many(
        config.BASEDIR, config.TRAIN_DATASET, add_gt=True, add_mask=add_mask)
    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    imgs = list(imgs)
    import os
    import pandas as pd
    csv_path = os.path.join(config.BASEDIR, 'train_ship_segmentations_v2.csv')
    if config.PSEUDO:
        pseudo_csv_path = './pseudo_label1.csv'
        df = pd.concat((pd.read_csv(f, engine="python") for f in [csv_path, pseudo_csv_path]))
        df = df.dropna(axis=0)
        df = df.set_index('ImageId')
    else:
        df = pd.read_csv(csv_path, engine="python")
        df = df.dropna(axis=0)
        df = df.set_index('ImageId')
    ds = DataFromList(imgs, shuffle=True)
    def preprocess(img):
        im, fname = img['image_data'], img['id']
        multi_mask = getAnnotation(df, fname)
        if multi_mask is None:
            return None
        im = cv2.imread(im)
        #============================
        #if random.random() > 0.5:
        #    im = np.fliplr(im) # h, w, 3
        #    multi_mask = np.fliplr(multi_mask)
        #im, multi_mask = do_flip_transpose2(im, multi_mask, type=random.randint(0,7))
        augmented = strong_aug()(image=im, mask=multi_mask)
        im, multi_mask = augmented['image'], augmented['mask']
        #============================
        # Resize
        im, multi_mask = fix_resize_transform_range(im, multi_mask, [768, 2000], 1.0)
        im = pad_to_factor(im)
        multi_mask = pad_to_factor(multi_mask)

        boxes, klass, masks, is_crowd = multi_mask_to_annotation(multi_mask)
        if len(boxes) == 0 or np.min(np_area(boxes)) <= 0:
            log_once("Input have zero area box: {}".format(fname), 'warn')
            return None
        # rpn anchor:
        try:
            if config.FPN:
                fm_labels, fm_boxes = get_rpn_anchor_input_FPN(im, boxes, is_crowd)
            else:
                fm_labels, fm_boxes = get_rpn_anchor_input(im, boxes, is_crowd)
                
            if not len(boxes):
                raise MalformedData("No valid gt_boxes!")
        except MalformedData as e:
            log_once("Input {} is filtered for training: {}".format(fname, str(e)), 'warn')
            return None

        ret = [im, fm_labels, fm_boxes, boxes, klass, masks]
        """
        from viz import draw_annotation, draw_mask
        viz = draw_annotation(im, boxes, klass)
        for ind, mask in enumerate(masks):
            viz = draw_mask(viz, mask)
            cv2.imwrite("./test_{}.jpg".format(np.random.rand()), viz)
        if (len(boxes) > 3):
            exit()
        """
        return ret

    ds = MapData(ds, preprocess)
    ds = PrefetchDataZMQ(ds, 6)
    return ds


def get_test_dataflow(add_mask=True):
    """
    Return a training dataflow. Each datapoint is:
    image, fm_labels, fm_boxes, gt_boxes, gt_class [, masks]
    """
    imgs = Detection.load_many(
        config.BASEDIR, config.VAL_DATASET, add_gt=False, add_mask=add_mask)
    # no filter for training
    ds = DataFromListOfDict(imgs, ['image_data', 'id'])

    def f(image):
        im = cv2.imread(image)
        return im

    ds = MapDataComponent(ds, f, 0)
    ds = PrefetchDataZMQ(ds, 1)
    return ds

def get_resnet_train_dataflow():
    imgs = ResnetDetection.load_many(
        config.BASEDIR, config.TRAIN_DATASET)
    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    imgs = list(imgs)

    ds = DataFromList(imgs, shuffle=True)
    augmentors = get_resnet_augmentor()
    def preprocess(img):
        im, fname, label = img['image_data'], img['id'], img['with_ship']
        im = cv2.imread(im)
        #============Aug================
        im = cv2.resize(im, (config.RESNET_SIZE, config.RESNET_SIZE))
        augmented = strong_aug()(image=im)
        im = augmented['image']
        # im, multi_mask = do_flip_transpose2(im, multi_mask, type=random.randint(0,7))
        #============================
        ret = [im, label]
        return ret
    ds = MapData(ds, preprocess)
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    ds = BatchData(ds, config.RESNET_BATCH)
    ds = PrefetchDataZMQ(ds, 6)
    return ds

def get_resnet_val_dataflow():
    imgs = ResnetDetection.load_many(
        config.BASEDIR, config.VAL_DATASET)
    imgs = list(imgs)
    # ds = DataFromListOfDict(imgs, ['image_data', 'with_ship', 'id'])
    ds = DataFromList(imgs, shuffle=False)
    def f(img):
        image, label = img['image_data'], img['with_ship']
        im = cv2.imread(image)
        im = cv2.resize(im, (config.RESNET_SIZE, config.RESNET_SIZE))
        return [im, label]

    ds = MapData(ds, f)
    ds = BatchData(ds, config.RESNET_BATCH)
    ds = PrefetchDataZMQ(ds, 1)
    return ds

#=====================================#

def get_debug_dataflow(add_mask=True, imageHW=768):
    """
    Return a training dataflow. Each datapoint is:
    image, fm_labels, fm_boxes, gt_boxes, gt_class [, masks]
    """
    imgs = Detection.load_many(
        config.BASEDIR, config.TRAIN_DATASET, add_gt=True, add_mask=add_mask)
    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    imgs = list(imgs)
    import os
    import pandas as pd
    csv_path = os.path.join(config.BASEDIR, 'train_ship_segmentations_v2.csv')
    df = pd.read_csv(csv_path, engine="python")
    df = df.dropna(axis=0)
    df = df.set_index('ImageId')

    ds = DataFromList(imgs, shuffle=True)
    def preprocess(img):
        im, fname = img['image_data'], img['id']
        multi_mask = getAnnotation(df, fname)
        im = cv2.imread(im)
        im, multi_mask = fix_resize_transform_range(im, multi_mask, [imageHW, imageHW], 1.0)
        boxes, klass, masks, is_crowd = multi_mask_to_annotation(multi_mask)
        return boxes
    ds = MapData(ds, preprocess)
    ds = PrefetchDataZMQ(ds, 6)
    return ds

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

def cal_avg_iou_per_gt(gt_boxes, sizes, ratios, imageHW=768):
    def filter_box_inside(im, boxes):
        h, w = im.shape[:2]
        indices = np.where(
            (boxes[:, 0] >= 0) &
            (boxes[:, 1] >= 0) &
            (boxes[:, 2] <= w) &
            (boxes[:, 3] <= h))[0]
        return indices
    im = np.zeros((imageHW, imageHW))
    ALL_ANCHORS = get_all_anchors(sizes=sizes, ratios=ratios)
    H, W = im.shape[:2]
    featureH, featureW = H // 16, W // 16
    featuremap_anchors = ALL_ANCHORS[:featureH, :featureW, :, :]
    featuremap_anchors_flatten = featuremap_anchors.reshape((-1, 4))
    inside_ind = filter_box_inside(im, featuremap_anchors_flatten)
    inside_anchors = featuremap_anchors_flatten[inside_ind, :]

    anchors = inside_anchors
    box_ious = np_iou(anchors, gt_boxes)  # NA x NB
    ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
    ious_max_per_anchor = box_ious.max(axis=1)
    ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
    return ious_max_per_gt.mean()

if __name__ == '__main__':
    
    imgs = Detection.load_many(
        config.BASEDIR, config.TRAIN_DATASET, add_gt=True, add_mask=True)
    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    imgs = list(imgs)
    import os
    import pandas as pd
    csv_path = os.path.join(config.BASEDIR, 'train_ship_segmentations_v2.csv')
    df = pd.read_csv(csv_path, engine="python")
    df = df.dropna(axis=0)
    df = df.set_index('ImageId')
    from tqdm import tqdm
    for img in tqdm(imgs, total=len(imgs)):
        im, fname = img['image_data'], img['id']
        multi_mask = getAnnotation(df, fname)
        
        im = cv2.imread(im)
        #============================
        # Resize
        augmented = strong_aug()(image=im, mask=multi_mask)
        im, multi_mask = augmented['image'], augmented['mask']
        boxes, klass, masks, is_crowd = multi_mask_to_annotation(multi_mask)
        if len(boxes) == 0 or np.min(np_area(boxes)) <= 0:
            log_once("Input have zero area box: {}".format(fname), 'warn')
            print(boxes)
            exit()
        """
        from viz import draw_annotation, draw_mask
        viz = draw_annotation(im, boxes, klass)
        for ind, mask in enumerate(masks):
            viz = draw_mask(viz, mask)
            cv2.imwrite("./eval_gt/{}.jpg".format(fname), viz)
        """
    """    
    # for each gt, find all those anchors (including ties) that has the max ious with it
    ANCHOR_SIZES = (32,64,128,256,512)
    RAIOS = (0.5,1,2)
    #ANCHOR_SIZES = (16, 32, 64, 128, 256)
    from tensorpack.dataflow import PrintData
    from tqdm import tqdm
    imageHW = 2000
    ds = get_debug_dataflow(imageHW=imageHW)
    ds.reset_state()
    all_boxes = []
    all_boxes_hw = []
    all_boxes_ratio = []
    for idx, boxes in tqdm(enumerate(ds.get_data()), total=ds.size()):
        for i in boxes:
            all_boxes.append(i)
            all_boxes_hw.append([i[2]-i[0], i[3]-i[1]])
            all_boxes_ratio.append( (i[2]-i[0])/float(i[3]-i[1]) )
    print(np.array(all_boxes_ratio).mean())
    all_boxes = np.array(all_boxes)
    all_boxes_hw = np.array(all_boxes_hw)
    print(cal_avg_iou_per_gt(all_boxes, sizes=ANCHOR_SIZES, ratios=RAIOS, imageHW=imageHW))
    out = kmeans(all_boxes_hw, k=5)
    print("Accuracy: {:.2f}%".format(avg_iou(all_boxes_hw, out) * 100))
    print("Boxes:\n {}".format(out))
    """
    
        
