#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
import matplotlib
matplotlib.use('Agg')
import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import math
import numpy as np
import json
import tensorflow as tf
import zipfile
import pickle

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu

from basemodel import (
    image_preprocess, pretrained_resnet_FPN, pretrained_resnet_conv4, resnet_conv5)
from model import *
from data import (
    get_train_dataflow, get_test_dataflow,
    get_all_anchors, get_all_anchors_FPN, get_resnet_train_dataflow, get_resnet_val_dataflow)
from viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs, shrink_poly, draw_refined_outputs)
from common import print_config
from eval import (
    eval_on_dataflow, detect_one_image_TTA2, detect_one_image_TTA, detect_one_image, DetectionResult, print_evaluation_scores)
import config
from airbus import Detection, ResnetDetection
#import matplotlib.pyplot as plt
import collections
import nibabel as nib
#import SimpleITK as sitk
from basemodel_tp import (
    resnet_fpn_backbone
)
def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu

def get_resnet_model_output_names():
    return ['final_probs', 'final_labels']

def get_model_output_names():
    ret = ['final_boxes', 'final_probs', 'final_labels']
    if config.MODE_MASK:
        ret.append('final_masks')
    return ret

class ResnetModel(ModelDesc):
    def _get_inputs(self):
        ret = [
            InputDesc(tf.float32, (None, None, None, 3), 'image'),
            InputDesc(tf.int32, (None,), 'labels'),
        ]
        return ret

    def _build_graph(self, inputs):
        is_training = get_current_tower_context().is_training
        image, label = inputs
        tf.summary.image('viz', image, max_outputs=10)
        image = image_preprocess(image, bgr=True)
        image = tf.transpose(image, [0, 3, 1, 2])
        conv4 = pretrained_resnet_conv4(image, config.RESNET_NUM_BLOCK[:3])
        # B*c*h*w
        conv5 = resnet_conv5(conv4, config.RESNET_NUM_BLOCK[-1])
        # train head only
        logits = cls_head('cls', conv5)
        if is_training:
            loss = cls_loss(logits, label)
            wd_cost = regularize_cost(
                '(?:group1|group2|group3|cls)/.*W',
                l2_regularizer(1e-4), name='wd_cost')

            self.cost = tf.add_n([
                loss, wd_cost], 'total_cost')                

            add_moving_summary(self.cost, wd_cost)
        else:
            final_prob = tf.nn.sigmoid(logits)
            tf.identity(final_prob, name='final_probs')
            #final_label = tf.where(final_prob > 0.5, tf.ones_like(final_prob), tf.zeros_like(final_prob))
            final_label = tf.round(final_prob)
            tf.identity(final_label, name='final_labels')
    
    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        tf.summary.scalar('learning_rate', lr)

        factor = get_batch_factor()
        if factor != 1:
            lr = lr / float(factor)
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            opt = optimizer.AccumGradOptimizer(opt, factor)
        else:
            opt = tf.train.MomentumOptimizer(lr, 0.9)
        #opt = tf.train.AdamOptimizer(lr)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

class Model(ModelDesc):
    def _get_inputs(self):
        if config.FPN:
            ret = [
                InputDesc(tf.float32, (None, None, 3), 'image'),
                InputDesc(tf.int32, (None,), 'anchor_labels'), # stride*H*W*num_size
                InputDesc(tf.float32, (None, 4), 'anchor_boxes'),
                InputDesc(tf.float32, (None, 4), 'gt_boxes'),
                InputDesc(tf.int64, (None,), 'gt_labels')]  # all > 0
        else:
            ret = [
                InputDesc(tf.float32, (None, None, 3), 'image'),
                InputDesc(tf.int32, (None, None, config.NUM_ANCHOR), 'anchor_labels'),
                InputDesc(tf.float32, (None, None, config.NUM_ANCHOR, 4), 'anchor_boxes'),
                InputDesc(tf.float32, (None, 4), 'gt_boxes'),
                InputDesc(tf.int64, (None,), 'gt_labels')]  # all > 0
        if config.MODE_MASK:
            ret.append(
                InputDesc(tf.uint8, (None, None, None), 'gt_masks')
            )   # NR_GT x height x width
        return ret

    def _preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def _get_anchors_FPN(self, image):
        all_fm_anchors = []
        # [P6 ~ P2]
        for stride, size in zip(config.FPN_STRIDES, config.FPN_SIZES):
            with tf.name_scope('anchors_stride_{}'.format(stride)):
                all_anchors = tf.constant(get_all_anchors_FPN(stride, size), name='all_anchors_stride_{}'.format(stride), dtype=tf.float32)
                fm_anchors = tf.slice(
                    all_anchors, [0, 0, 0, 0], tf.stack([
                        tf.shape(image)[0] // stride,
                        tf.shape(image)[1] // stride,
                        -1, -1]), name='fm_anchors')
                #fm_anchors = tf.Print(fm_anchors, [tf.shape(fm_anchors)], message="layer fm_anchors: ")
                all_fm_anchors.append(tf.reshape(fm_anchors, [-1, 4]))
        #all_fm_anchors = tf.concat(all_fm_anchors, 0)
        #all_fm_anchors = tf.Print(all_fm_anchors, [tf.shape(all_fm_anchors)], message="layer all: ")
        return all_fm_anchors

    def _get_anchors(self, image):
        """
        Returns:
            FSxFSxNAx4 anchors,
        """
        # FSxFSxNAx4 (FS=MAX_SIZE//ANCHOR_STRIDE)
        with tf.name_scope('anchors'):
            all_anchors = tf.constant(get_all_anchors(), name='all_anchors', dtype=tf.float32)
            fm_anchors = tf.slice(
                all_anchors, [0, 0, 0, 0], tf.stack([
                    tf.shape(image)[0] // config.ANCHOR_STRIDE,
                    tf.shape(image)[1] // config.ANCHOR_STRIDE,
                    -1, -1]), name='fm_anchors')

            return fm_anchors

    def _build_graph(self, inputs):
        is_training = get_current_tower_context().is_training
        if config.MODE_MASK:
            image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, gt_masks = inputs
        else:
            image, anchor_labels, anchor_boxes, gt_boxes, gt_labels = inputs
        
        
        if config.FPN:
            fm_anchors = self._get_anchors_FPN(image)
            image = self._preprocess(image)     # 1CHW
            image_shape2d = tf.shape(image)[2:]
            #featuremap = pretrained_resnet_FPN(image, config.RESNET_NUM_BLOCK)
            featuremap = resnet_fpn_backbone(image, config.RESNET_NUM_BLOCK)
            # [P6, P5, P4, P3, P2]
        else:
            fm_anchors = self._get_anchors(image)
            image = self._preprocess(image)     # 1CHW
            image_shape2d = tf.shape(image)[2:]
            featuremap = pretrained_resnet_conv4(image, config.RESNET_NUM_BLOCK[:3])
            #cls_feat = cls_head('mtl', featuremap)
             
        if config.FPN:
            rpn_label_logits_FPN, rpn_box_logits_FPN = rpn_head_FPN('rpn', featuremap, 256, len(config.ANCHOR_RATIOS))
            decoded_boxes = decode_bbox_target_FPN(rpn_box_logits_FPN, fm_anchors)  # fHxfWxNAx4, floatbox
            proposal_boxes, proposal_scores = generate_rpn_proposals_FPN(
                decoded_boxes, # layer-wise
                rpn_label_logits_FPN, # layer-wise
                image_shape2d)
            anchor_boxes_encoded = encode_bbox_target(anchor_boxes, tf.concat(fm_anchors, 0))
        else:
            rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, 1024, config.NUM_ANCHOR)
            decoded_boxes = decode_bbox_target(rpn_box_logits, fm_anchors)  # fHxfWxNAx4, floatbox
            proposal_boxes, proposal_scores = generate_rpn_proposals(
                tf.reshape(decoded_boxes, [-1, 4]),
                tf.reshape(rpn_label_logits, [-1]),
                image_shape2d)
            anchor_boxes_encoded = encode_bbox_target(anchor_boxes, fm_anchors)     
        if config.FPN:
            # RCNN FPN 
            if is_training:
                # sample proposal boxes in training
                if config.OHEM:
                    rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt, all_inds_wrt_gt = sample_fast_rcnn_targets_RELATION(
                        proposal_boxes, gt_boxes, gt_labels)
                else:
                    rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt, all_inds_wrt_gt = sample_fast_rcnn_targets_RELATION(
                        proposal_boxes, gt_boxes, gt_labels)

                boxes_on_featuremap = rcnn_sampled_boxes
            else:
                # use all proposal boxes in inference
                boxes_on_featuremap = proposal_boxes

            roi_resized = roi_align_FPN(featuremap, boxes_on_featuremap, 7)
        else:
            if is_training:
                # sample proposal boxes in training
                #rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
                #    proposal_boxes, gt_boxes, gt_labels)
                rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt, all_inds_wrt_gt = sample_fast_rcnn_targets_RELATION(
                        proposal_boxes, gt_boxes, gt_labels)
                boxes_on_featuremap = rcnn_sampled_boxes * (1.0 / config.ANCHOR_STRIDE)
            else:
                # use all proposal boxes in inference
                boxes_on_featuremap = proposal_boxes * (1.0 / config.ANCHOR_STRIDE)

            roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

        def ff_true():
            if config.FPN:
                feature_fastrcnn = roi_resized
                head_feature = fastrcnn_2fc_head('fastrcnn', roi_resized)
                fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(
                    'fastrcnn/outputs', head_feature, config.NUM_CLASS, class_agnostic_regression=False)
                #fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_head_FPN('fastrcnn', feature_fastrcnn, config.NUM_CLASS)
            else:
                feature_fastrcnn = resnet_conv5(roi_resized, config.RESNET_NUM_BLOCK[-1])    # nxcx7x7
                fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_head('fastrcnn', feature_fastrcnn, config.NUM_CLASS)

            return feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits

        def ff_false():
            if config.FPN:
                ncls = config.NUM_CLASS
                return tf.zeros([0, 256, 7, 7]), tf.zeros([0, ncls]), tf.zeros([0, ncls, 4])
            else:
                ncls = config.NUM_CLASS
                return tf.zeros([0, 2048, 7, 7]), tf.zeros([0, ncls]), tf.zeros([0, ncls - 1, 4])

        feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits = tf.cond(
            tf.size(boxes_on_featuremap) > 0, ff_true, ff_false)

        if is_training:
            if config.FPN:
                # rpn loss
                rpn_label_loss, rpn_box_loss = rpn_losses(
                    anchor_labels, anchor_boxes_encoded, tf.concat(rpn_label_logits_FPN, 0), tf.concat(rpn_box_logits_FPN, 0))
            else:
                # rpn loss
                rpn_label_loss, rpn_box_loss = rpn_losses(
                    anchor_labels, anchor_boxes_encoded, rpn_label_logits, rpn_box_logits)
           
            # fastrcnn loss
            fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])   # fg inds w.r.t all samples
            fg_sampled_boxes = tf.gather(rcnn_sampled_boxes, fg_inds_wrt_sample)

            with tf.name_scope('fg_sample_patch_viz'):
                fg_sampled_patches = crop_and_resize(
                    image, fg_sampled_boxes,
                    tf.zeros_like(fg_inds_wrt_sample, dtype=tf.int32), 300)
                fg_sampled_patches = tf.transpose(fg_sampled_patches, [0, 2, 3, 1])
                fg_sampled_patches = tf.reverse(fg_sampled_patches, axis=[-1])  # BGR->RGB
                tf.summary.image('viz', fg_sampled_patches, max_outputs=30)

            if config.OHEM:
                matched_gt_boxes = tf.gather(gt_boxes, all_inds_wrt_gt)

                encoded_boxes = encode_bbox_target(
                    matched_gt_boxes,
                    rcnn_sampled_boxes) * tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS)
                
                fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses_OHEM(
                    rcnn_labels, fastrcnn_label_logits,
                    encoded_boxes,
                    fastrcnn_box_logits)
            else:
                matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

                encoded_boxes = encode_bbox_target(
                    matched_gt_boxes,
                    fg_sampled_boxes) * tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS)
                
                fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(
                    rcnn_labels, fastrcnn_label_logits,
                    encoded_boxes,
                    tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample))

            if config.MODE_MASK:
                if config.FPN:
                    fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)
                    roi_feature_maskrcnn = roi_align_FPN(featuremap, rcnn_sampled_boxes, 14)
                    fg_feature = tf.gather(roi_feature_maskrcnn, fg_inds_wrt_sample)
                    #mask_logits = maskrcnn_head_FPN('maskrcnn', fg_feature, config.NUM_CLASS)   # #fg x #cat x 14x14
                    mask_logits = maskrcnn_upXconv_head('maskrcnn', fg_feature, config.NUM_CLASS-1, 4)

                    gt_masks_for_fg = tf.gather(gt_masks, fg_inds_wrt_gt)  # nfg x H x W
                    target_masks_for_fg = crop_and_resize(
                        tf.expand_dims(gt_masks_for_fg, 1),
                        fg_sampled_boxes,
                        tf.range(tf.size(fg_inds_wrt_gt)), 14*2*2)  # nfg x 1x14x14 28*28 for FPN
                    target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                    mrcnn_loss = maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg)
                else:
                    # maskrcnn loss
                    fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)
                    fg_feature = tf.gather(feature_fastrcnn, fg_inds_wrt_sample)
                    mask_logits = maskrcnn_head('maskrcnn', fg_feature, config.NUM_CLASS)   # #fg x #cat x 14x14

                    gt_masks_for_fg = tf.gather(gt_masks, fg_inds_wrt_gt)  # nfg x H x W
                    target_masks_for_fg = crop_and_resize(
                        tf.expand_dims(gt_masks_for_fg, 1),
                        fg_sampled_boxes,
                        tf.range(tf.size(fg_inds_wrt_gt)), 14)  # nfg x 1x14x14
                    target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                    mrcnn_loss = maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg)
            else:
                mrcnn_loss = 0.0

            
            wd_cost = regularize_cost(
                '(?:group1|group2|group3|rpn|fastrcnn|maskrcnn)/.*W',
                l2_regularizer(1e-4), name='wd_cost')

            self.cost = tf.add_n([
                rpn_label_loss, rpn_box_loss,
                fastrcnn_label_loss, fastrcnn_box_loss,
                mrcnn_loss,
                wd_cost], 'total_cost')                

            add_moving_summary(self.cost, wd_cost)
        else:
            label_probs = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
            anchors = tf.tile(tf.expand_dims(proposal_boxes, 1), [1, config.NUM_CLASS, 1])   # #proposal x #Cat x 4
            decoded_boxes = decode_bbox_target(
                fastrcnn_box_logits /
                tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS), anchors)
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

            # indices: Nx2. Each index into (#proposal, #category)
            pred_indices, final_probs = fastrcnn_predictions_cascade(decoded_boxes, label_probs)
            final_probs = tf.identity(final_probs, 'final_probs')
            final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
            final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')

            if config.MODE_MASK:
                # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
                def f1():
                    if config.FPN:
                        roi_resized = roi_align_FPN(featuremap, final_boxes , 14)
                        feature_maskrcnn = roi_resized
                        mask_logits = maskrcnn_upXconv_head('maskrcnn', feature_maskrcnn, config.NUM_CLASS-1, 4)
                        #mask_logits = maskrcnn_head_FPN(
                        #    'maskrcnn', feature_maskrcnn, config.NUM_CLASS)   # #result x #cat x 14x14
                        indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                        final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx14x14
                        return tf.sigmoid(final_mask_logits)
                    else:
                        roi_resized = roi_align(featuremap, final_boxes * (1.0 / config.ANCHOR_STRIDE), 14)
                        feature_maskrcnn = resnet_conv5(roi_resized, config.RESNET_NUM_BLOCK[-1])
                        mask_logits = maskrcnn_head(
                            'maskrcnn', feature_maskrcnn, config.NUM_CLASS)   # #result x #cat x 14x14
                        indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                        final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx14x14
                        return tf.sigmoid(final_mask_logits)

                final_masks = tf.cond(tf.size(final_probs) > 0, f1, lambda: tf.zeros([0, 14, 14]))
                tf.identity(final_masks, name='final_masks')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate', lr)

        factor = get_batch_factor()
        if factor != 1:
            lr = lr / float(factor)
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            opt = optimizer.AccumGradOptimizer(opt, factor)
        else:
            opt = tf.train.MomentumOptimizer(lr, 0.9)
        # opt = tf.train.AdamOptimizer(lr)
        #opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def offline_evaluate(pred_func, output_file):
    df = get_test_dataflow(add_mask=False)
    all_results = eval_on_dataflow(
        df, lambda img: detect_one_image(img, pred_func))
    #print(all_results)
#    input()
    with open(output_file, 'w') as f:
        json.dump(all_results, f, cls=MyEncoder)
    ret = print_evaluation_scores(output_file)
    print(ret)

def clean_overlap_instance(predicts, scores):
    shape = np.array(predicts).shape # n * h * w
    if (shape[0] == 0):
        return []
    predicts = np.array(predicts)
    sort_ind = np.argsort(scores)[::-1]
    predicts = predicts[sort_ind]
    overlap = np.zeros((shape[1], shape[2]))
    # let the highest score to occupy pixel
    for mm in range(len(predicts)):
        mask = predicts[mm]
        overlap += mask
        mask[overlap>1] = 0
        predicts[mm] = mask
    # del mask if overlapped too much
    del_ind = np.where(np.sum(predicts, axis=(1,2)) < 1)[0]
    if len(del_ind)>0:
        if len(del_ind)<len(predicts):
            print('Empty mask, deleting', len(del_ind), 'masks')
            predicts = np.delete(predicts, del_ind, axis=0)
        else:
            predicts = np.zeros([1, shape[1], shape[2]])
            predicts[0,0,0] = 1
    return predicts

def rle_encoding(mask, shape=(768, 768)):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def mask_to_box(mask):
    # special for ship detection
    _, cnt, _ = cv2.findContours(mask, 1, 2)
    rect = cv2.minAreaRect(cnt[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(mask, [box], 0, 1, -1)
    return mask

def split_mask(mask):
    from scipy import ndimage
    threshold = 0
    threshold_obj = 30 #ignor predictions composed of "threshold_obj" pixels or less
    labled, n_objs = ndimage.label(mask > threshold)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if(obj.sum() > threshold_obj): result.append(obj)
    return result

def predict_many(pred_func, input_files):
    import time
    import pandas as pd
    with open("./id_no_shop_384_2.pkl", "rb") as f:
        id_no_ship = pickle.load(f)
    #fid = open('result.csv','w+')
    #fid.write('ImageId,EncodedPixels\n')
    ship_list_dict = []
    with tqdm.tqdm(total=len(input_files)) as pbar:
        for idx, imgData in enumerate(input_files):
            img = cv2.imread(imgData[0])
            filename = imgData[1]
            ImageId = filename
            if ImageId in id_no_ship:
                ship_list_dict.append({'ImageId':ImageId,'EncodedPixels':np.nan})
                pbar.update()
                continue
            #s = time.time()
            results = detect_one_image(img.copy(), pred_func)
            mask_instances = [r.mask for r in results]
            score_instances = [r.score for r in results]
            #mask_whole = detect_one_image_TTA(img.copy(), pred_func)
            #print(time.time() - s)
            """
            if (len(results) == 0):
                # no detection in image
                result_one_line = ImageId+','+ "" +'\n'
                ship_list_dict.append({'ImageId':ImageId,'EncodedPixels':np.nan})
                #fid.write(result_one_line)
                pbar.update()
                continue
            
            #boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
            #class_ids: [N] Integer class IDs for each bounding box
            #scores: [N] Float probability scores of the class_id
            #masks: [height, width, num_instances] Instance masks
            
            r = {}
            r['class_ids'] = []
            r['scores'] = []
            r['masks'] = []
            for det in results:
                if np.count_nonzero(det.mask) > 0:
                    r['class_ids'].append(det.class_id)
                    r['scores'].append(det.score)
                    r['masks'].append(det.mask)
            if len(r['masks']) == 0:
                ship_list_dict.append({'ImageId':ImageId,'EncodedPixels':np.nan})
                print('no_mask')
                pbar.update()
                continue
            
            r['masks']  = np.array(r['masks']) # n, h, w
            #r['masks'] = np.transpose(r['masks'], [1,2,0])
            ImageId = filename
            #print(filename, r['masks'].shape)
            LabelId = r['class_ids'] 
            mask_whole = np.zeros((img.shape[0], img.shape[1]))
            #mask_clean = clean_overlap_instance(r['masks'], r['scores'])
            mask_clean = r['masks']
            for i in range(mask_clean.shape[0]):
                #box_mask = mask_to_box(mask_clean[i])
                mask_whole[mask_clean[i] > 0] = 1
                #EncodedPixels = rle_encoding(mask_clean[i])
                #ship_list_dict.append({'ImageId':ImageId,'EncodedPixels':EncodedPixels})
            """
            #masks = split_mask(mask_whole)
            masks = clean_overlap_instance(mask_instances, score_instances)
            if len(masks) == 0:
                ship_list_dict.append({'ImageId':ImageId,'EncodedPixels':np.nan})
                print('no_mask!!!')
                pbar.update()
                continue
            for mask in masks:
                ship_list_dict.append({'ImageId':ImageId,'EncodedPixels':rle_encoding(mask)})
            #if idx < 30:
            #    cv2.imwrite(os.path.join("output", filename), mask_whole*255)
            pbar.update()
        #fid.close()
        pred_df = pd.DataFrame(ship_list_dict)
        pred_df = pred_df[['ImageId','EncodedPixels']]
        pred_df.to_csv('submission.csv', index=False)
        print('done!')

class EvalCallback(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'],
            get_model_output_names())
        self.df = get_test_dataflow(add_mask=True)

    def _before_train(self):
        EVAL_TIMES = 5  # eval 5 times during training
        interval = self.trainer.max_epoch // (EVAL_TIMES + 1)
        self.epochs_to_eval = set([interval * k for k in range(1, EVAL_TIMES)])
        self.epochs_to_eval.add(self.trainer.max_epoch)

    def _eval(self):
        all_results, local_score = eval_on_dataflow(self.df, lambda img: detect_one_image(img, self.pred))
        """
        output_file = os.path.join(
            logger.get_logger_dir(), 'outputs{}.json'.format(self.global_step))
        with open(output_file, 'w') as f:
            json.dump(all_results, f, cls=MyEncoder)
        scores = print_evaluation_scores(output_file)
        """
        scores = {}
        scores['local'] = local_score
        for k, v in scores.items():
            self.trainer.monitors.put_scalar(k, v)

    def _trigger_epoch(self):
        if self.epoch_num > 0 and self.epoch_num % 10 == 0:
            self._eval()

class ResnetEvalCallback(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'],
            get_resnet_model_output_names())
        self.df = get_resnet_val_dataflow()

    def _eval(self):
        from tensorpack.utils.utils import get_tqdm_kwargs
        score = 0.0
        ind = 0.0
        self.df.reset_state()
        with tqdm.tqdm(total=self.df.size(), **get_tqdm_kwargs()) as pbar:
            for img, la in self.df.get_data():
                ind += 1
                final_probs, final_labels = self.pred(img)
                score += (np.array(final_labels) == np.array(la)).mean()
                pbar.update()
        print("Val Acc: ", score / ind)
        self.trainer.monitors.put_scalar("Acc", score / ind)

    def _trigger_epoch(self):
        #if self.epoch_num > 0 and self.epoch_num % 5 == 0:
        self._eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help='logdir', default='train_log/fastrcnn')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', help='path to the output json eval file')
    parser.add_argument('--predict', help='path to the input image file')
    parser.add_argument('--lr_find', action='store_true')
    parser.add_argument('--cyclic', action='store_true')
    args = parser.parse_args()
    if args.datadir:
        config.BASEDIR = args.datadir

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize or args.evaluate or args.predict:
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

        assert args.load
        print_config()

        #pred = OfflinePredictor(PredictConfig(
        #    model=Model(),
        #    session_init=get_model_loader(args.load),
        #    input_names=['image'],
        #    output_names=get_model_output_names()))

        # imgs = OARDetection.load_many(config.BASEDIR, config.TEST_DATASET, add_gt=False)

        if args.visualize:
            visualize(args.load)
#            imgs = [img['file_name'] for img in imgs]
#            predict_many(pred, imgs)
        else:
            if args.evaluate:
                if config.RESNET:
                    pred = OfflinePredictor(PredictConfig(
                        model=ResnetModel(),
                        session_init=get_model_loader(args.load),
                        input_names=['image'],
                        output_names=get_resnet_model_output_names()))
                    images = ResnetDetection.load_many(config.BASEDIR, config.VAL_DATASET) 
                    images = list(images)
                    imgs = [img['image_data'] for img in images]
                    labels = [img['with_ship'] for img in images]
                    from tensorpack.utils.utils import get_tqdm_kwargs
                    score = 0.0
                    ind = 0.0
                    b_size = config.RESNET_BATCH
                    final_pred_mask = []
                    with tqdm.tqdm(total=len(imgs)//b_size+1, **get_tqdm_kwargs()) as pbar:
                        for i in range(len(imgs)//b_size + 1):
                            ind += 1
                            start = i * b_size
                            end = start + b_size if start + b_size < len(imgs) else len(imgs)
                            batch_image = imgs[start:end]
                            batch_image = np.array([cv2.resize(cv2.imread(im),(config.RESNET_SIZE,config.RESNET_SIZE)) for im in batch_image])
                            batch_label = np.array(labels[start:end])
                            final_probs, final_labels = pred(batch_image)
                            final_pred_mask += list(final_labels)
                            score += (np.array(final_labels) == np.array(batch_label)).mean()
                            pbar.update()
                    print("Val Acc: ", score / (ind), len(final_pred_mask))
                else:
                    pred = OfflinePredictor(PredictConfig(
                        model=Model(),
                        session_init=get_model_loader(args.load),
                        input_names=['image'],
                        output_names=get_model_output_names()))
                    df = get_test_dataflow(add_mask=True)
                    df.reset_state()
                    all_results, local_score = eval_on_dataflow(df, lambda img: detect_one_image_TTA(img, pred))
                    print("F2 Score: ", local_score)
                    
            elif args.predict:
                if config.RESNET:
                    pred = OfflinePredictor(PredictConfig(
                        model=ResnetModel(),
                        session_init=get_model_loader(args.load),
                        input_names=['image'],
                        output_names=get_resnet_model_output_names()))
                    images = ResnetDetection.load_many(config.BASEDIR, config.TEST_DATASET) 
                    images = list(images)
                    imgs = [img['image_data'] for img in images]
                    image_ids = [img['id'] for img in images]
                    from tensorpack.utils.utils import get_tqdm_kwargs
                    score = 0.0
                    ind = 0.0
                    b_size = config.RESNET_BATCH
                    final_pred_mask = []
                    with tqdm.tqdm(total=len(imgs)//b_size+1, **get_tqdm_kwargs()) as pbar:
                        for i in range(len(imgs)//b_size + 1):
                            ind += 1
                            start = i * b_size
                            end = start + b_size if start + b_size < len(imgs) else len(imgs)
                            batch_image = imgs[start:end]
                            batch_image = np.array([cv2.resize(cv2.imread(im),(config.RESNET_SIZE,config.RESNET_SIZE)) for im in batch_image])
                            final_probs, final_labels = pred(batch_image)
                            final_pred_mask += list(final_labels)
                            pbar.update()
                    id_ship = [_id for _id, la in zip(image_ids, final_pred_mask) if la > 0]
                    id_no_ship = [_id for _id, la in zip(image_ids, final_pred_mask) if la == 0]
                    print(id_ship[0])
                    with open("./id_shop.pkl", "wb") as f:
                        pickle.dump(id_ship, f)
                    with open("./id_no_shop.pkl", "wb") as f:
                        pickle.dump(id_no_ship, f)
                else:
                    imgs = Detection.load_many(config.BASEDIR, config.TEST_DATASET, add_gt=False)   # to load the class names into caches
                    # filter with zero-ship
                    imgs = [(img['image_data'], img['id']) for img in imgs]
                    pred = OfflinePredictor(PredictConfig(
                        model=Model(),
                        session_init=get_model_loader(args.load),
                        input_names=['image'],
                        output_names=get_model_output_names()))
                    predict_many(pred, imgs)
    else:
       
        if args.lr_find:
            base_lr = 0.0001
            max_lr = 0.01
            stepnum = 1000
            max_epoch = 20 # run 20 to find lr
            schedule = [(0, base_lr)]
            for e in range(1, max_epoch):
                offset = (max_lr-base_lr)/(max_epoch-1)
                schedule.append((e, base_lr+offset*e))
            LR_RANGE_TEST_SCHEDULE = ScheduledHyperParamSetter('learning_rate', schedule)
            TRAINING_SCHEDULE = LR_RANGE_TEST_SCHEDULE
        elif args.cyclic:
            from custom_utils import CyclicLearningRateSetter
            if config.RESNET:
                base_lr = 0.0001
                max_lr = 0.004
                step_size = (180000 // 64) * 2
            else:
                base_lr = 0.0001
                max_lr = 0.003
                step_size = (5000 // 1) * 2
            # Current 60000 steps to reach 0.7 LB
            stepnum = 5000 # step to save model and eval
            max_epoch = 20 # how many cycle / 4 = 5 cycle (2*step_size = 1 cycle)
            CYCLIC_SCHEDULE = CyclicLearningRateSetter('learning_rate', base_lr=base_lr, max_lr=max_lr, step_size=step_size)
            TRAINING_SCHEDULE = CYCLIC_SCHEDULE
        else:
            # heuristic setting for baseline
            if config.RESNET:
                stepnum = 2000
                max_epoch = 40
                TRAINING_SCHEDULE = ScheduledHyperParamSetter('learning_rate', [(0, 4e-3), (10, 1e-3), (30, 1e-4)])
            else:
                stepnum = 2000
                max_epoch = 120
                TRAINING_SCHEDULE = ScheduledHyperParamSetter('learning_rate', [(0, 3e-3), (40, 1e-3), (80, 1e-4)])

        #==========LR Range Test===============#
        if config.RESNET:
            logger.set_logger_dir(args.logdir)
            print_config()
            #stepnum = 2000
            # stepnum = 10000
            #warmup_epoch = max(math.ceil(500.0 / stepnum), 5)
            #factor = get_batch_factor()

            cfg = TrainConfig(
                model=ResnetModel(),
                data=QueueInput(get_resnet_train_dataflow()),
                callbacks=[
                    ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                    TRAINING_SCHEDULE,
                    ResnetEvalCallback(),
                    GPUUtilizationTracker(),
                ],
                steps_per_epoch=stepnum,
                max_epoch=max_epoch,
                session_init=get_model_loader(args.load) if args.load else None,
            )
            trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu(), mode='nccl')
            launch_train_with_config(cfg, trainer)
        else:
            logger.set_logger_dir(args.logdir)
            print_config()
            #stepnum = 2000
            # stepnum = 10000
            #warmup_epoch = max(math.ceil(500.0 / stepnum), 5)
            #factor = get_batch_factor()

            cfg = TrainConfig(
                model=Model(),
                data=QueueInput(get_train_dataflow(add_mask=config.MODE_MASK)),
                callbacks=[
                    ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                    # linear warmup
                    TRAINING_SCHEDULE,
                    EvalCallback(),
                    GPUUtilizationTracker(),
                ],
                steps_per_epoch=stepnum,
                max_epoch=max_epoch,
                session_init=get_model_loader(args.load) if args.load else None,
            )
            trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu(), mode='nccl')
            launch_train_with_config(cfg, trainer)
