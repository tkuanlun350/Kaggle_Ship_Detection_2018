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
from basemodel_tp import (
    resnet_fpn_backbone
)
from model import *
from data import (
    get_train_dataflow, get_test_dataflow,
    get_all_anchors, get_all_anchors_FPN, get_resnet_train_dataflow, get_resnet_val_dataflow)
from viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs, shrink_poly, draw_refined_outputs)
from common import print_config
from eval import (
    eval_on_dataflow, detect_one_image_TTA, detect_one_image, DetectionResult, print_evaluation_scores)
import config
from airbus import Detection, ResnetDetection
#import matplotlib.pyplot as plt
import collections
import nibabel as nib
#import SimpleITK as sitk

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

class Model(ModelDesc):
    def _get_inputs(self):
        ret = [
            InputDesc(tf.float32, (None, None, 3), 'image'),
            InputDesc(tf.int32, (None,), 'anchor_labels'), # stride*H*W*num_size
            InputDesc(tf.float32, (None, 4), 'anchor_boxes'),
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

    def get_fastrcnn_loss(self, proposals, fastrcnn_box_logits, fastrcnn_label_logits):
        ret = []
        BBOX_REG_WEIGHTS = [[10., 10., 5., 5.], [20., 20., 10., 10.], [30., 30., 15., 15.]]
        for idx, proposal in enumerate(proposals):
            with tf.name_scope('cascade_loss_stage{}'.format(idx + 1)):
                boxes, labels_per_box, fg_inds_wrt_gt, gt_boxes, gt_labels = proposal
                matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)
                fg_inds_wrt_sample = tf.reshape(tf.where(labels_per_box > 0), [-1])   # fg inds w.r.t all samples
                fg_sampled_boxes = tf.gather(boxes, fg_inds_wrt_sample)
                encoded_boxes = encode_bbox_target(
                    matched_gt_boxes,
                    fg_sampled_boxes) * tf.constant(BBOX_REG_WEIGHTS[idx])
                
                fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(
                    labels_per_box, fastrcnn_label_logits[idx],
                    encoded_boxes,
                    tf.gather(fastrcnn_box_logits[idx], fg_inds_wrt_sample))
                
                ret.extend([fastrcnn_label_loss, fastrcnn_box_loss])
        return ret

    def match_box_with_gt(self, boxes, gt_boxes, gt_labels, iou_threshold):
        from utils.box_ops import pairwise_iou
        if self.is_training:
            with tf.name_scope('match_box_with_gt_{}'.format(iou_threshold)):
                iou = pairwise_iou(boxes, gt_boxes)  # NxM
                max_iou_per_box = tf.reduce_max(iou, axis=1)  # N
                best_iou_ind = tf.argmax(iou, axis=1)  # N
                labels_per_box = tf.gather(gt_labels, best_iou_ind)
                fg_mask = max_iou_per_box >= iou_threshold
                fg_inds_wrt_gt = tf.boolean_mask(best_iou_ind, fg_mask)
                labels_per_box = tf.stop_gradient(labels_per_box * tf.to_int64(fg_mask))
                return [boxes, labels_per_box, fg_inds_wrt_gt, gt_boxes, gt_labels]
        else:
            return [boxes, None, None, None, None]

    def run_rcnn_head(self, featuremap, proposal_boxes, stage=0):
        BBOX_REG_WEIGHTS = [[10., 10., 5., 5.], [20., 20., 10., 10.], [30., 30., 15., 15.]]
        reg_weights = tf.constant(BBOX_REG_WEIGHTS[stage], dtype=tf.float32)
        boxes = proposal_boxes[0]
        boxes_on_featuremap = boxes
        if config.PAN:
            roi_resized = roi_align_PAN(featuremap, boxes_on_featuremap, 7)
            # scale gradient by 1./3 but not modified in forward pass
            roi_resized =  [(1 - 1./3)*tf.stop_gradient(roi) + 1./3*roi for roi in roi_resized]
            #roi_resize
        else:
            roi_resized = roi_align_FPN(featuremap, boxes_on_featuremap, 7)
            # scale gradient by 1./3 but not modified in forward pass
            roi_resized =  (1 - 1./3)*tf.stop_gradient(roi_resized) + 1./3*roi_resized
            #roi_resized = self.scale_gradient(roi_resized)
        def ff_true():
            # fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_head_FPN('fastrcnn', roi_resized, config.NUM_CLASS, class_agnostic_regression=True)
            if config.PAN:
                #head_feature = [fastrcnn_fc_head_fusion('head', roi) for roi in roi_resized]
                head_feature = fastrcnn_fc_head_PAN('head', roi_resized)
            else:
                head_feature = fastrcnn_2fc_head('head', roi_resized)
            fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(
                'outputs', head_feature, config.NUM_CLASS, class_agnostic_regression=True)
            return fastrcnn_label_logits, fastrcnn_box_logits

        def ff_false():
            ncls = config.NUM_CLASS
            return tf.zeros([0, ncls]), tf.zeros([0, 1, 4])

        fastrcnn_label_logits, fastrcnn_box_logits = tf.cond(
            tf.size(boxes_on_featuremap) > 0, ff_true, ff_false)

        refined_boxes_logits = tf.reshape(fastrcnn_box_logits, [-1, 4])
        refined_boxes = decode_bbox_target(
            refined_boxes_logits / reg_weights,
            boxes
        )
        # refined_boxes = head.decoded_output_boxes_class_agnostic()
        refined_boxes = clip_boxes(refined_boxes, self.image_shape2d)
        return fastrcnn_box_logits, fastrcnn_label_logits, tf.stop_gradient(refined_boxes, name='output_boxes')

    def _build_graph(self, inputs):
        is_training = get_current_tower_context().is_training
        self.is_training = is_training

        if config.MODE_MASK:
            image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, gt_masks = inputs
        else:
            image, anchor_labels, anchor_boxes, gt_boxes, gt_labels = inputs
        
        fm_anchors = self._get_anchors_FPN(image)
        image = self._preprocess(image)     # 1CHW
        image_shape2d = tf.shape(image)[2:]
        self.image_shape2d = image_shape2d
        #featuremap = pretrained_resnet_FPN(image, config.RESNET_NUM_BLOCK)
        featuremap = resnet_fpn_backbone(image, config.RESNET_NUM_BLOCK)
        rpn_label_logits_FPN, rpn_box_logits_FPN = rpn_head_FPN('rpn', featuremap, 256, len(config.ANCHOR_RATIOS))
        decoded_boxes = decode_bbox_target_FPN(rpn_box_logits_FPN, fm_anchors)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals_FPN(
            decoded_boxes, # layer-wise
            rpn_label_logits_FPN, # layer-wise
            image_shape2d)
        anchor_boxes_encoded = encode_bbox_target(anchor_boxes, tf.concat(fm_anchors, 0))     
        
        if is_training:
            # sample proposal boxes in training
            rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
                proposal_boxes, gt_boxes, gt_labels)
        
            proposals1 = [rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt, gt_boxes, gt_labels]
        else:
            proposals1 = [proposal_boxes, None, None, None, None]

        with tf.variable_scope('cascade_rcnn_stage1'):
            box_logits1, label_logit1, regressed_box1 = self.run_rcnn_head(featuremap, proposals1, 0)
        with tf.variable_scope('cascade_rcnn_stage2'):
            proposals2 = self.match_box_with_gt(regressed_box1, gt_boxes, gt_labels, 0.6)
            box_logits2, label_logit2, regressed_box2 = self.run_rcnn_head(featuremap, proposals2, 1)
        with tf.variable_scope('cascade_rcnn_stage3'):
            proposals3 = self.match_box_with_gt(regressed_box2, gt_boxes, gt_labels, 0.7)
            box_logits3, label_logit3, regressed_box3 = self.run_rcnn_head(featuremap, proposals3, 2)
        
        if is_training:
            # rpn loss
            rpn_label_loss, rpn_box_loss = rpn_losses(
                    anchor_labels, anchor_boxes_encoded, tf.concat(rpn_label_logits_FPN, 0), tf.concat(rpn_box_logits_FPN, 0))
           
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

            cascade_proposal_boxes = [proposals1, proposals2, proposals3]
            frcnn_box_logits = [box_logits1, box_logits2, box_logits3]
            frcnn_label_logits = [label_logit1, label_logit2, label_logit3]
            cascade_astrcnn_losses = self.get_fastrcnn_loss(cascade_proposal_boxes, frcnn_box_logits, frcnn_label_logits)
            

            if config.MODE_MASK:
                # maskrcnn loss
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
                mrcnn_loss = 0.0

            
            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(1e-4), name='wd_cost')
            all_losses = [rpn_label_loss, rpn_box_loss, mrcnn_loss, wd_cost]
            all_losses.extend(cascade_astrcnn_losses)
            self.cost = tf.add_n(all_losses, 'total_cost')                

            add_moving_summary(self.cost, wd_cost)
        else:
            # regressed_box3
            decoded_boxes = tf.tile(tf.expand_dims(regressed_box3, 1), [1, config.NUM_CLASS, 1])
            #label_probs = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
            #anchors = tf.tile(tf.expand_dims(proposal_boxes, 1), [1, config.NUM_CLASS - 1, 1])   # #proposal x #Cat x 4
            #decoded_boxes = decode_bbox_target(
            #    fastrcnn_box_logits /
            #    tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS), anchors)
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

            # indices: Nx2. Each index into (#proposal, #category)
            label_probs1 = tf.nn.softmax(label_logit1, name='casacde_stage1_label_prob')
            label_probs2 = tf.nn.softmax(label_logit2, name='casacde_stage2_label_prob')
            label_probs3 = tf.nn.softmax(label_logit3, name='casacde_stage3_label_prob')
            label_probs = tf.multiply(tf.add_n([label_probs1, label_probs2, label_probs3]), (1.0 / 3), name='fastrcnn_all_probs')
            pred_indices, final_probs = fastrcnn_predictions_cascade(decoded_boxes, label_probs)
            final_probs = tf.identity(final_probs, 'final_probs')
            final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
            final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')

            if config.MODE_MASK:
                # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
                def f1():
                    roi_resized = roi_align_FPN(featuremap, final_boxes , 14)
                    feature_maskrcnn = roi_resized
                    mask_logits = maskrcnn_upXconv_head(
                        'maskrcnn', feature_maskrcnn, config.NUM_CLASS-1, 4)   # #result x #cat x 14x14
                    indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                    final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx14x14
                    return tf.sigmoid(final_mask_logits)

                final_masks = tf.cond(tf.size(final_probs) > 0, f1, lambda: tf.zeros([0, 28*2, 28*2]))
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
        opt = tf.train.MomentumOptimizer(lr, 0.9)
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
    del_ind = np.where(np.sum(predicts, axis=(1,2)) < 30)[0]
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
            #mask_whole = detect_one_image(img.copy(), pred_func)

            """
            #print(time.time() - s)
            if (len(results) == 0):
                # no detection in image
                result_one_line = ImageId+','+ "" +'\n'
                ship_list_dict.append({'ImageId':ImageId,'EncodedPixels':np.nan})
                #fid.write(result_one_line)
                pbar.update()
                continue
            
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
                mask_whole[mask_clean[i] > 0] = 1
                #EncodedPixels = rle_encoding(mask_clean[i])
                #ship_list_dict.append({'ImageId':ImageId,'EncodedPixels':EncodedPixels})
            """
            masks = clean_overlap_instance(mask_instances, score_instances)
            #masks = split_mask(mask_whole)
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
        scores = {}
        scores['local'] = local_score
        for k, v in scores.items():
            self.trainer.monitors.put_scalar(k, v)

    def _trigger_epoch(self):
        if self.epoch_num > 0 and self.epoch_num % 10 == 0:
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

    if args.evaluate:
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['image'],
            output_names=get_model_output_names()))
        df = get_test_dataflow(add_mask=True)
        df.reset_state()
        all_results, local_score = eval_on_dataflow(df, lambda img: detect_one_image(img, pred))
        print("F2 Score: ", local_score)
            
    elif args.predict:
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
            stepnum = 2000
            max_epoch = 120
            TRAINING_SCHEDULE = ScheduledHyperParamSetter('learning_rate', [(0, 3e-3), (40, 1e-3), (80, 1e-4)])

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

