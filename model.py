#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.models import ( 
    Conv2DTranspose, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, layer_register, Deconv2D, Dropout)

from utils.box_ops import pairwise_iou
import numpy as np
import config
import math

@under_name_scope()
def cls_loss(label_logits, label):
    with tf.name_scope('cls_label_metrics'):
        label_pred = tf.round(tf.nn.sigmoid(label_logits))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(label_pred), tf.to_float(label)), tf.float32), name="accuracy")
        add_moving_summary(accuracy)

    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(label), logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')
    return label_loss

@layer_register(log_shape=True)
def cls_head(feature):
    feature = GlobalAvgPooling('gap', feature, data_format='NCHW')
    fc1 = FullyConnected(
        'fc1', feature, 1024,
        W_init=tf.random_normal_initializer(stddev=0.01))
    fc1 = Dropout(fc1)
    fc2 = FullyConnected(
        'fc2', fc1, 1,
        W_init=tf.random_normal_initializer(stddev=0.01))
    return tf.squeeze(fc2, [1])

@under_name_scope()
def clip_boxes(boxes, window, name=None):
    """
    Args:
        boxes: nx4, xyxy
        window: [h, w]
    """
    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(window, [0]), [2])    # (4,)
    boxes = tf.minimum(boxes, tf.to_float(m), name=name)
    return boxes

def rpn_head_FPN(prefix, featuremaps, channel, num_anchors):
    """
    label_logits: layers xfHxfWxNA
    box_logits: layers xfHxfWxNAx4
    """
    rpn_label_logits = []
    rpn_box_logits = []
    is_training = get_current_tower_context().is_training
    dropout_rate = 0.5 if not is_training else 0.0
    with tf.variable_scope(prefix):
        for layer in range(len(featuremaps)):
            print('layer', layer)
            with argscope(Conv2D, data_format='NCHW',
                        W_init=tf.random_normal_initializer(stddev=0.01)):
                featuremap = featuremaps[layer]
                #featuremap = tf.Print(featuremap, [tf.shape(featuremap)], message="FM_{}".format(layer))
                hidden = Conv2D('conv0', featuremap, channel, 3, nl=tf.nn.relu, padding='SAME')
                #hidden = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format='channels_first')(hidden)
                label_logits = Conv2D('class', hidden, num_anchors, 1)
                box_logits = Conv2D('box', hidden, 4 * num_anchors, 1)
                # 1, NA(*4), im/16, im/16 (NCHW)

                label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # 1xfHxfWxNA
                label_logits = tf.squeeze(label_logits, 0)  # fHxfWxNA

                shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW
                box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
                box_logits = tf.reshape(box_logits, tf.stack([shp[2], shp[3], num_anchors, 4]))  # fHxfWxNAx4

                #box_logits = tf.Print(box_logits, [tf.shape(box_logits)], message="rpn_box{}".format(layer))

                rpn_label_logits.append(tf.reshape(label_logits, [-1]))
                rpn_box_logits.append(tf.reshape(box_logits, [-1, 4]))
                tf.get_variable_scope().reuse_variables()

    return rpn_label_logits, rpn_box_logits

@layer_register(log_shape=True)
def rpn_head(featuremap, channel, num_anchors):
    """
    Returns:
        label_logits: fHxfWxNA
        box_logits: fHxfWxNAx4
    """
    with argscope(Conv2D, data_format='NCHW',
                  W_init=tf.random_normal_initializer(stddev=0.01)):
        hidden = Conv2D('conv0', featuremap, channel, 3, nl=tf.nn.relu)

        label_logits = Conv2D('class', hidden, num_anchors, 1)
        box_logits = Conv2D('box', hidden, 4 * num_anchors, 1)
        # 1, NA(*4), im/16, im/16 (NCHW)

        label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # 1xfHxfWxNA
        label_logits = tf.squeeze(label_logits, 0)  # fHxfWxNA

        shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW
        box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
        box_logits = tf.reshape(box_logits, tf.stack([shp[2], shp[3], num_anchors, 4]))  # fHxfWxNAx4
    return label_logits, box_logits


@under_name_scope()
def rpn_losses(anchor_labels, anchor_boxes, label_logits, box_logits):
    """
    Args:
        anchor_labels: fHxfWxNA
        anchor_boxes: fHxfWxNAx4, encoded
        label_logits:  fHxfWxNA
        box_logits: fHxfWxNAx4

    Returns:
        label_loss, box_loss
    """
    with tf.device('/cpu:0'):
        valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
        pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
        nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
        nr_pos = tf.count_nonzero(pos_mask, dtype=tf.int32, name='num_pos_anchor')
        
        valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

    with tf.name_scope('label_metrics'):
        valid_label_prob = tf.nn.sigmoid(valid_label_logits)
        summaries = []
        with tf.device('/cpu:0'):
            for th in [0.5, 0.2, 0.1]:
                valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
                nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
                pos_prediction_corr = tf.count_nonzero(
                    tf.logical_and(
                        valid_label_prob > th,
                        tf.equal(valid_prediction, valid_anchor_labels)),
                    dtype=tf.int32)
                summaries.append(tf.truediv(
                    pos_prediction_corr,
                    nr_pos, name='recall_th{}'.format(th)))
                precision = tf.to_float(tf.truediv(pos_prediction_corr, nr_pos_prediction))
                precision = tf.where(tf.equal(nr_pos_prediction, 0), 0.0, precision, name='precision_th{}'.format(th))
                summaries.append(precision)
        add_moving_summary(*summaries)
    
    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(valid_anchor_labels), logits=valid_label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')
    """
    alpha = 0.25
    gamma = 2
    sigmoid_p = tf.nn.sigmoid(valid_label_logits)
    #valid_labels = tf.cast(tf.one_hot(valid_anchor_labels, config.NUM_CLASS), tf.float32)
    valid_labels = tf.cast(valid_anchor_labels, tf.float32)
    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = tf.where(valid_labels >= sigmoid_p, valid_labels - sigmoid_p, zeros)
    neg_p_sub = tf.where(valid_labels > zeros, zeros, sigmoid_p)
    label_loss_focal = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                        - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    label_loss_focal = tf.reduce_mean(label_loss_focal)
    label_loss = label_loss + label_loss_focal
    """
    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    delta = 1.0 / 9
    box_loss = tf.losses.huber_loss(
        pos_anchor_boxes, pos_box_logits, delta=delta,
        reduction=tf.losses.Reduction.SUM) / delta
    box_loss = tf.div(
        box_loss,
        tf.cast(nr_valid, tf.float32), name='box_loss')

    add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return label_loss, box_loss

@under_name_scope()
def decode_bbox_target_FPN(box_predictions_FPN, anchors_FPN):
    """
    Args:
        box_predictions: [P2...P6][(..., 4)], logits
        anchors: (..., 4), floatbox. Must have the same shape

    Returns:
        box_decoded: (..., 4), float32. With the same shape.
    """
    layer_boxes = []
    # P2,3,4,5,6
    stride = config.FPN_STRIDES
    for layer in range(len(box_predictions_FPN)):
        box_predictions = box_predictions_FPN[layer]
        anchors = anchors_FPN[layer]
        orig_shape = tf.shape(anchors)
        box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
        box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
        # each is (...)x1x2
        anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
        anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

        waha = anchors_x2y2 - anchors_x1y1
        xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

        wbhb = tf.exp(tf.minimum(
            box_pred_twth, np.log(config.MAX_SIZE / stride[layer]))) * waha
        xbyb = box_pred_txty * waha + xaya
        x1y1 = xbyb - wbhb * 0.5
        x2y2 = xbyb + wbhb * 0.5    # (...)x1x2
        out = tf.concat([x1y1, x2y2], axis=-2)
        layer_boxes.append(tf.reshape(out, orig_shape))
    return layer_boxes

@under_name_scope()
def decode_bbox_target(box_predictions, anchors):
    """
    Args:
        box_predictions: (..., 4), logits
        anchors: (..., 4), floatbox. Must have the same shape

    Returns:
        box_decoded: (..., 4), float32. With the same shape.
    """
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
    # each is (...)x1x2
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    wbhb = tf.exp(tf.minimum(
        box_pred_twth, config.BBOX_DECODE_CLIP)) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5    # (...)x1x2
    out = tf.concat([x1y1, x2y2], axis=-2)
    return tf.reshape(out, orig_shape)

@under_name_scope()
def encode_bbox_target(boxes, anchors):
    """
    Args:
        boxes: (..., 4), float32
        anchors: (..., 4), float32

    Returns:
        box_encoded: (..., 4), float32 with the same shape.
    """
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)
    wbhb = boxes_x2y2 - boxes_x1y1
    xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

    # Note that here not all boxes are valid. Some may be zero
    txty = (xbyb - xaya) / waha
    twth = tf.log(wbhb / waha)  # may contain -inf for invalid boxes
    encoded = tf.concat([txty, twth], axis=1)  # (-1x2x2)
    return tf.reshape(encoded, tf.shape(boxes))

@under_name_scope()
def generate_rpn_proposals_FPN(boxesFPN, scoresFPN, img_shape):
    """
    Args:
        boxes: nx4 float dtype, decoded to floatbox already
        scores: n float, the logits
        img_shape: [h, w]

    Returns:
        boxes: kx4 float
        scores: k logits
    """
    #assert boxesFPN.shape.ndims == 2, boxesFPN.shape
    if get_current_tower_context().is_training:
        PRE_NMS_TOPK = 2000 # per FPN level
        POST_NMS_TOPK = config.TRAIN_POST_NMS_TOPK # 2000
    else:
        PRE_NMS_TOPK = 1000
        POST_NMS_TOPK = config.TEST_POST_NMS_TOPK
    layer_topk = []
    layer_topk_scores = []
    for layer in range(len(boxesFPN)):
        boxes = boxesFPN[layer]
        scores = scoresFPN[layer]
        topk = tf.minimum(PRE_NMS_TOPK, tf.size(scores))
        topk_scores_per_layer, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
        topk_boxes_per_layer = tf.gather(boxes, topk_indices)
        topk_boxes_per_layer = clip_boxes(topk_boxes_per_layer, img_shape)
        layer_topk.append(topk_boxes_per_layer)
        layer_topk_scores.append(topk_scores_per_layer)
    # collect and flatten
    topk_boxes = tf.concat(layer_topk, 0)
    # topk_boxes = tf.Print(topk_boxes, [tf.shape(topk_boxes)], message="topk_boxes")
    topk_scores = tf.concat(layer_topk_scores, 0)
    # topk_scores = tf.Print(topk_scores, [tf.shape(topk_scores)], message="topk_scores")
    topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
    topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
    # nx1x2 each
    wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
    valid = tf.reduce_all(wbhb > config.RPN_MIN_SIZE, axis=1)  # n,
    topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
    topk_valid_scores = tf.boolean_mask(topk_scores, valid)

    topk_valid_boxes_y1x1y2x2 = tf.reshape(
        tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
        (-1, 4), name='nms_input_boxes')
    nms_indices = tf.image.non_max_suppression(
        topk_valid_boxes_y1x1y2x2,
        topk_valid_scores,
        max_output_size=POST_NMS_TOPK,
        iou_threshold=config.RPN_PROPOSAL_NMS_THRESH)

    topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
    final_boxes = tf.gather(
        topk_valid_boxes,
        nms_indices, name='boxes')
    final_scores = tf.gather(topk_valid_scores, nms_indices, name='scores')
    tf.sigmoid(final_scores, name='probs')  # for visualization
    return final_boxes, final_scores

@under_name_scope()
def generate_rpn_proposals(boxes, scores, img_shape):
    """
    Args:
        boxes: nx4 float dtype, decoded to floatbox already
        scores: n float, the logits
        img_shape: [h, w]

    Returns:
        boxes: kx4 float
        scores: k logits
    """
    assert boxes.shape.ndims == 2, boxes.shape
    if get_current_tower_context().is_training:
        PRE_NMS_TOPK = config.TRAIN_PRE_NMS_TOPK
        POST_NMS_TOPK = config.TRAIN_POST_NMS_TOPK
    else:
        PRE_NMS_TOPK = config.TEST_PRE_NMS_TOPK
        POST_NMS_TOPK = config.TEST_POST_NMS_TOPK

    topk = tf.minimum(PRE_NMS_TOPK, tf.size(scores))
    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
    topk_boxes = tf.gather(boxes, topk_indices)
    topk_boxes = clip_boxes(topk_boxes, img_shape)

    topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
    topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
    # nx1x2 each
    wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
    valid = tf.reduce_all(wbhb > config.RPN_MIN_SIZE, axis=1)  # n,
    topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
    topk_valid_scores = tf.boolean_mask(topk_scores, valid)

    topk_valid_boxes_y1x1y2x2 = tf.reshape(
        tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
        (-1, 4), name='nms_input_boxes')
    nms_indices = tf.image.non_max_suppression(
        topk_valid_boxes_y1x1y2x2,
        topk_valid_scores,
        max_output_size=POST_NMS_TOPK,
        iou_threshold=config.RPN_PROPOSAL_NMS_THRESH)

    topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
    final_boxes = tf.gather(
        topk_valid_boxes,
        nms_indices, name='boxes')
    final_scores = tf.gather(topk_valid_scores, nms_indices, name='scores')
    tf.sigmoid(final_scores, name='probs')  # for visualization
    return final_boxes, final_scores


@under_name_scope()
def proposal_metrics(iou):
    """
    Add summaries for RPN proposals.

    Args:
        iou: nxm, #proposal x #gt
    """
    # find best roi for each gt, for summary only
    best_iou = tf.reduce_max(iou, axis=0)
    mean_best_iou = tf.reduce_mean(best_iou, name='best_iou_per_gt')
    summaries = [mean_best_iou]
    with tf.device('/cpu:0'):
        for th in [0.3, 0.5]:
            recall = tf.truediv(
                tf.count_nonzero(best_iou >= th),
                tf.size(best_iou, out_type=tf.int64),
                name='recall_iou{}'.format(th))
            summaries.append(recall)
    add_moving_summary(*summaries)

@under_name_scope()
def sample_fast_rcnn_targets_FPN(boxes, gt_boxes, gt_labels, roi_resized):
    """
    Sample some ROIs from all proposals for training.

    Args:
        boxes: nx4 region proposals, floatbox
        gt_boxes: mx4, floatbox
        gt_labels: m, int32
        roi_resized: n*7*7*256

    Returns:
        sampled_boxes: tx4 floatbox, the rois
        sampled_labels: t labels, in [0, #class-1]. Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
            It contains the matching GT of each foreground roi.
    """
    iou = pairwise_iou(boxes, gt_boxes)     # nxm
    proposal_metrics(iou)

    # add ground truth as proposals as well
    boxes = tf.concat([boxes, gt_boxes], axis=0)    # (n+m) x 4
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)   # (n+m) x m
    # #proposal=n+m from now on

    def sample_fg_bg(iou):
        fg_mask = tf.reduce_max(iou, axis=1) >= config.FASTRCNN_FG_THRESH

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            config.FASTRCNN_BATCH_PER_IM * config.FASTRCNN_FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            config.FASTRCNN_BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)
        return fg_inds, bg_inds

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals

    best_iou_ind = tf.argmax(iou, axis=1)   # #proposal, each in 0~m-1
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # num_fg

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)   # indices w.r.t all n+m proposal boxes
    ret_boxes = tf.gather(boxes, all_indices, name='sampled_proposal_boxes')
    ###
    roi_resized = tf.gather(roi_resized, all_indices, name='sampled_roi_feature')
    #roi_resized = tf.Print(roi_resized, [tf.shape(roi_resized)], name="roi_resized_sampled")
    ###
    ret_labels = tf.concat(
        [tf.gather(gt_labels, fg_inds_wrt_gt),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0, name='sampled_labels')
    # stop the gradient -- they are meant to be ground-truth
    return tf.stop_gradient(ret_boxes), tf.stop_gradient(ret_labels), fg_inds_wrt_gt, roi_resized

@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels):
    """
    Sample some ROIs from all proposals for training.

    Args:
        boxes: nx4 region proposals, floatbox
        gt_boxes: mx4, floatbox
        gt_labels: m, int32

    Returns:
        sampled_boxes: tx4 floatbox, the rois
        sampled_labels: t labels, in [0, #class-1]. Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
            It contains the matching GT of each foreground roi.
    """
    iou = pairwise_iou(boxes, gt_boxes)     # nxm
    proposal_metrics(iou)

    # add ground truth as proposals as well
    boxes = tf.concat([boxes, gt_boxes], axis=0)    # (n+m) x 4
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)   # (n+m) x m
    # #proposal=n+m from now on

    def sample_fg_bg(iou):
        fg_mask = tf.reduce_max(iou, axis=1) >= config.FASTRCNN_FG_THRESH

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            config.FASTRCNN_BATCH_PER_IM * config.FASTRCNN_FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            config.FASTRCNN_BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)
        return fg_inds, bg_inds

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals

    best_iou_ind = tf.argmax(iou, axis=1)   # #proposal, each in 0~m-1
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # num_fg

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)   # indices w.r.t all n+m proposal boxes
    #all_inds_wrt_gt = tf.gather(best_iou_ind, all_indices)

    ret_boxes = tf.gather(boxes, all_indices, name='sampled_proposal_boxes')

    ret_labels = tf.concat(
        [tf.gather(gt_labels, fg_inds_wrt_gt),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0, name='sampled_labels')
    # stop the gradient -- they are meant to be ground-truth
    return tf.stop_gradient(ret_boxes), tf.stop_gradient(ret_labels), tf.stop_gradient(fg_inds_wrt_gt)

@under_name_scope()
def sample_fast_rcnn_targets_RELATION(boxes, gt_boxes, gt_labels):
    iou = pairwise_iou(boxes, gt_boxes)     # nxm
    proposal_metrics(iou)

    nongt = tf.zeros(tf.expand_dims(tf.shape(boxes)[0], 0))
    gt = tf.ones(tf.expand_dims(tf.shape(gt_boxes)[0], 0))
    gt_nongt = tf.concat([nongt, gt], axis=0)
    proposal_boxes_num = tf.shape(boxes)[0]

    #-------------------
    if not config.SAMPLING:
        # use all proposal boxes
        fg_mask = tf.reduce_max(iou, axis=1) >= config.FASTRCNN_FG_THRESH
        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_fg = tf.size(fg_inds, name="num_fg")
        num_bg = tf.size(bg_inds, name="num_bg")
        add_moving_summary(num_fg, num_bg)
    else:
        #-------------------
        # sample 3:1 rate
        def sample_fg_bg(iou):
            fg_mask = tf.reduce_max(iou, axis=1) >= config.FASTRCNN_FG_THRESH

            fg_inds = tf.reshape(tf.where(fg_mask), [-1])
            num_fg = tf.minimum(int(
                config.FASTRCNN_BATCH_PER_IM * config.FASTRCNN_FG_RATIO),
                tf.size(fg_inds), name='num_fg')
            fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

            bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
            num_bg = tf.minimum(
                config.FASTRCNN_BATCH_PER_IM - num_fg,
                tf.size(bg_inds), name='num_bg')
            bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

            add_moving_summary(num_fg, num_bg)
            return fg_inds, bg_inds
        fg_inds, bg_inds = sample_fg_bg(iou)
        #nongt_indices = tf.concat([fg_inds, bg_inds], axis=0)
        #sampled_proposal_scores = tf.gather(proposal_scores, nongt_indices)
        #---------------------
    boxes = tf.concat([boxes, gt_boxes], axis=0)    # (n+m) x 4
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)   # (n+m) x m

    gt_inds = tf.cast(tf.range(proposal_boxes_num, tf.shape(boxes)[0], dtype=tf.int32), tf.int64)
    fg_inds = tf.concat([fg_inds, gt_inds], axis=0)

    best_iou_ind = tf.argmax(iou, axis=1)   # #proposal, each in 0~m-1
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # num_fg

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)   # indices w.r.t all n+m proposal boxes
    all_inds_wrt_gt = tf.gather(best_iou_ind, all_indices)

    ret_boxes = tf.gather(boxes, all_indices)

    ret_labels = tf.concat(
        [tf.gather(gt_labels, fg_inds_wrt_gt),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)

    gt_nongt_after_sampling = tf.gather(gt_nongt, all_indices)
    nongt_after_sampling = tf.reshape(tf.where(tf.equal(gt_nongt_after_sampling, 0)), [-1])
    gt_after_sampling = tf.reshape(tf.where(tf.equal(gt_nongt_after_sampling, 1)), [-1])

    gt_nongt_wrt_retboxes = tf.concat([nongt_after_sampling, gt_after_sampling], axis=0)
    ret_boxes = tf.gather(ret_boxes, gt_nongt_wrt_retboxes, name='sampled_proposal_boxes')
    ret_labels = tf.gather(ret_labels, gt_nongt_wrt_retboxes, name='sampled_labels')

    
    # stop the gradient -- they are meant to be ground-truth
    return tf.stop_gradient(ret_boxes), tf.stop_gradient(ret_labels), tf.stop_gradient(fg_inds_wrt_gt), tf.stop_gradient(all_inds_wrt_gt)

@under_name_scope()
def crop_and_resize(image, boxes, box_ind, crop_size):
    """
    Better-aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Args:
        image: NCHW
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size

    @under_name_scope()
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.

        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample assumes floating point coordinate (0.0, 0.0) is the same as pixel value (0, 0))

        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize

        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
        spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

        nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
        ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

        nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(image_shape[1] - 1)
        nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(image_shape[0] - 1)

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(image)[2:]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])   # 1hwc
    ret = tf.image.crop_and_resize(
        image, boxes, box_ind,
        crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
    return ret

@under_name_scope()
def roi_align_FPN(featuremaps, boxes, output_shape):
    """
    Args:
        featuremap: [1xCxHxW] * 5 P6~P2
        boxes: Nx4 floatbox
        output_shape: int

    Returns:
        NxCxoHxoW
    """
    
    def tf_log2(x):
        return tf.log(x) / tf.log(2.0)
    # feature map [P6, P5, P4, P3, P2]
    boxes = tf.stop_gradient(tf.expand_dims(boxes, 0))  # TODO
    x1, y1, x2, y2 = tf.split(boxes, 4, axis=2) # num * x1y1x2y2
    w = tf.maximum(x2 - x1, 0)
    h = tf.maximum(y2 - y1, 0)
    roi_level = tf_log2(tf.sqrt(h * w + 1e-8) / (224.0))
    roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
    roi_level = tf.squeeze(roi_level, 2)
    roi_level = tf.stop_gradient(roi_level)
    #roi_level = tf.Print(roi_level, [roi_level], message="This is roi_level: ")
    # limit to P5 ~ P2
    pooled = tf.zeros([0, 256, output_shape, output_shape])
    inds = tf.cast(tf.zeros([0, 1]), tf.int32)
    proposals = []
    strides = [4., 8., 16., 32.]
    for level in range(2, 6):
        featuremap_to_crop = featuremaps[4 - (level - 2)]
        # order : P6(idx 4) ~ P2(idx0) => idx shift
        # P2 -> f[4], P3 -> f[3], P4 -> f[2], P5 -> f[1]
        id_for_box_wrt_level = tf.where(tf.equal(roi_level, level))
        level_boxes = tf.gather_nd(boxes, id_for_box_wrt_level)
        box_indices = tf.reshape(tf.cast(id_for_box_wrt_level[:,1], tf.int32), [-1, 1])
        #level_boxes = tf.Print(level_boxes, [tf.shape(level_boxes)], message="level_boxes_{}".format(level))
        level_boxes = tf.stop_gradient(level_boxes)
        box_indices = tf.stop_gradient(box_indices)
        
        #box_indices = tf.Print(box_indices, [tf.shape(box_indices), box_indices], message="box_indices: ")
        def ff_true(level_boxes, level, featuremap_to_crop, pooled, box_indices, inds):
            level_boxes = level_boxes * (1.0 / strides[level - 2])
            ret = crop_and_resize(
                    featuremap_to_crop, level_boxes,
                    tf.zeros([tf.shape(level_boxes)[0]], dtype=tf.int32),
                    output_shape * 2)
            ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
            pooled_return = tf.cond(tf.size(pooled) > 0, lambda: tf.concat([pooled, ret], axis=0), lambda: ret)
            inds_return = tf.cond(tf.size(inds) > 0, lambda: tf.concat([inds, box_indices], axis=0), lambda: box_indices)
            pooled_return = tf.identity(pooled_return)
            inds_return = tf.identity(inds_return)
            ### identity to prevent tf.cond cause summary crash
            return pooled_return, inds_return

        def ff_false(pooled, inds):
            #ret = tf.cast(tf.zeros([0, 256, 7, 7]), tf.float32)
            return tf.identity(pooled), tf.identity(inds)

        pooled, inds = tf.cond(tf.size(level_boxes) > 0, 
                        lambda: ff_true(level_boxes, level, 
                                            featuremap_to_crop, pooled,
                                            box_indices, inds), 
                        lambda: ff_false(pooled, inds))

    #pooled = tf.concat(pooled, axis=0)
    inds = tf.reshape(inds, [-1, 1])
    print(inds.get_shape())
    pooled = tf.scatter_nd(inds, pooled, tf.shape(pooled))
    #pooled = tf.Print(pooled, [tf.shape(pooled)], message="pooled: ")
    #inds = tf.Print(inds, [tf.shape(inds), inds], message="inds: ")
    print(pooled.get_shape())
    # Find a way to get id back to original shape to match label
    # Or gather label to match box

    return pooled

@under_name_scope()
def roi_align(featuremap, boxes, output_shape):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        output_shape: int

    Returns:
        NxCxoHxoW
    """
    boxes = tf.stop_gradient(boxes)  # TODO
    # sample 4 locations per roi bin
    ret = crop_and_resize(
        featuremap, boxes,
        tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
        output_shape * 2)
    ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return ret

@layer_register(log_shape=True)
def fastrcnn_head_RELATION(feature, position_embedding, nongt_dim, num_classes):
    with argscope([Conv2D], data_format='NCHW'):
        fc1 = Conv2D('rcnn_fc1', feature, 1024, 7, nl=BNReLU, padding='VALID') #1*1
        attention_1 = attention_module_multi_head('att1', fc1, position_embedding,
                                                nongt_dim=nongt_dim, fc_dim=16, feat_dim=1024,
                                                index=1, group=16,
                                                dim=(1024, 1024, 1024))
        fc1 = tf.nn.relu(fc1 + attention_1)
        fc2 = Conv2D('rcnn_fc2', fc1, 1024, 1, nl=BNReLU, padding='SAME')
        attention_2 = attention_module_multi_head('att2', fc2, position_embedding,
                                                nongt_dim=nongt_dim, fc_dim=16, feat_dim=1024,
                                                index=1, group=16,
                                                dim=(1024, 1024, 1024))
        fc2 = tf.nn.relu(fc2 + attention_2)

        fc2 = tf.squeeze(fc2, [2, 3])
   
    # 1*1024*1*1
    classification = FullyConnected(
        'class', fc2, num_classes,
        W_init=tf.random_normal_initializer(stddev=0.01))
    box_regression = FullyConnected(
        'box', fc2, (num_classes - 1) * 4,
        W_init=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, (-1, num_classes - 1, 4))
    return classification, box_regression

@layer_register(log_shape=True)
def fastrcnn_head_FPN(feature, num_classes, class_agnostic_regression=False):
    with argscope([Conv2D], data_format='NCHW'):
        fc1 = Conv2D('rcnn_fc1', feature, 1024, 7, nl=BNReLU, padding='VALID') #1*1
        fc2 = Conv2D('rcnn_fc2', fc1, 1024, 1, nl=BNReLU, padding='SAME')
        fc2 = tf.squeeze(fc2, [2, 3])
    # 1*1024*1*1
    classification = FullyConnected(
        'class', fc2, num_classes,
        W_init=tf.random_normal_initializer(stddev=0.01))

    num_classes_for_box = 1 if class_agnostic_regression else num_classes
    box_regression = FullyConnected(
        'box', fc2, num_classes_for_box * 4,
        W_init=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, (-1, num_classes_for_box, 4))
    return classification, box_regression

@layer_register(log_shape=True)
def fastrcnn_head(feature, num_classes, class_agnostic_regression=False):
    """
    Args:
        feature (NxCx7x7):
        num_classes(int): num_category + 1

    Returns:
        cls_logits (Nxnum_class), reg_logits (Nx num_class-1 x 4)
    """
    feature = GlobalAvgPooling('gap', feature, data_format='NCHW')
    classification = FullyConnected(
        'class', feature, num_classes,
        W_init=tf.random_normal_initializer(stddev=0.01))
    num_classes_for_box = 1 if class_agnostic_regression else num_classes
    box_regression = FullyConnected(
        'box', feature, num_classes_for_box * 4,
        W_init=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, (-1, num_classes_for_box, 4))
    return classification, box_regression

def smooth_l1(labels, predictions, delta=1.0):
    error = tf.subtract(predictions, labels)
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    
    linear = tf.subtract(abs_error, quadratic)
    losses = tf.add(
        tf.multiply(
            tf.convert_to_tensor(0.5, dtype=quadratic.dtype),
            tf.multiply(quadratic, quadratic)),
        tf.multiply(delta, linear))
    return losses

@under_name_scope()
def fastrcnn_losses_OHEM(labels, label_logits, boxes, box_logits):
    """
    Args:
        labels: n,
        label_logits: nxC
        boxes: nx4, encoded
        box_logits: nx(C-1)x4
    We need to make sure only correct class have non-zero class regression weight
    """
    print(box_logits.shape)
    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds, out_type=tf.int64)
    empty_fg = tf.equal(num_fg, 0)

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.to_int64(tf.equal(fg_label_pred, 0)), name='num_zero')
        false_negative = tf.where(
            empty_fg, 0., tf.to_float(tf.truediv(num_zero, num_fg)), name='false_negative')
        fg_accuracy = tf.where(
            empty_fg, 0., tf.reduce_mean(tf.gather(correct, fg_inds)), name='fg_accuracy')
    
    per_roi_label_loss = tf.stop_gradient(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits))
    # set bg weight to zero
    bbox_weight = tf.scatter_nd(tf.cast(fg_inds, tf.int32), tf.ones_like(fg_inds), shape=tf.shape(labels))
    #bbox_weight = tf.where(labels > 0, tf.ones_like(labels), tf.zeros_like(labels))
    # set only correct class have regression target
    if int(box_logits.shape[1]) > 1:
        indices = tf.stack(
            [tf.range(tf.shape(labels)[0]),
            tf.to_int32(labels) - 1], axis=1)  # #fgx2
        box_target = tf.gather_nd(box_logits, indices) # (num_fg_roi, 4)
    else:
        box_target = tf.reshape(box_logits, [-1, 4])

    per_roi_box_loss = tf.cast(bbox_weight, tf.float32) * tf.reduce_sum(smooth_l1(boxes, box_target), axis=1)
    #per_roi_box_loss = tf.reduce_sum(per_roi_box_loss, axis=1)
    
    per_roi_total_loss = tf.stop_gradient(per_roi_box_loss + per_roi_label_loss)
    top_k_loss, top_k_indice = tf.nn.top_k(per_roi_total_loss, k=128)
    top_k_indice = tf.stop_gradient(top_k_indice)
    #===========
    
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=label_logits)
    box_loss = tf.cast(bbox_weight, tf.float32) * tf.reduce_sum(smooth_l1(boxes, box_target), axis=1)
    label_loss = tf.gather(label_loss, top_k_indice)
    box_loss = tf.gather(box_loss, top_k_indice)

    label_loss = tf.reduce_mean(label_loss, name='label_loss')
    box_loss = tf.truediv(tf.reduce_sum(box_loss), tf.to_float(tf.shape(top_k_indice)[0]), name='box_loss')

   
    add_moving_summary(label_loss, box_loss, accuracy, fg_accuracy, false_negative, tf.to_float(num_fg, name='num_fg_label'))
    return label_loss, box_loss

@under_name_scope()
def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits):
    """
    Args:
        labels: n,
        label_logits: nxC
        fg_boxes: nfgx4, encoded
        fg_box_logits: nfgx(C-1)x4
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')

    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds, out_type=tf.int64)
    empty_fg = tf.equal(num_fg, 0)
    if int(fg_box_logits.shape[1]) > 1:
        indices = tf.stack(
            [tf.range(tf.shape(fg_labels)[0]),
            tf.to_int32(fg_labels) - 1], axis=1)  # #fgx2
        fg_box_logits = tf.gather_nd(fg_box_logits, indices) # (num_fg_roi, 4)
    else:
        fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])

    #indices = tf.stack(
    #    [tf.range(num_fg),
    #     tf.to_int32(fg_labels) - 1], axis=1)  # #fgx2
    #fg_box_logits = tf.gather_nd(fg_box_logits, indices)

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.to_int64(tf.equal(fg_label_pred, 0)), name='num_zero')
        false_negative = tf.where(
            empty_fg, 0., tf.to_float(tf.truediv(num_zero, num_fg)), name='false_negative')
        fg_accuracy = tf.where(
            empty_fg, 0., tf.reduce_mean(tf.gather(correct, fg_inds)), name='fg_accuracy')

    box_loss = tf.losses.huber_loss(
        fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
    box_loss = tf.truediv(
        box_loss, tf.to_float(tf.shape(labels)[0]), name='box_loss')

    add_moving_summary(label_loss, box_loss, accuracy, fg_accuracy, false_negative, tf.to_float(num_fg, name='num_fg_label'))
    return label_loss, box_loss

@under_name_scope()
def rpn_predictions(boxes, probs):
    print(boxes.shape) #5,1,4
    print(probs.shape) #5 ? ? 7
    boxes = tf.transpose(boxes, [1, 0, 2])  # #catxnx4
    probs = tf.transpose(probs[:, 1:], [1, 0])  # #catxn
    def f(X):
        """
        prob: n probabilities
        box: nx4 boxes

        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob)
        # filter by score threshold
        ids = tf.reshape(tf.where(prob > config.RESULT_SCORE_THRESH), [-1])
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        # NMS within each class
        selection = tf.image.non_max_suppression(
            box, prob, config.RESULTS_PER_IM, config.FASTRCNN_NMS_THRESH)
        selection = tf.to_int32(tf.gather(ids, selection))
        # sort available in TF>1.4.0
        # sorted_selection = tf.contrib.framework.sort(selection, direction='ASCENDING')
        sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
        mask = tf.sparse_to_dense(
            sparse_indices=sorted_selection,
            output_shape=output_shape,
            sparse_values=True,
            default_value=False)
        return mask

    masks = tf.map_fn(f, (probs, boxes), dtype=tf.bool,
                      parallel_iterations=10)     # #cat x N
    selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
    probs = tf.boolean_mask(probs, masks)

    # filter again by sorting scores
    topk_probs, topk_indices = tf.nn.top_k(
        probs,
        tf.minimum(config.RESULTS_PER_IM, tf.size(probs)),
        sorted=False)
    filtered_selection = tf.gather(selected_indices, topk_indices)
    filtered_selection = tf.reverse(filtered_selection, axis=[1], name='filtered_indices')
    return filtered_selection, topk_probs

def np_soft_nms(dets, thresh, max_dets, score_thres=0.5):
    if dets.shape[0] == 0:
        return np.zeros((0, 5))

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    scores = scores[order]

    if max_dets == -1:
        max_dets = order.size

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0

    while order.size > 0 and keep_cnt < max_dets:
        i = order[0]
        dets[i, 4] = scores[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:]
        scores = rescore(ovr, scores[1:], thresh)
        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]
    dets = dets[keep, :]
    # filter
    indices = np.where(dets[:,-1] > score_thres)[0]
    return keep[indices]

def rescore(overlap, scores, thresh, type='gaussian'):
    assert overlap.shape[0] == scores.shape[0]
    if type == 'linear':
        inds = np.where(overlap >= thresh)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(- overlap**2 / thresh)

    return scores

@under_name_scope()
def fastrcnn_predictions_cascade(boxes, probs):
    """
    Generate final results from predictions of all proposals.

    Args:
        boxes: n#catx4 floatbox in float32
        probs: nx#class
    """
    print(boxes.shape) #?,1,4
    print(probs.shape) #? 2
    assert boxes.shape[1] == config.NUM_CLASS
    assert probs.shape[1] == config.NUM_CLASS
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #catxnx4
    boxes.set_shape([None, config.NUM_CLASS - 1, None])
    probs = tf.transpose(probs[:, 1:], [1, 0])  # #catxn
    SOFTNMS_SCORE_THRES = 0.7

    def softnms(X):
        prob, box = X
        output_shape = tf.shape(prob)
        # reshape tp n*5
        dets = tf.concat([box, tf.expand_dims(prob, 1)], axis=1) #n*5
        selection = tf.py_func(np_soft_nms, [dets, 0.3, config.RESULTS_PER_IM, SOFTNMS_SCORE_THRES], tf.int64) 
        selection = tf.to_int32(selection)
        sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
        mask = tf.sparse_to_dense(
            sparse_indices=sorted_selection,
            output_shape=output_shape,
            sparse_values=True,
            default_value=False)
        return mask

    def f(X):
        """
        prob: n probabilities
        box: nx4 boxes

        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob)
        # filter by score threshold
        ids = tf.reshape(tf.where(prob > config.RESULT_SCORE_THRESH), [-1])
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        # NMS within each class
        selection = tf.image.non_max_suppression(
            box, prob, config.RESULTS_PER_IM, config.FASTRCNN_NMS_THRESH)
        selection = tf.to_int32(tf.gather(ids, selection))
        # sort available in TF>1.4.0
        # sorted_selection = tf.contrib.framework.sort(selection, direction='ASCENDING')
        sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
        mask = tf.sparse_to_dense(
            sparse_indices=sorted_selection,
            output_shape=output_shape,
            sparse_values=True,
            default_value=False)
        return mask

    if config.SOFTNMS:
        masks = tf.map_fn(softnms, (probs, boxes), dtype=tf.bool,
                    parallel_iterations=10)     # #cat x N
    else:
        masks = tf.map_fn(f, (probs, boxes), dtype=tf.bool,
                      parallel_iterations=10)     # #cat x N
    selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
    probs = tf.boolean_mask(probs, masks)

    # filter again by sorting scores
    topk_probs, topk_indices = tf.nn.top_k(
        probs,
        tf.minimum(config.RESULTS_PER_IM, tf.size(probs)),
        sorted=False)
    filtered_selection = tf.gather(selected_indices, topk_indices)
    filtered_selection = tf.reverse(filtered_selection, axis=[1], name='filtered_indices')
    return filtered_selection, topk_probs

@under_name_scope()
def fastrcnn_predictions(boxes, probs):
    """
    Generate final results from predictions of all proposals.

    Args:
        boxes: n#catx4 floatbox in float32
        probs: nx#class
    """
    print(boxes.shape) #?,1,4
    print(probs.shape) #? 2
    assert boxes.shape[1] == config.NUM_CLASS - 1
    assert probs.shape[1] == config.NUM_CLASS
    boxes = tf.transpose(boxes, [1, 0, 2])  # #catxnx4
    probs = tf.transpose(probs[:, 1:], [1, 0])  # #catxn
    SOFTNMS_SCORE_THRES = 0.7

    def softnms(X):
        prob, box = X
        output_shape = tf.shape(prob)
        # reshape tp n*5
        dets = tf.concat([box, tf.expand_dims(prob, 1)], axis=1) #n*5
        selection = tf.py_func(np_soft_nms, [dets, 0.3, config.RESULTS_PER_IM, SOFTNMS_SCORE_THRES], tf.int64) 
        selection = tf.to_int32(selection)
        sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
        mask = tf.sparse_to_dense(
            sparse_indices=sorted_selection,
            output_shape=output_shape,
            sparse_values=True,
            default_value=False)
        return mask

    def f(X):
        """
        prob: n probabilities
        box: nx4 boxes

        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob)
        # filter by score threshold
        ids = tf.reshape(tf.where(prob > config.RESULT_SCORE_THRESH), [-1])
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        # NMS within each class
        selection = tf.image.non_max_suppression(
            box, prob, config.RESULTS_PER_IM, config.FASTRCNN_NMS_THRESH)
        selection = tf.to_int32(tf.gather(ids, selection))
        # sort available in TF>1.4.0
        # sorted_selection = tf.contrib.framework.sort(selection, direction='ASCENDING')
        sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
        mask = tf.sparse_to_dense(
            sparse_indices=sorted_selection,
            output_shape=output_shape,
            sparse_values=True,
            default_value=False)
        return mask

    if config.SOFTNMS:
        masks = tf.map_fn(softnms, (probs, boxes), dtype=tf.bool,
                    parallel_iterations=10)     # #cat x N
    else:
        masks = tf.map_fn(f, (probs, boxes), dtype=tf.bool,
                      parallel_iterations=10)     # #cat x N
    selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
    probs = tf.boolean_mask(probs, masks)

    # filter again by sorting scores
    topk_probs, topk_indices = tf.nn.top_k(
        probs,
        tf.minimum(config.RESULTS_PER_IM, tf.size(probs)),
        sorted=False)
    filtered_selection = tf.gather(selected_indices, topk_indices)
    filtered_selection = tf.reverse(filtered_selection, axis=[1], name='filtered_indices')
    return filtered_selection, topk_probs

@layer_register(log_shape=True)
def maskrcnn_head_FPN(feature, num_class):
    """
    Args:
        feature (NxCx7x7):
        num_classes(int): num_category + 1

    Returns:
        mask_logits (N x num_category x 28 x 28):
    """
    with argscope([Conv2D, Deconv2D], data_format='NCHW',
                  W_init=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_in', distribution='normal')):
        l = Deconv2D('deconv1', feature, 256, 2, stride=2, nl=tf.nn.relu)
        l = Conv2D('conv0', l, 256, 3, nl=tf.nn.relu)
        l = Conv2D('conv1', l, 256, 3, nl=tf.nn.relu)
        l = Conv2D('conv2', l, 256, 3, nl=tf.nn.relu)
        l = Conv2D('conv3', l, 256, 3, nl=tf.nn.relu)
        l = Deconv2D('deconv2', l, 256, 2, stride=2, nl=tf.nn.relu)
        l = Conv2D('conv4', l, num_class - 1, 1)
    return l

@layer_register(log_shape=True)
def maskrcnn_head(feature, num_class):
    """
    Args:
        feature (NxCx7x7):
        num_classes(int): num_category + 1

    Returns:
        mask_logits (N x num_category x 14 x 14):
    """
    with argscope([Conv2D, Deconv2D], data_format='NCHW',
                  W_init=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_in', distribution='normal')):
        l = Deconv2D('deconv', feature, 256, 2, stride=2, nl=tf.nn.relu)
        l = Conv2D('conv', l, num_class - 1, 1)
    return l

def dice_loss(y_true, y_pred):
     epsilon_denominator = 0.001
     y_true_f = tf.reshape(y_true, [-1])
     y_pred_f = tf.reshape(y_pred, [-1])
     intersection = tf.reduce_sum(y_true_f * y_pred_f)
     score = (2. * intersection + epsilon_denominator) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + epsilon_denominator)
     return 1 - tf.reduce_mean(score)

@under_name_scope()
def maskrcnn_loss(mask_logits, fg_labels, fg_target_masks):
    """
    Args:
        mask_logits: #fg x #category x14x14
        fg_labels: #fg, in 1~#class
        fg_target_masks: #fgx14x14, int
    """
    num_fg = tf.size(fg_labels)
    indices = tf.stack([tf.range(num_fg), tf.to_int32(fg_labels) - 1], axis=1)  # #fgx2
    mask_logits = tf.gather_nd(mask_logits, indices)  # #fgx14x14
    mask_probs = tf.sigmoid(mask_logits)

    # add some training visualizations to tensorboard
    with tf.name_scope('mask_viz'):
        viz = tf.concat([fg_target_masks, mask_probs], axis=1)
        viz = tf.expand_dims(viz, 3)
        viz = tf.cast(viz * 255, tf.uint8, name='viz')
        tf.summary.image('mask_truth|pred', viz, max_outputs=10)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=fg_target_masks, logits=mask_logits)
    loss = tf.reduce_mean(loss, name='maskrcnn_loss')
    loss_d = dice_loss(fg_target_masks, tf.nn.sigmoid(mask_logits))

    pred_label = mask_probs > 0.5
    truth_label = fg_target_masks > 0.5
    accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(pred_label, truth_label)),
        name='accuracy')
    pos_accuracy = tf.logical_and(
        tf.equal(pred_label, truth_label),
        tf.equal(truth_label, True))
    pos_accuracy = tf.reduce_mean(tf.to_float(pos_accuracy), name='pos_accuracy')
    fg_pixel_ratio = tf.reduce_mean(tf.to_float(truth_label), name='fg_pixel_ratio')

    add_moving_summary(loss, loss_d, accuracy, fg_pixel_ratio, pos_accuracy)
    return loss + loss_d

@under_name_scope()
def extract_position_matrix(sliced_rois, nongt_dim=300):
    """
    boxes: Nx4 floatbox
    """
    xmin, ymin, xmax, ymax = tf.split(sliced_rois, 4, axis=1)
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    delta_x = center_x - tf.transpose(center_x)
    delta_x = tf.div(delta_x, bbox_width)
    delta_x = tf.log(tf.maximum(tf.abs(delta_x), 1e-3))

    delta_y = center_y - tf.transpose(center_y)
    delta_y = tf.div(delta_y, bbox_height)
    delta_y = tf.log(tf.maximum(tf.abs(delta_y), 1e-3))

    delta_width = tf.div(bbox_width, tf.transpose(bbox_width))
    delta_width = tf.log(delta_width)
    delta_height = tf.div(bbox_height, tf.transpose(bbox_height))
    delta_height = tf.log(delta_height)
    
    concat_list = [delta_x, delta_y, delta_width, delta_height] # 4*num_rois*num_rois
    for idx, sym in enumerate(concat_list):
            sym = tf.slice(sym, [0,0], [-1, nongt_dim]) # num_rois*nongtdim
            concat_list[idx] = tf.expand_dims(sym, axis=2)
    position_matrix = tf.concat(concat_list, axis=2) # num_rois*nongtdim*4
    return position_matrix

@under_name_scope()
def extract_label_embedding(classificaiton_feature, feat_dim=64):
    cf = tf.argmax(classificaiton_feature, axis=-1) # extract predicted feature
    cf = tf.stop_gradient(tf.one_hot(cf, depth=config.NUM_CLASS))
    return cf

@under_name_scope()
def extract_position_embedding(position_matrix, feat_dim=64, wave_length=1000):
    feat_range = tf.range(0, feat_dim / 8)
    dim_mat = tf.pow(tf.ones(8)*wave_length, (8. / feat_dim) * feat_range)
    dim_mat = tf.reshape(dim_mat, [1, 1, 1, -1])
    position_matrix = tf.expand_dims(100.0 * position_matrix, axis=3)
    div_mat = tf.div(position_matrix, dim_mat)
    sin_mat = tf.sin(div_mat)
    cos_mat = tf.cos(div_mat)
    embedding = tf.concat([sin_mat, cos_mat], axis=3)
    # embedding, [num_rois, nongt_dim, feat_dim]
    sha = embedding.get_shape().as_list()
    print(embedding.shape)
    embedding = tf.reshape(embedding, [-1, sha[1], feat_dim])
    print(embedding.shape)
    return embedding

@layer_register(log_shape=True)
def attention_module_multi_head(roi_feat, position_embedding,
                                    nongt_dim=300, fc_dim=16, feat_dim=1024,
                                    dim=(1024, 1024, 1024),
                                    group=16, index=1):
    print(roi_feat.shape)
    roi_feat = tf.reshape(roi_feat, [-1, 1024])
    dim_group = (dim[0] // group, dim[1] // group, dim[2] // group)
    nongt_roi_feat = tf.slice(roi_feat, [0, 0], [nongt_dim, -1])
    #sha = position_embedding.get_shape().as_list()
    #position_embedding_reshape = tf.reshape(position_embedding, [-1, sha[-1]])
    position_embedding_reshape = tf.transpose(position_embedding, (2, 0, 1))
        # [1, 64, num_rois, nongt_dim]
    position_embedding_reshape = tf.expand_dims(position_embedding_reshape, 0)
    with argscope([Conv2D], data_format='NCHW',
                  W_init=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_in', distribution='normal')):
        position_feat_1 = Conv2D('pair_pos_fc1_' + str(index), position_embedding_reshape, fc_dim, 1, nl=tf.identity)
    print(position_feat_1.shape)
    #position_feat_1 = FullyConnected(
    #    'pair_pos_fc1_' + str(index), position_embedding_reshape, fc_dim,
    #    W_init=tf.random_normal_initializer(stddev=0.01))
    position_feat_1_relu = tf.nn.relu(position_feat_1) # Max operation
    # aff_weight, [num_rois, nongt_dim, fc_dim]
    #aff_weight = tf.reshape(position_feat_1_relu, [-1, nongt_dim, fc_dim])
    # aff_weight, [num_rois, fc_dim, nongt_dim]
    #aff_weight = tf.transpose(aff_weight, (0, 2, 1))
    aff_weight = tf.transpose(position_feat_1_relu, (2, 1, 3, 0))
    # aff_weight, [num_rois, fc_dim, nongt_dim]
    aff_weight = tf.squeeze(aff_weight, axis=-1)

    q_data = FullyConnected('query_' + str(index),
                                       roi_feat,
                                       dim[0], W_init=tf.random_normal_initializer(stddev=0.01))
    q_data_batch = tf.reshape(q_data, (-1, group, dim_group[0]))
    q_data_batch = tf.transpose(q_data_batch, (1, 0, 2))

    k_data = FullyConnected('key_' + str(index),
                                       nongt_roi_feat,
                                       dim[1], W_init=tf.random_normal_initializer(stddev=0.01))
    k_data_batch = tf.reshape(k_data, (-1, group, dim_group[1]))
    k_data_batch = tf.transpose(k_data_batch, (1, 0, 2))

    v_data = nongt_roi_feat
    
    aff = tf.keras.backend.batch_dot(q_data_batch, tf.transpose(k_data_batch, [0,2,1]))
    # aff_scale, [group, num_rois, nongt_dim]
    aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
    aff_scale = tf.transpose(aff_scale, (1, 0, 2))
    
    assert fc_dim == group, 'fc_dim != group'
    # weighted_aff, [num_rois, fc_dim, nongt_dim]
    
    weighted_aff = tf.log(tf.maximum(aff_weight, 1e-6)) + aff_scale
    aff_softmax = tf.nn.softmax(weighted_aff, dim=2, name='softmax_' + str(index))
    # [num_rois * fc_dim, nongt_dim]
    sha = aff_softmax.get_shape().as_list()
    aff_softmax_reshape = tf.reshape(aff_softmax, (-1, sha[2]))
    # output_t, [num_rois * fc_dim, feat_dim]
    output_t = tf.tensordot(aff_softmax_reshape, v_data, axes=[[1],[0]])
    # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
    output_t = tf.reshape(output_t, (-1, fc_dim * feat_dim, 1, 1))
    # linear_out, [num_rois, dim[2], 1, 1]
    with argscope([Conv2D], data_format='NCHW',
                  W_init=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_in', distribution='normal')):
        linear_out = Conv2D('linear_out_' + str(index), output_t, dim[2], 1, data_format='NCHW', split=fc_dim, nl=tf.identity)
    # [num_rois, dim[2]]]
    if config.FPN == False:
        linear_out = tf.squeeze(linear_out, axis=[-2, -1])
    #print(linear_out.shape)
    return linear_out

@layer_register(log_shape=True)
def fastrcnn_2fc_head(feature):
    """
    Args:
        feature (any shape):
    Returns:
        2D head feature
    """
    dim = 1024
    init = tf.variance_scaling_initializer()
    hidden = FullyConnected('fc6', feature, dim, kernel_initializer=init, activation=tf.nn.relu)
    hidden = FullyConnected('fc7', hidden, dim, kernel_initializer=init, activation=tf.nn.relu)
    return hidden

@layer_register(log_shape=True)
def fastrcnn_outputs(feature, num_classes, class_agnostic_regression=False):
    classification = FullyConnected(
        'class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    num_classes_for_box = 1 if class_agnostic_regression else num_classes
    box_regression = FullyConnected(
        'box', feature, num_classes_for_box * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, (-1, num_classes_for_box, 4), name='output_box')
    return classification, box_regression

@layer_register(log_shape=True)
def maskrcnn_upXconv_head(feature, num_category, num_convs, norm=None):
    """
    Args:
        feature (NxCx s x s): size is 7 in C4 models and 14 in FPN models.
        num_category(int):
        num_convs (int): number of convolution layers
        norm (str or None): either None or 'GN'
    Returns:
        mask_logits (N x num_category x 2s x 2s):
    """
    l = feature
    norm = config.NORM == 'GN'
    with argscope([Conv2D, Conv2DTranspose], data_format='channels_first',
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out', distribution='normal')):
        # c2's MSRAFill is fan_out
        for k in range(num_convs):
            l = Conv2D('fcn{}'.format(k), l, 256, 3, activation=tf.nn.relu)
            if norm:
                l = GroupNorm('gn{}'.format(k), l)
        l = Conv2DTranspose('deconv', l, 256, 2, strides=2, activation=tf.nn.relu)
        for k in range(2):
            l = Conv2D('fcn_2_{}'.format(k), l, 256, 3, activation=tf.nn.relu)
            if norm:
                l = GroupNorm('gn_2_{}'.format(k), l)
        l = Conv2DTranspose('deconv_2', l, 256, 2, strides=2, activation=tf.nn.relu)
        l = Conv2D('conv', l, num_category, 1)
    return l

@layer_register(log_shape=True)
def fastrcnn_Xconv1fc_head(feature, num_convs, norm=None):
    """
    Args:
        feature (NCHW):
        num_classes(int): num_category + 1
        num_convs (int): number of conv layers
        norm (str or None): either None or 'GN'
    Returns:
        2D head feature
    """
    from basemodel_tp import GroupNorm
    use_gn = config.NORM == 'GN'
    l = feature
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out', distribution='normal')):
        for k in range(num_convs):
            l = Conv2D('conv{}'.format(k), l, 256, 3, activation=tf.nn.relu)
            if use_gn:
                l = GroupNorm('gn{}'.format(k), l)
        l = FullyConnected('fc', l, 1024,
                           kernel_initializer=tf.variance_scaling_initializer(), activation=tf.nn.relu)
    return l

@under_name_scope()
def roi_align_PAN(featuremaps, boxes, output_shape):
    # [N6,N5,N4,N3,N2]
    # featuremaps[idx:0 ~ idx:4]
    boxes = tf.stop_gradient(boxes)  # TODO
    boxes_per_layer = []
    strides = [2., 4., 8., 16., 32., 64.]
    for level in range(1, 6):
        featuremap_to_crop = featuremaps[5 - (level - 1)]
        level_boxes = boxes * (1.0 / strides[level - 1])
        level_boxes = tf.stop_gradient(level_boxes)
        # sample 4 locations per roi bin
        ret = crop_and_resize(
            featuremap_to_crop, level_boxes,
            tf.zeros([tf.shape(level_boxes)[0]], dtype=tf.int32),
            output_shape * 2)
        ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
        #ret = tf.Print(ret, [tf.shape(ret)], message="ret")
        boxes_per_layer.append(ret)
    return boxes_per_layer

@under_name_scope()
@auto_reuse_variable_scope
def fastrcnn_fc_head_fusion(feature):
    dim = 1024
    init = tf.variance_scaling_initializer()
    fc1 = FullyConnected('fc6', feature, dim, kernel_initializer=init, activation=tf.nn.relu)
    return fc1

@layer_register(log_shape=True)
def fastrcnn_fc_head_PAN(level_feature):
    dim = 1024
    init = tf.variance_scaling_initializer()
    level_feature = [fastrcnn_fc_head_fusion(roi) for roi in level_feature]
    level_feature = tf.add_n(level_feature)   
    hidden = FullyConnected('fc7', level_feature, dim, kernel_initializer=init, activation=tf.nn.relu)
    return hidden