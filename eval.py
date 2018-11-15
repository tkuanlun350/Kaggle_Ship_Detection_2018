#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple
import numpy as np
import cv2
from subprocess import call
import zipfile
import nibabel
from tensorpack.utils.utils import get_tqdm_kwargs
from viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs, shrink_poly, draw_refined_outputs)
#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
#import pycocotools.mask as cocomask

from contextlib import ExitStack
from common import CustomResize
import config
import pandas as pd

import collections
import nibabel as nib

DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])

def binary_dice3d(s,g):
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum() # TP
    s1 = s.sum() # TP + FP 
    s2 = g.sum() # TP + FN
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    return dice

def binary_recall(s, g):
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum() # TP
    s1 = s.sum() # TP + FP 
    s2 = g.sum() # TP + FN
    recall = (s0 + 1e-10)/(s2 + 1e-10)
    return recall

def load_nifty_volume_as_array(filename, with_header = False):
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    if(with_header):
        return data, img.affine, img.header
    else:
        return data

def eval_dice(gts, results):
    d = {}
    for type_idx in range(1, config.NUM_CLASS):
        dice, recall = dice_of_oar_dataset(gts, results, type_idx)
        dice = np.asarray(dice)
        recall = np.asarray(recall)
        dice_mean = dice.mean(axis = 0)
        recall_mean = recall.mean(axis=0)
        dice_std  = dice.std(axis = 0)
        test_type = type_idx
        print('organ type', config.CLASS_NAMES[test_type])
        print('dice mean', dice_mean)
        print('recall mean', recall_mean)
        d[config.CLASS_NAMES[test_type]+"_dice"] = dice_mean[0]
        d[config.CLASS_NAMES[test_type]+"_recall"] = recall_mean[0]
    return d

def dice_of_oar_dataset(gt, pred, type_idx):
    dice_all_data = []
    recall_all_data = []
    for i in range(len(gt)):
        g_volume = gt[i]
        s_volume = pred[i]
        dice_one_volume = []
        temp_dice = binary_dice3d(s_volume == type_idx, g_volume == type_idx)
        temp_recall = binary_recall(s_volume == type_idx, g_volume == type_idx)
        dice_one_volume = [temp_dice]
        dice_all_data.append(dice_one_volume)
        recall_all_data.append([temp_recall])
    return dice_all_data, recall_all_data

def fill_full_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    """
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
    x1 = max(x0, x1)    # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    # Use box as mask in ship detection as the approach of annotation.
    #mask = np.ones((h, w)).astype('uint8')
    
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret

def fill_full_mask_TTA(box, mask, shape):
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
    x1 = max(x0, x1)    # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    
    mask = cv2.resize(mask, (w, h))  
    ret = np.zeros(shape)
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret

def do_flip_transpose(image, type=0):
    #choose one of the 8 cases

    if type==1: #rotate90
        image = image.transpose(1,0,2)
        image = cv2.flip(image,1)
        
    if type==2: #rotate180
        image = cv2.flip(image,-1)

    if type==3: #rotate270
        image = image.transpose(1,0,2)
        image = cv2.flip(image,0)

    if type==4: #flip left-right
        image = cv2.flip(image,1)

    if type==5: #flip up-down
        image = cv2.flip(image,0)
        
    if type==6:
        image = cv2.flip(image,1)
        image = image.transpose(1,0,2)
        image = cv2.flip(image,1)
        
    if type==7:
        image = cv2.flip(image,0)
        image = image.transpose(1,0,2)
        image = cv2.flip(image,1)

    return image

def undo_flip_transpose(image, type=0):
    #choose one of the 8 cases

    if type==1: #rotate90
        image = cv2.flip(image,1)
        image = image.transpose(1,0)
        
    if type==2: #rotate180
        image = cv2.flip(image,-1)

    if type==3: #rotate270
        image = image.transpose(1,0)
        image = cv2.flip(image,1)

    if type==4: #flip left-right
        image = cv2.flip(image,1)

    if type==5: #flip up-down
        image = cv2.flip(image,0)
        
    if type==6:
        image = cv2.flip(image,1)
        image = image.transpose(1,0)
        image = cv2.flip(image,1)
        
    if type==7:
        image = cv2.flip(image,0)
        image = image.transpose(1,0)
        image = cv2.flip(image,1)

    return image

def bbox_vflip(bbox, shape):
    """Flip a bounding box vertically around the x-axis."""
    x_min, y_min, x_max, y_max = bbox
    return [x_min, shape[0] - y_max, x_max, shape[0] - y_min]


def bbox_hflip(bbox, shape):
    """Flip a bounding box horizontally around the y-axis."""
    x_min, y_min, x_max, y_max = bbox
    return [shape[1] - x_max, y_min, shape[1] - x_min, y_max]

def detect_one_image_TTA2(img, model_func):
    orig_shape = img.shape[:2]
    SCALES = [1800, 2000]
    all_scale_results = []
    augs = [0, 4]
    mask_whole = np.zeros((img.shape[0], img.shape[1]))
    for s in SCALES:
        mask_whole_d = np.zeros((img.shape[0], img.shape[1]))
        for d in augs:
            img = do_flip_transpose(img, d)
            resizer = CustomResize(s, config.MAX_SIZE)
            resized_img = resizer.augment(img.copy())
            scale = (resized_img.shape[0] * 1.0 / img.shape[0] + resized_img.shape[1] * 1.0 / img.shape[1]) / 2
            boxes, probs, labels, *masks = model_func(resized_img)
            boxes = boxes / scale

            if masks:
                # has mask
                full_masks = [fill_full_mask_TTA(box, mask, orig_shape)
                            for box, mask in zip(boxes, masks[0])]
                masks = full_masks
            else:
                # fill with none
                masks = [None] * len(boxes)

            results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
            for re in results:
                mask_whole_d += undo_flip_transpose(re.mask, d)
        mask_whole_d = mask_whole_d / float(len(augs))
        mask_whole += mask_whole_d
    mask_whole = mask_whole / float(len(SCALES))
    mask_whole = mask_whole > 0.5
    return mask_whole.astype('uint8')

def np_soft_nms(dets, masks, thresh, max_dets, score_thres=0.5):
    
    def mask_overlap(target, masks):
        ious = []
        for m in masks:
            iou = IoU(target, m)
            ious.append(iou)
        return np.array(ious)

    def rescore(overlap, scores, thresh, type='gaussian'):
        assert overlap.shape[0] == scores.shape[0]
        if type == 'linear':
            inds = np.where(overlap >= thresh)[0]
            scores[inds] = scores[inds] * (1 - overlap[inds])
        else:
            scores = scores * np.exp(- overlap**2 / thresh)

        return scores

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
        ovr = mask_overlap(masks[i], masks[order[1:]])
        order = order[1:]
        scores = rescore(ovr, scores[1:], thresh, type='gaussian')
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

def detect_one_image_TTA(img, model_func):
    orig_shape = img.shape[:2]
    SCALES = [1800, 2000]
    all_scale_results = []
    augs = [0]
    mask_whole = np.zeros((img.shape[0], img.shape[1]))
    ndets = []
    nmasks = []
    nscores = []
    for s in SCALES:
        for d in augs:
            img_aug = do_flip_transpose(img.copy(), d)
            resizer = CustomResize(s, config.MAX_SIZE)
            resized_img = resizer.augment(img_aug.copy())
            scale = (resized_img.shape[0] * 1.0 / img.shape[0] + resized_img.shape[1] * 1.0 / img.shape[1]) / 2
            boxes, probs, labels, *masks = model_func(resized_img)
            boxes = boxes / scale
            masks = masks[0] 
            if d == 4:
                # flip-left right
                boxes = [bbox_hflip(bb, img_aug.shape) for bb in boxes]
                masks = [undo_flip_transpose(m, d) for m in masks] 
            elif d == 5:
                 # flip-up down
                boxes = [bbox_vflip(bb, img_aug.shape) for bb in boxes]
                masks = [undo_flip_transpose(m, d) for m in masks] 
            if len(boxes) > 0:
                #ndets.append(boxes)
                #nmasks.append(masks)
                #nscores.append(probs)
                ndets.extend(boxes)
                nmasks.extend(masks)
                nscores.extend(probs)
            #mask_after_merge = merge_instance_prob(mask_instances, score_instances)
            #for mm in mask_after_merge:
            #    mask_whole += mm
    
    iou_thres = 0.8
    if (len(ndets) == 0):
        return [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    """
    if (len(ndets) == 1):
        full_masks = [fill_full_mask(box, mask, orig_shape)
                        for box, mask in zip(ndets[0], nmasks[0])]
        return [DetectionResult(*args) for args in zip(ndets[0], nscores[0], np.ones_like(nscores[0]).astype('uint8'), full_masks)]
    #nmasks = np.array(nmasks)
    #ndets = np.array(ndets)
    #nscores = np.array(nscores)
    from utils.np_box_ops import iou as np_iou
    # find best iou box in other detector
    outboxes = []
    outmasks = []
    outscores = []
    used = []
    for i in range(len(ndets)):
        current_dets = ndets[i]
        current_masks = nmasks[i]
        current_scores = nscores[i]
        target_dets = [d for idx, d in enumerate(ndets) if idx != i]
        target_masks = [d for idx, d in enumerate(nmasks) if idx != i]
        target_scores = [d for idx, d in enumerate(nscores) if idx != i]
        found = [[] for _ in range(len(current_dets))]
        for j in range(len(target_dets)):
            box_ious = np_iou(np.array(current_dets), np.array(target_dets[j]))
            ious_argmax_per_current = box_ious.argmax(axis=1) # ncurrent
            ious_max_per_current = box_ious.max(axis=1) #ncurrent
            # loop every current box to find best box with iou > thres in target
            for index, (_id, score) in enumerate(zip(ious_argmax_per_current, ious_max_per_current)):
                if score >= iou_thres:
                    found[index].append([j, _id])
        # now current contain every other model's best fit
        for ii in range(len(found)):
            _boxes = []
            _masks = []
            _scores = []
            if len(found[ii]) == 0:
                _boxes.append(current_dets[ii])
                _masks.append(current_masks[ii])
                _scores.append(current_scores[ii])
            else:
                for id1, id2 in found[ii]:
                    _boxes.append(target_dets[id1][id2])
                    _masks.append(target_masks[id1][id2])
                    _scores.append(target_scores[id1][id2])
            average_box = np.mean(np.array(_boxes), axis=0)
            average_mask = np.mean(np.array(_masks), axis=0)
            average_score = np.mean(np.array(_scores), axis=0)
        
            outboxes.append(average_box)
            outmasks.append(average_mask)
            outscores.append(average_score)
    """
    outboxes = ndets
    outmasks = nmasks
    outscores = nscores

    outboxes = np.array(outboxes)
    outscores = np.array(outscores)
    outmasks =  np.array(outmasks)
    outmasks = [fill_full_mask(box, mask, orig_shape)
                        for box, mask in zip(outboxes, outmasks)]
    outmasks = np.array(outmasks)
    keep = np_soft_nms(np.hstack((outboxes, outscores[...,np.newaxis])), outmasks, 0.5, 100, 0.5)
    outboxes = outboxes[keep]
    outscores = outscores[keep]
    outmasks = outmasks[keep]
    #full_masks = [fill_full_mask(box, mask, orig_shape)
    #                    for box, mask in zip(outboxes, outmasks)]
    results = [DetectionResult(*args) for args in zip(outboxes, outscores, np.ones_like(outscores).astype('uint8'), outmasks)]
    return results

def merge_instance_prob(predicts, scores):
    shape = np.array(predicts).shape # n * h * w
    if (shape[0] == 0):
        return [0]
    predicts = np.array(predicts)
    sort_ind = np.argsort(scores)[::-1]
    predicts = predicts[sort_ind]
    overlap = np.zeros((shape[1], shape[2]))
    # let the highest score to occupy pixel
    # from high score -> low score, clear overlap
    for mm in range(len(predicts)):
        mask = predicts[mm].copy()
        mask[mask>0] = 1 # set box of mask to 1
        overlap += mask # overlap > 1 indicates dup
        predicts[mm][overlap>1] = 0    
    return predicts

def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(config.SHORT_EDGE_SIZE, config.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = (resized_img.shape[0] * 1.0 / img.shape[0] + resized_img.shape[1] * 1.0 / img.shape[1]) / 2
    boxes, probs, labels, *masks = model_func(resized_img)
    boxes = boxes / scale
    if masks:
        # has mask
        full_masks = [fill_full_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results

def mask_to_box(mask):
    # special for ship detection
    _, cnt, _ = cv2.findContours(mask, 1, 2)
    rect = cv2.minAreaRect(cnt[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(mask, [box], 0, 1, -1)
    return mask

def clean_overlap_instance(predicts, scores, img_id):
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
    del_ind = np.where(np.sum(predicts, axis=(1,2)) < 7)[0]
    if len(del_ind)>0:
        if len(del_ind)<len(predicts):
            print('Empty mask, deleting', len(del_ind), 'masks', img_id)
            predicts = np.delete(predicts, del_ind, axis=0)
        else:
            predicts = np.zeros([1, shape[1], shape[2]])
            predicts[0,0,0] = 1
    return predicts

def eval_on_dataflow(df, detect_func, tqdm_bar=None):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.
    Return:
        list of dict, to be dumped to COCO json format
    """
    csv_path = os.path.join(config.BASEDIR, 'train_ship_segmentations_v2.csv')
    seg_df = pd.read_csv(csv_path, engine="python")
    seg_df = seg_df.dropna(axis=0)
    seg_df = seg_df.set_index('ImageId')

    df.reset_state()
    all_results = []
    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323
    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        score = 0.0
        all_score = []
        count = 0.0
        eval_names = []
        eval_imgs = []
        all_det = []
        all_im = []
        for img, img_id in df.get_data():
            results = detect_func(img)
            #mask_whole = detect_func(img)
            #all_det.append(mask_whole)
            all_im.append(img)
            eval_names.append(img_id)
            final = draw_final_outputs(img, results)
            cv2.imwrite('./eval_out_bb/{}'.format(img_id), final)
            mask_instances = [r.mask for r in results]
            score_instances = [r.score for r in results]
            
            masks = clean_overlap_instance(mask_instances, score_instances, img_id)
            if len(masks) == 0:
                print("no mask!!", img_id)
                v = 0
            else:
                v = local_eval(masks, img_id, seg_df) #pred, imgId
            score += v
            all_score.append(v)
            count += 1
            tqdm_bar.update(1)
        for k in np.array(all_score).argsort()[:20]:
            print(all_score[k], eval_names[k])
        #    cv2.imwrite("./eval_out/{}".format(eval_names[k]), all_im[k])
        #    cv2.imwrite("./eval_out/{}_mask.jpg".format(eval_names[k].split(".")[0]), all_det[k]*255)
        print("Local Eval: ", score/count)
    return all_results, score/count


# https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_evaluation_scores(json_file):
    ret = {}
    annofile = os.path.join(config.BASEDIR, 'Airbus_val.json')
    coco = COCO(annofile)
    cocoDt = coco.loadRes(json_file)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
    # recall => cocoEval.stats[8]
    for k in range(6):
        ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]
    ap = cocoEval.stats[0]
    recall = cocoEval.stats[8]
    ret['mF2(bbox)'] = (1+4) * (ap*recall)/(4*ap+recall)

    cocoEval = COCOeval(coco, cocoDt, 'segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    for k in range(6):
        ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
    ap = cocoEval.stats[0]
    recall = cocoEval.stats[8]
    ret['mF2(segm)'] = (1+4) * (ap*recall)/(4*ap+recall)
    return ret

def IoU(pred, targs):
    pred = (pred > 0.5).astype(float)
    intersection = (pred*targs).sum()
    return intersection / ((pred+targs).sum() - intersection + 1.0)

def split_mask(mask):
    from scipy import ndimage
    threshold = 0.0
    threshold_obj = 0 #ignor predictions composed of "threshold_obj" pixels or less
    labled,n_objs = ndimage.label(mask > threshold)
    result = []
    for i in range(n_objs):
        obj = (labled == i + 1).astype(int)
        if(obj.sum() > threshold_obj): result.append(obj)
    return result

def get_mask_ind(img_id, df, shape = (768,768)): #return mask for each ship
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return []
    if(type(masks) == str): masks = [masks]
    result = []
    for mask in masks:
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
        result.append(img.reshape(shape).T)
    return result

def get_score(pred, true):
    n_th = 10
    b = 4 # beta^2
    thresholds = [0.5 + 0.05*i for i in range(n_th)]
    n_masks = len(true)
    n_pred = len(pred)
    ious = []
    score = 0
    for mask in true:
        buf = []
        for p in pred: buf.append(IoU(p,mask))
        ious.append(buf)
    for t in thresholds:   
        tp, fp, fn = 0, 0, 0
        for i in range(n_masks):
            match = False
            for j in range(n_pred):
                if ious[i][j] > t: match = True
            if not match: fn += 1
        
        for j in range(n_pred):
            match = False
            for i in range(n_masks):
                if ious[i][j] > t: match = True
            if match: tp += 1
            else: fp += 1
        score += ((b+1)*tp)/((b+1)*tp + b*fn + fp) 
    return score/n_th

def local_eval(pred_mask, imageId, df):
    true = get_mask_ind(imageId, df)
    return get_score(pred_mask, true)

if __name__ == "__main__":
    gt_filename = "/data/dataset/oar/data/label/428146.nii.gz"
    gts = [load_nifty_volume_as_array(gt_filename)]
    r = [load_nifty_volume_as_array(gt_filename)]
    eval_dice(gts, r)
