import numpy as np
import cv2
import math
import random
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time
import matplotlib.pyplot as plt

import skimage
import skimage.color
import skimage.morphology
from scipy import ndimage
### for processing mask
import skimage.morphology as morph
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from scipy.stats import itemfreq
import scipy.misc

from tensorpack.dataflow import (
    imgaug, dataset, AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)

WIDTH = 256
HEIGHT = 256

def get_resnet_augmentor():
    augmentors = [
        imgaug.RandomOrderAug(
            [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                imgaug.Contrast((0.6, 1.4), clip=False),
                imgaug.Saturation(0.4, rgb=False),
                # rgb-bgr conversion for the constants copied from fb.resnet.torch
                imgaug.Lighting(0.1,
                                eigval=np.asarray(
                                    [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                eigvec=np.array(
                                    [[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]],
                                    dtype='float32')[::-1, ::-1]
                                )])
    ]
    return augmentors

def augment_brightness_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_shift_scale_rotate_transform2( image, mask,
                        shift_limit=[-0.0625,0.0625], scale_limit=[1/1.2,1.2],
                        rotate_limit=[-10,10], borderMode=cv2.BORDER_REFLECT_101 , u=0.5):

    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height, width, channel = image.shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(scale_limit[0],scale_limit[1])
        sx    = scale
        sy    = scale
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)

        image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

        mask = mask.astype(np.float32)
        mask = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                    borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        mask = mask.astype(np.int32)

    return image, mask

def color_aug(image, mask):
    if 1:
        type = random.randint(0,4)
        if type==0:
            image = random_transform(image, u=0.5, func=do_color_shift, alpha0=[-0.2,0.2], alpha1=[-0.2,0.2], alpha2=[-0.2,0.2])

        elif type==1:
            image = random_transform(image, u=0.5, func=do_saturation_shift, alpha=[0,0.3])

        elif type==2:
            image = random_transform(image, u=0.5, func=do_decolor)
        else:
            pass
        #print('illumintaion',image.dtype)
    
    # illumintaion ------------
    if 0:
        type = random.randint(0,1)
        if type==0:
            image = random_transform(image, u=0.5, func=do_custom_process1, gamma=[0.8,2.0],alpha=[0.7,0.9],beta=[1.0,1.0])
        else:
            pass
        #print('illumintaion',image.dtype)

    # geometric ------------
    
    if 1: 
        type = random.randint(0,1)
        image, mask = random_shift_scale_rotate_transform2(image, mask)
        if type == 0:
            grid = random.randint(8, 64)
            distort = random.random()*0.5
            image, mask = do_elastic_transform2(image, mask, grid=grid, distort=distort)
        #print('geometric',image.dtype)
    return image, mask

def multi_mask_to_annotation(multi_mask):
    H,W      = multi_mask.shape[:2]
    box      = []
    label    = []
    instance = []
    is_crowd = []
    num_masks = multi_mask.max()
    for i in range(num_masks):
        mask = (multi_mask==(i+1))
        if mask.sum()>1:

            y,x = np.where(mask)
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            w = (x1-x0)+1
            h = (y1-y0)+1


            border = max(2, round(0.2*(w+h)/2))
            #border = max(1, round(0.1*min(w,h)))
            #border = 0
            x0 = x0-border
            x1 = x1+border
            y0 = y0-border
            y1 = y1+border

            #clip
            x0 = max(0,x0)
            y0 = max(0,y0)
            x1 = min(W-1,x1)
            y1 = min(H-1,y1)
            
            #if (is_invalid_box([x0,y0,x1,y1])):
            #    continue

            box.append([x0,y0,x1,y1])
            label.append(1)
            instance.append(mask)
            is_crowd.append(0) # ignored cases

    box      = np.array(box, np.float32)
    label    = np.array(label, np.int32)
    instance = np.array(instance, np.float32)
    is_crowd = np.array(is_crowd, np.int32)

    return box, label, instance, is_crowd

## for debug ---

def dummy_transform(image, limit1, limit2, limit3):
    print ('\tdummy_transform')
    print ('\tlimit1=%0.1f, limit2=%0.1f, limit3=%0.1f'%(limit1, limit2, limit3))
    return image

def random_dummy_transform(image, u, func, **kwargs):
    if random.random() < u:

        limits = []
        for k in kwargs:
            #print (k)
            #print (kwargs[k])
            #print ('')

            limit = kwargs[k]
            l = random.uniform(limit[0],limit[1])
            limits.append(l)

        #print(limits)
        image = func(image, *limits)

    return image




## customize augmentation ---

def do_custom_process1(image, gamma=2.0,alpha=0.8,beta=2.0):

    image1 = image.astype(np.float32)
    image1 = image1**(gamma)
    image1 = image1/image1.max()*255

    image2 = (alpha)*image1 + (1-alpha)*image
    image2 = np.clip(beta*image2,0,255).astype(np.uint8)

    image  = image2
    return image


## make random transform =====================================================================
def random_transform(image, u, func, **kwargs):
    if random.random() < u:

        limits = []
        for k in kwargs:
            # print (k)
            # print (kwargs[k])
            # print ('')

            limit = kwargs[k]
            l = random.uniform(limit[0],limit[1])
            limits.append(l)

        #print(limits)
        image = func(image, *limits)

    return image


def random_transform2(image, mask, u, func, **kwargs):
    if random.random() < u:

        limits = []
        for k in kwargs:
            #print (k)
            #print (kwargs[k])
            #print ('')

            limit = kwargs[k]
            l = random.uniform(limit[0],limit[1])
            limits.append(l)

        #print("limit", limits)
        image, mask = func(image, mask, *limits)

    return image, mask

## illumination ====================================================================================

def do_brightness_shift(image, alpha=0.125):
    image = image.astype(np.float32)
    image = image + alpha*255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_brightness_multiply(image, alpha=1):
    image = image.astype(np.float32)
    image = alpha*image
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_contrast(image, alpha=1.0):
    image = image.astype(np.float32)
    gray  = image * np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
    gray  = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    image = alpha*image  + gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table) # apply gamma correction using the lookup table


def do_clahe(image, clip=2, grid=16):
    grid=int(grid)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid)).apply(gray)
    lab   = cv2.merge((gray, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image

## color ====================================================================================
def do_decolor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

def do_saturation_shift(image, alpha=1):

    gray  = image * np.array([[[0.114, 0.587,  0.299]]])
    gray  = np.sum(gray,axis=2, keepdims=True)
    image = alpha*image  + (1.0 - alpha)*gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def do_hue_shift(image, alpha=0):
    h   = int(alpha*180)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


# https://www.kaggle.com/c/data-science-bowl-2018/discussion/53940
def do_color_shift(image, alpha0=0,alpha1=0, alpha2=0):
    image = image.astype(np.float32) + np.array([alpha0,alpha1,alpha2])*255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


## filter ====================================================================================

def do_unsharp(image, size=9, strength=0.25, alpha=5 ):
    image = image.astype(np.float32)
    size  = 1+2*(int(size)//2)
    strength = strength*255
    blur  = cv2.GaussianBlur(image, (size,size), strength)
    image = alpha*image + (1-alpha)*blur
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image



#noise
def do_gaussian_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = np.random.normal(0,sigma,(H,W))
    noisy = gray + noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def do_speckle_noise(image, sigma=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = sigma*np.random.randn(H,W)
    noisy = gray + gray * noise

    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

def do_inv_speckle_noise(image, sigma=0.5):
    #print(image.shape)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray, a, b = cv2.split(lab)
    gray = gray.astype(np.float32)/255
    H,W  = gray.shape

    noise = sigma*np.random.randn(H,W)
    noisy = gray + (1-gray) * noise
    #print(gray.shape, noise.shape)
    noisy = (np.clip(noisy,0,1)*255).astype(np.uint8)
    #print(noisy.shape, a.shape, b.shape)
    lab   = cv2.merge((noisy, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image

# elif noise_type == "salt_and_pepper":
#     salt_pepper_ratio = 0.5
#     noisy    = np.copy(gray)
#
#     num_salt = np.ceil(limit * H*W * salt_pepper_ratio)
#     coords = [np.random.randint(0, i - 1, int(num_salt)) for i in gray.shape]
#     noisy[coords] = 1
#
#     num_pepper = np.ceil(amount* H*W * (1. - salt_pepper_ratio))
#     coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in gray.shape]
#     noisy[coords] = 0
#
# elif noise_type == "poisson":
#     num_values = len(np.unique(gray))
#     num_values = 2 ** np.ceil(np.log2(num_values))
#     noisy      = np.random.poisson(gray * num_values) / float(num_values)


## geometric ====================================================================================
def relabel_mask(mask):

    data = mask[:,:,np.newaxis]
    unique_color = set( tuple(v) for m in data for v in m )
    print(unique_color)

    H,W  = data.shape[:2]
    mask = np.zeros((H,W),np.int32)
    for color in unique_color:
        #print(color)
        if color == (0,): continue

        m = (data==color).all(axis=2)
        label  = skimage.morphology.label(m)

        index = [label!=0]
        mask[index] = label[index]+mask.max()

    return mask


def do_shift_scale_rotate2( image, mask, dx=0, dy=0, scale=1, angle=0 ):
    borderMode=cv2.BORDER_REFLECT_101
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width, channel = image.shape
    sx = scale
    sy = scale
    cc = math.cos(angle/180*math.pi)*(sx)
    ss = math.sin(angle/180*math.pi)*(sy)
    rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

    box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
    box1 = box0 - np.array([width/2,height/2])
    box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0,box1)

    image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    mask = mask.astype(np.float32)
    mask = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = mask.astype(np.int32)
    #mask = relabel_mask(mask)

    return image, mask

def do_flip_transpose2(image, mask, type=0):
    #choose one of the 8 cases

    if type==1: #rotate90
        image = image.transpose(1,0,2)
        image = cv2.flip(image,1)

        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,1)


    if type==2: #rotate180
        image = cv2.flip(image,-1)
        mask  = cv2.flip(mask,-1)


    if type==3: #rotate270
        image = image.transpose(1,0,2)
        image = cv2.flip(image,0)

        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,0)


    if type==4: #flip left-right
        image = cv2.flip(image,1)
        mask  = cv2.flip(mask,1)


    if type==5: #flip up-down
        image = cv2.flip(image,0)
        mask  = cv2.flip(mask,0)

    if type==6:
        image = cv2.flip(image,1)
        image = image.transpose(1,0,2)
        image = cv2.flip(image,1)

        mask = cv2.flip(mask,1)
        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,1)

    if type==7:
        image = cv2.flip(image,0)
        image = image.transpose(1,0,2)
        image = cv2.flip(image,1)

        mask = cv2.flip(mask,0)
        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,1)


    return image, mask


def do_crop2(image, mask, x,y,w,h):

    H,W = image.shape[:2]
    assert(H>=h)
    assert(W>=w)

    if (x==-1 & y==-1):
        x=(W-w)//2
        y=(H-h)//2

    if (x,y,w,h) != (0,0,W,H):
        image = image[y:y+h, x:x+w]
        mask  = mask[y:y+h, x:x+w]

    return image, mask


def fix_crop_transform2(image, mask, x,y,w,h):
    return do_crop2(image, mask, x,y,w,h )

def random_crop_transform2(image, mask, w,h, u=0.5):
    x,y = -1,-1
    if random.random() < u:

        H,W = image.shape[:2]
        if H!=h:
            y = np.random.choice(H-h)
        else:
            y=0

        if W!=w:
            x = np.random.choice(W-w)
        else:
            x=0

    return do_crop2(image, mask, x,y,w,h )


def do_elastic_transform2(image, mask, grid=32, distort=0.2):
    borderMode=cv2.BORDER_REFLECT_101
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width,np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end   = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step*(1+random.uniform(-distort,distort))

        xx[start:end] = np.linspace(prev,cur,end-start)
        prev=cur


    y_step = int(grid)
    yy = np.zeros(height,np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end   = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step*(1+random.uniform(-distort,distort))

        yy[start:end] = np.linspace(prev,cur,end-start)
        prev=cur

    #grid
    map_x,map_y =  np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    #image = map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=borderMode,borderValue=(0,0,0,))

    mask = mask.astype(np.float32)
    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=borderMode,borderValue=(0,0,0,))
    mask = mask.astype(np.int32)
    #mask = relabel_mask(mask)

    return image, mask


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    image = np.zeros((5,5))
    random_dummy_transform(image, 0.5, dummy_transform,  limits1=[1,2], limits2=[4,5], limits3=[8,9]  )


    print('\nsucess!')
