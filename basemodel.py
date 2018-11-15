#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: basemodel.py

import tensorflow as tf
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.models import (
    Conv2D, MaxPooling, BatchNorm, BNReLU, layer_register)
import config

@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=tf.constant_initializer(1.)):
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name='output')
    return tf.reshape(out, orig_shape, name='output')

def get_norm(zero_init=False):
    if config.NORM == 'GN':
        Norm = GroupNorm
        layer_name = 'gn'
    else:
        Norm = BatchNorm
        layer_name = 'bn'
    return lambda x, name=None: Norm(layer_name, x, gamma_initializer=tf.zeros_initializer() if zero_init else None)

def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)

        mean = [0.485, 0.456, 0.406]    # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std
        return image


def get_bn(zero_init=False):
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)


def resnet_shortcut(l, n_out, stride, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        if stride == 2:
            l = l[:, :, :-1, :-1]
            return Conv2D('convshortcut', l, n_out, 1,
                          stride=stride, padding='VALID', nl=nl)
        else:
            return Conv2D('convshortcut', l, n_out, 1,
                          stride=stride, nl=nl)
    else:
        return l


def resnet_bottleneck(l, ch_out, stride):
    l, shortcut = l, l
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    if stride == 2:
        l = tf.pad(l, [[0, 0], [0, 0], [0, 1], [0, 1]])
        l = Conv2D('conv2', l, ch_out, 3, stride=2, nl=BNReLU, padding='VALID')
    else:
        l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_norm(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_norm(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features,
                               stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l

def pretrained_resnet_FPN(image, num_blocks, prefix=''):
    def nonlin(x, name):
        x = get_norm()(x)
        return tf.nn.relu(x)

    assert len(num_blocks) == 4
    end_points = {}
    featuremaps = []
    with argscope([Conv2D, MaxPooling, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, nl=nonlin, use_bias=False), \
            argscope(BatchNorm, use_local_stat=False):
        l = tf.pad(image, [[0, 0], [0, 0], [2, 3], [2, 3]])
        l = Conv2D(prefix + 'conv0', l, 64, 7, stride=2, padding='VALID')
        l = tf.pad(l, [[0, 0], [0, 0], [0, 1], [0, 1]])
        l = MaxPooling(prefix + 'pool0', l, shape=3, stride=2, padding='VALID')
        l = resnet_group(l, prefix + 'group0', resnet_bottleneck, 64, num_blocks[0], 1)
        if config.FREEZE_C2:
            l = tf.stop_gradient(l)
        end_points['C2'] = l
        l = resnet_group(l, prefix + 'group1', resnet_bottleneck, 128, num_blocks[1], 2)
        end_points['C3'] = l
        l = resnet_group(l, prefix + 'group2', resnet_bottleneck, 256, num_blocks[2], 2)
        end_points['C4'] = l
        l = resnet_group(l, prefix + 'group3', resnet_bottleneck, 512, num_blocks[3], stride=2)
        end_points['C5'] = l
        
        # build FPN head
        for stage in range(5, 1, -1):
            if stage == 5:
                end_points['P{}'.format(stage)] = \
                    Conv2D(prefix+'fpn_lateral_{}'.format(stage), 
                           end_points['C{}'.format(stage)],
                           256, 1, padding='SAME')
                end_points['P6'] = MaxPooling(prefix + 'pool_fpn6', tf.pad(end_points['P5'], [[0, 0], [0, 0], [0, 1], [0, 1]]), shape=3, stride=2, padding='VALID')
            else:
                lateral = Conv2D(prefix+'fpn_lateral_{}'.format(stage), 
                           end_points['C{}'.format(stage)],
                           256, 1, padding='SAME')
                upsample = end_points['P{}'.format(stage+1)]
                fused = tf.add(
                            tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(upsample, [0, 2, 3, 1]), size=tf.shape(lateral)[-2:]), [0, 3, 1, 2]),
                            lateral
                        )
                end_points['P{}'.format(stage)] = Conv2D(prefix+"fpn_fused_{}".format(stage), fused, 256, 3, padding='SAME')
        # add P6 for RPN
        #end_points['P6'] = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(upsample, [0, 2, 3, 1]), size=tf.shape(end_points['P5'])[-2:]//2 ), [0, 3, 1, 2])
        print("FPN keys:", end_points.keys())
        for stage in range(6, 1, -1):
            featuremaps.append(end_points['P{}'.format(stage)])
        """
        if config.PAN:
            for stage in range(2, 7):
                if stage == 2:
                    end_points['N{}'.format(stage)] = end_points['P{}'.format(stage)]
                else:
                    downsample = tf.pad(end_points['N{}'.format(stage-1)], [[0, 0], [0, 0], [0, 1], [0, 1]])
                    downsample = Conv2D(prefix+'pan_down_{}'.format(stage), 
                            downsample,
                            256, 3, stride=2, padding='VALID')
                    lateral = end_points['P{}'.format(stage)]
                    fused = downsample + lateral
                    end_points['N{}'.format(stage)] = Conv2D(prefix+"pan_fused_{}".format(stage), fused, 256, 3, padding='SAME')
            # add P6 for RPN
            #end_points['P6'] = tf.transpose(tf.image.resize_nearest_neighbor(tf.transpose(upsample, [0, 2, 3, 1]), size=tf.shape(end_points['P5'])[-2:]//2 ), [0, 3, 1, 2])
            print("PAN keys:", end_points.keys())
            for stage in range(6, 1, -1):
                featuremaps.append(end_points['N{}'.format(stage)])
        """
    # 16x downsampling up to now
    return featuremaps

def pretrained_resnet_conv4(image, num_blocks, prefix=''):
    assert len(num_blocks) == 3
    with argscope([Conv2D, MaxPooling, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, nl=tf.identity, use_bias=False), \
            argscope(BatchNorm, use_local_stat=None):
        l = tf.pad(image, [[0, 0], [0, 0], [2, 3], [2, 3]])
        l = Conv2D(prefix + 'conv0', l, 64, 7, stride=2, nl=BNReLU, padding='VALID')
        l = tf.pad(l, [[0, 0], [0, 0], [0, 1], [0, 1]])
        l = MaxPooling(prefix + 'pool0', l, shape=3, stride=2, padding='VALID')
        l = resnet_group(l, prefix + 'group0', resnet_bottleneck, 64, num_blocks[0], 1)
        # TODO replace var by const to enable folding
        l = tf.stop_gradient(l)
        l = resnet_group(l, prefix + 'group1', resnet_bottleneck, 128, num_blocks[1], 2)
        l = resnet_group(l, prefix + 'group2', resnet_bottleneck, 256, num_blocks[2], 2)
    # 16x downsampling up to now
    return l


@auto_reuse_variable_scope
def resnet_conv5(image, num_block):
    with argscope([Conv2D, BatchNorm], data_format='NCHW'), \
            argscope(Conv2D, nl=tf.identity, use_bias=False), \
            argscope(BatchNorm, use_local_stat=None):
        # 14x14:
        l = resnet_group(image, 'group3', resnet_bottleneck, 512, num_block, stride=2)
        return l
