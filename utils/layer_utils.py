# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def conv2d(inputs, filters, kernel_size, strides=1, is_training=True):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 2
        pad_end = pad_total // 2
        pad_beg = pad_total - pad_end

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)

    normalizer_fn=tf.contrib.slim.batch_norm
    normalizer_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    weights_initializer = tf.truncated_normal_initializer(stddev=0.09)
    weights_regularizer = tf.contrib.layers.l2_regularizer(0.00004)

    inputs = slim.conv2d(inputs,filters,
                                 kernel_size,
                                 stride=strides,
                                 normalizer_fn=normalizer_fn,
                                 activation_fn=tf.nn.relu6,
                                 normalizer_params=normalizer_params,
                                 weights_initializer=weights_initializer,
                                 weights_regularizer=weights_regularizer,
                         padding=('SAME' if strides == 1 else 'VALID'))

  #  inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                     #   padding=('SAME' if strides == 1 else 'VALID'))

    return inputs

def darknet53_body(inputs, is_training=True):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1,  is_training=is_training)
        net = conv2d(net, filters * 2, 3, is_training=is_training)

        net = net + shortcut

        return net
    
    # first two conv2d layers
    net = conv2d(inputs, 32,  3, strides=1, is_training=is_training)
    net = conv2d(net, 64,  3, strides=2, is_training=is_training)

    # res_block * 1
    net = res_block(net, 32)
    # shortcut = net
    # net = conv2d(net, 28, 1, is_training=is_training)  # conv2
    # net = conv2d(net, 64, 3, is_training=is_training)  # conv3
    # net = net + shortcut

    net = conv2d(net, 128, 3, strides=2, is_training=is_training)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2, is_training=is_training)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2, is_training=is_training)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2, is_training=is_training)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3


def yolo_block(inputs, filters, is_training=True):
    net = conv2d(inputs, filters * 1, 1, is_training=is_training)
    net = conv2d(net, filters * 2, 3, is_training=is_training)
    net = conv2d(net, filters * 1, 1, is_training=is_training)
    net = conv2d(net, filters * 2, 3, is_training=is_training)
    net = conv2d(net, filters * 1, 1, is_training=is_training)
    route = net
    net = conv2d(net, filters * 2, 3, is_training=is_training)
    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs


