# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import json
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

    f = open("/home/yangy/detect_face_ori/net_channel.json", "r")
    channel_list = json.loads(f.read())
    f.close()
    
    # first two conv2d layers
    net = conv2d(inputs, channel_list["yolov3_darknet53_body_Conv"],  3, strides=1, is_training=is_training)#conv
    net = conv2d(net, 64,  3, strides=2, is_training=is_training)#conv1

    # res_block * 1
    shortcut=net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_2"], 1, is_training=is_training)#conv2
    net = conv2d(net, 64, 3, is_training=is_training)#conv3
    net = net+shortcut

    net = conv2d(net, 128, 3, strides=2, is_training=is_training)#conv4

    # res_block * 2
    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_5"], 1, is_training=is_training)  # conv5
    net = conv2d(net, 128, 3, is_training=is_training)  # conv6
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_7"], 1, is_training=is_training)  # conv7
    net = conv2d(net, 128, 3, is_training=is_training)  # conv8
    net = net + shortcut

    net = conv2d(net, 256, 3, strides=2, is_training=is_training)#conv9

    # res_block * 8
    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_10"], 1, is_training=is_training)  # conv10
    net = conv2d(net, 256, 3, is_training=is_training)  # conv11
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_12"], 1, is_training=is_training)  # conv12
    net = conv2d(net, 256, 3, is_training=is_training)  # conv13
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_14"], 1, is_training=is_training)  # conv14
    net = conv2d(net, 256, 3, is_training=is_training)  # conv15
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_16"], 1, is_training=is_training)  # conv16
    net = conv2d(net, 256, 3, is_training=is_training)  # conv17
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_18"], 1, is_training=is_training)  # conv18
    net = conv2d(net, 256, 3, is_training=is_training)  # conv19
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_20"], 1, is_training=is_training)  # conv20
    net = conv2d(net, 256, 3, is_training=is_training)  # conv21
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_22"], 1, is_training=is_training)  # conv22
    net = conv2d(net, 256, 3, is_training=is_training)  # conv23
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_24"], 1, is_training=is_training)  # conv24
    net = conv2d(net, 256, 3, is_training=is_training)  # conv25
    net = net + shortcut

    route_1 = net
    net = conv2d(net, 512, 3, strides=2, is_training=is_training)#conv26

    # res_block * 8
    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_27"], 1, is_training=is_training)  # conv27
    net = conv2d(net, 512, 3, is_training=is_training)  # conv28
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_29"], 1, is_training=is_training)  # conv29
    net = conv2d(net, 512, 3, is_training=is_training)  # conv30
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_31"], 1, is_training=is_training)  # conv31
    net = conv2d(net, 512, 3, is_training=is_training)  # conv32
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_33"], 1, is_training=is_training)  # conv33
    net = conv2d(net, 512, 3, is_training=is_training)  # conv34
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_35"], 1, is_training=is_training)  # conv35
    net = conv2d(net, 512, 3, is_training=is_training)  # conv36
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_37"], 1, is_training=is_training)  # conv37
    net = conv2d(net, 512, 3, is_training=is_training)  # conv38
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_39"], 1, is_training=is_training)  # conv39
    net = conv2d(net, 512, 3, is_training=is_training)  # conv40
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_41"], 1, is_training=is_training)  # conv41
    net = conv2d(net, 512, 3, is_training=is_training)  # conv42
    net = net + shortcut

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2, is_training=is_training)#conv43

    # res_block * 4
    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_44"], 1, is_training=is_training)  # conv44
    net = conv2d(net, 1024, 3, is_training=is_training)  # conv45
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_46"], 1, is_training=is_training)  # conv46
    net = conv2d(net, 1024, 3, is_training=is_training)  # conv47
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_48"], 1, is_training=is_training)  # conv48
    net = conv2d(net, 1024, 3, is_training=is_training)  # conv49
    net = net + shortcut

    shortcut = net
    net = conv2d(net, channel_list["yolov3_darknet53_body_Conv_50"], 1, is_training=is_training)  # conv50
    net = conv2d(net, 1024, 3, is_training=is_training)  # conv51
    net = net + shortcut

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


