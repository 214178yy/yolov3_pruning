#!/usr/bin/env python

import tensorflow as tf
import os
import sys

# from model import yolov3
from sparsemodel import yolov3
import os

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange

import args

from utils.data_utils import get_batch_data
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms
#sys.path.append('../..')


#tf.app.flags.DEFINE_string('train_dir', './checkpoint', 'training directory')

tf.app.flags.DEFINE_string('ckpt', './checkpoint/altered/0.5036_prune0.2', 'ckpt to be frozen')
# tf.app.flags.DEFINE_string('output_node', 'MobilenetV1/Squeeze', 'name of output name')
tf.app.flags.DEFINE_string('frozen_pb_name', 'sparse.pb', 'output pb name')
FLAGS = tf.app.flags.FLAGS

def freeze():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    # setting placeholders
    is_training = tf.placeholder(tf.bool, name="phase_train")
    handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
    # register the gpu nms operation here for the following evaluation scheme
    pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])

    ##################
    # tf.data pipeline
    ##################
    train_dataset = tf.data.TextLineDataset(args.train_file)
    train_dataset = train_dataset.shuffle(args.train_img_cnt)
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, args.class_num, args.img_size, args.anchors, 'train', args.multi_scale_train,
                                  args.use_mix_up, args.letterbox_resize],
                             Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    train_dataset = train_dataset.prefetch(args.prefetech_buffer)

    val_dataset = tf.data.TextLineDataset(args.val_file)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, args.class_num, args.img_size, args.anchors, 'val', False, False,
                                  args.letterbox_resize],
                             Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    val_dataset.prefetch(args.prefetech_buffer)
    
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    # get an element from the chosen dataset iterator
    image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]

    # tf.data pipeline will lose the data `static` shape, so we need to set it manually
    image_ids.set_shape([None])
    image.set_shape([None, None, None, 3])
    for y in y_true:
        y.set_shape([None, None, None, None, None])

    ##################
    # Model definition
    ##################
    yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay,
                        args.weight_decay, use_static_shape=False)
    image = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='IteratorGetNext')
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(image, is_training=False)

    tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())

    # write frozen graph
    saver = tf.train.Saver(tf.global_variables())

    # if ckpt and ckpt.model_checkpoint_path:
    print('====>>start  check')
    saver.restore(sess, FLAGS.ckpt)
    print('======================>>>sucess restore')
    # else:
    #     print('no path===============================>>>')

    #  tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    #  frozen_func.graph.as_graph_def()
    print( '====>> input')
    print(tf.get_default_graph().get_tensor_by_name("IteratorGetNext:1"))
    #IteratorGetNext_1:0
    print(tf.get_default_graph().get_tensor_by_name("IteratorGetNext_1:0"))


    frozen_gd = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['yolov3/yolov3_head/feature_map_001','yolov3/yolov3_head/feature_map_002','yolov3/yolov3_head/feature_map_003'])
    tf.train.write_graph(
        frozen_gd,
        './freeze/',
        FLAGS.frozen_pb_name,
        as_text=False)
    print('======>>save')
    
def main(unused_arg):
    freeze()

if __name__ == '__main__':
    tf.app.run(main)
