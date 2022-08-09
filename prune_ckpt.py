#!/usr/bin/env python

import tensorflow as tf
import os
import sys

# from model import yolov3
from sparsemodel import yolov3
import tensorflow as tf
import numpy as np
import args
import json

from utils.data_utils import get_batch_data
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms
#sys.path.append('../..')


tf.app.flags.DEFINE_string('train_dir', './checkpoint', 'training directory')
tf.app.flags.DEFINE_string('restore_path', './checkpoint/altered/0.5036_prune0.2', 'training directory')
tf.app.flags.DEFINE_string('model_name', 'pruned', 'model name')
tf.app.flags.DEFINE_string('new_checkpoint_path', './checkpoint/pruned/', 'model name')
tf.app.flags.DEFINE_string('frozen_pb_name', 'sparse.pb', 'output pb name')
FLAGS = tf.app.flags.FLAGS

def freeze():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

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

    graph=tf.get_default_graph()
    #tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())


    # write frozen graph
    saver = tf.train.Saver(tf.global_variables())
    # tensor_to_restore = tf.contrib.framework.get_variables_to_restore()
    # for i in tensor_to_restore:
    #     print(i)
    # print(tensor_to_restore)

    input_layers = {}
    with open("./data/net_def.json", 'r') as f:
        net_def_list = json.loads(f.read())
        for i in net_def_list:
            if "yolov3_darknet53_body_Conv" in i['name']:
                layer_name="yolov3/darknet53_body/"+i['name'][22:-12]
                if 'Conv' in i['previous_layer'][0]:
                    input_layers[layer_name]="yolov3/darknet53_body/"+i['previous_layer'][0][22:-12]
                else:
                    input_layers[layer_name]=-1
            elif "yolov3_yolov3_head_Conv" in i['name']:
                layer_name="yolov3/yolov3_head/"+i['name'][19:-12]
                if 'Conv' in i['previous_layer'][0]:
                    input_layers[layer_name] = "yolov3/yolov3_head/" + i['previous_layer'][0][19:-12]
                else:
                    input_layers[layer_name] = -1

    ##############
    #剪枝层权重初始化
    ##############
    output_channel={}
    for var_name, _ in tf.contrib.framework.list_variables(FLAGS.restore_path):  # 得到checkpoint文件中所有的参数（名字，形状）元组
        var = tf.contrib.framework.load_variable(FLAGS.restore_path, var_name)  # 得到上述参数的值
        pair_tensor=tf.contrib.framework.get_variables_to_restore([var_name])
        if var.shape==pair_tensor[0].shape:
            update=tf.assign(pair_tensor[0],var)
            sess.run(update)
        else:
            pass
        if "gamma" in var_name:
            out_channel_idx = np.argwhere(var)[:, 0].tolist()
            output_channel[var_name[0:-6]]=out_channel_idx

    for vname in output_channel.keys():
        idx=output_channel[vname]
        gamma_var = tf.contrib.framework.load_variable(FLAGS.restore_path, vname + '/gamma')
        gamma_tensor = tf.contrib.framework.get_variables_to_restore([vname+'/gamma'])
        update_gamma=tf.assign(gamma_tensor[0],gamma_var[idx])
        sess.run(update_gamma)
        beta_var = tf.contrib.framework.load_variable(FLAGS.restore_path, vname + '/beta')
        beta_tensor = tf.contrib.framework.get_variables_to_restore([vname + '/beta'])
        update_beta = tf.assign(beta_tensor[0], beta_var[idx])
        sess.run(update_beta)
        mm_var = tf.contrib.framework.load_variable(FLAGS.restore_path, vname + '/moving_mean')
        mm_tensor = tf.contrib.framework.get_variables_to_restore([vname + '/moving_mean'])
        update_mm = tf.assign(mm_tensor[0], mm_var[idx])
        sess.run(update_mm)
        mv_var = tf.contrib.framework.load_variable(FLAGS.restore_path, vname + '/moving_variance')
        mv_tensor = tf.contrib.framework.get_variables_to_restore([vname + '/moving_variance'])
        update_mv = tf.assign(mv_tensor[0], mv_var[idx])
        sess.run(update_mv)
        # conv weights的shape是(kernel_size,kernel_size,input_channel,output_channel)
        conv_var = tf.contrib.framework.load_variable(FLAGS.restore_path, vname[0:-9] + 'weights')
        conv_tensor = tf.contrib.framework.get_variables_to_restore([vname[0:-9] + 'weights'])
        input_name = input_layers[vname[0:-10]]
        if input_name == -1:
            update_conv=tf.assign(conv_tensor[0], conv_var[:,:,:,idx])
        else:
            input_name=input_name+'/BatchNorm'
            input_channel_idx=output_channel[input_name]
            temp_var=conv_var[:,:,:,idx]
            update_conv = tf.assign(conv_tensor[0], temp_var[:, :, input_channel_idx, :])
        sess.run(update_conv)

    checkpoint_path = os.path.join(FLAGS.new_checkpoint_path, FLAGS.model_name)
    saver.save(sess,checkpoint_path)
    # # if ckpt and ckpt.model_checkpoint_path:
    # print('====>>start  check')
    # saver.restore(sess, './checkpoint/altered/0.5036_prune0.2')
    # print('======================>>>sucess restore')
    # # else:
    # #     print('no path===============================>>>')
    #
    # #  tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # #  frozen_func.graph.as_graph_def()
    print( '====>> input')
    print(tf.get_default_graph().get_tensor_by_name("IteratorGetNext:1"))
    # #IteratorGetNext_1:0
    # # print(tf.get_default_graph().get_tensor_by_name("IteratorGetNext_1:0"))
    #
    print('====>freeze')
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
