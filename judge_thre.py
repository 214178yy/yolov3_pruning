import tensorflow as tf
import argparse
import os
import numpy as np
import json

parser = argparse.ArgumentParser(description='')

parser.add_argument("--checkpoint_path", default='./checkpoint/xishutrain/best_model_Epoch_10_step_31679_mAP_0.4638_loss_13.0942_lr_0.001', help="restore ckpt")  # 原参数路径
parser.add_argument("--new_checkpoint_path", default='./checkpoint/altered/', help="path_for_new ckpt")  # 新参数保存路径
# parser.add_argument("--add_prefix", default='deeplab_v2/', help="prefix for addition")  # 新参数名称中加入的前缀名

args = parser.parse_args()


def main():
    if not os.path.exists(args.new_checkpoint_path):
        os.makedirs(args.new_checkpoint_path)
    with tf.Session() as sess:
        ###########
        # 测试剪枝gamma阈值时使用
        ###########
        gamma_list = []
        highest_thre = []

        ignore_set = set()
        ignore_set.add("yolov3_yolov3_head_Conv_22")
        ignore_set.add("yolov3_yolov3_head_Conv_14")
        ignore_set.add("yolov3_yolov3_head_Conv_6")
        with open("./data/net_def.json", 'r') as f:
            net_def_list = json.loads(f.read())
            for i in net_def_list:
                if i['operation'] == "add":
                    for name in i['previous_layer']:
                        ignore_set.add(name[0:-12])
                elif i['operation'] == "upsample":
                    for name in i['previous_layer']:
                        ignore_set.add(name[0:-12])
        for var_name, _ in tf.contrib.framework.list_variables(args.checkpoint_path):  # 得到checkpoint文件中所有的参数（名字，形状）元组
            var = tf.contrib.framework.load_variable(args.checkpoint_path, var_name)  # 得到上述参数的值
            if "gamma" in var_name:
                if var_name[0:-16].replace('/','_') in ignore_set:
                    continue
        ###########
        #测试剪枝gamma阈值时使用
        ###########
                print(var_name)
                gamma_list.append(var)
                highest_thre.append(np.max(var))
        highest_thre = min(highest_thre)
        gamma_list = [i for item in gamma_list for i in item]
        sorted_gamma = sorted(map(abs, gamma_list))
        thre_index = np.where(sorted_gamma == highest_thre)[0][0]
        percent_limit = thre_index / len(gamma_list)

        # print channel prune limit
        print(f'Threshold should be less than {highest_thre:.4f}.')
        print(f'The corresponding prune ratio is {percent_limit:.3f}.')
        percent = 0.4
        thre_index = int(len(sorted_gamma) * percent)
        thre = sorted_gamma[thre_index]
        print("used thresh is:", thre)

if __name__ == '__main__':
    main()