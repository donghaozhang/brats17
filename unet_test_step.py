# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function
import numpy as np
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
# from tensorflow.contrib.data import Iterator
from util.data_loader import *
from util.data_process import *
from util.train_test_func import *
from util.parse_config import parse_config
from train import NetFactory

def test(config_file):
    # 1, load configure file
    print('1 Load Configure File')
    config = parse_config(config_file)
    config_data = config['data']
    config_net1 = config.get('network1', None)
    config_test = config['testing']
    batch_size  = config_test.get('batch_size', 1)

    # 2.1, network for whole tumor
    print('2.1 Construct Network Graph for Brain Tumor')
    if (config_net1):
        net_type1 = config_net1['net_type']
        net_name1 = config_net1['net_name']
        data_shape1 = config_net1['data_shape']
        print('the data_shape1 used for training is ', data_shape1)
        label_shape1 = config_net1['label_shape']
        print('the label_shape1 used for training is ', label_shape1)
        class_num1 = config_net1['class_num']

        # construct graph for 1st network
        full_data_shape1 = [batch_size] + data_shape1
        print('full_data_shape1 is ', full_data_shape1)
        x1 = tf.placeholder(tf.float32, shape=full_data_shape1)
        net_class1 = NetFactory.create(net_type1)
        net1 = net_class1(num_classes=class_num1, w_regularizer=None,
                          b_regularizer=None, name=net_name1)
        # net1.set_params(config_net1)
        if net_type1 == 'MSNet':
            net1.set_params(config_net1)
        predicty1 = net1(x1, is_training=True)
        proby1 = tf.nn.softmax(predicty1)

    # 3, create session and load trained models
    print('3. Create Session and Load Trained Models')
    all_vars = tf.global_variables()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    if(config_net1):
        net1_vars = [x for x in all_vars if x.name[0:len(net_name1) + 1]==net_name1 + '/']
        # for x in all_vars:
            # print('the value of x is ', x.name)
        saver1 = tf.train.Saver(net1_vars)
        saver1.restore(sess, config_net1['model_file'])

    # 4, load test images
    print('4. Load Test Images')
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    image_num = dataloader.get_total_image_number()
    print('image_num is ', image_num)

    # 5, start to test
    print('5. Start to test')
    test_slice_direction = config_test.get('test_slice_direction', 'all')
    save_folder = config_data['save_folder']
    test_time = []
    struct = ndimage.generate_binary_structure(3, 2)
    margin = config_test.get('roi_patch_margin', 5)

    for i in range(image_num):
        [temp_imgs, temp_weight, temp_name, img_names, temp_bbox, temp_size] = dataloader.get_image_data_with_name(i)
        t0 = time.time()
        print('temp_imgs', temp_imgs[0].shape, temp_imgs[1].shape)
        prob1 = volume_probability_prediction_dynamic_shape(temp_imgs, data_shape1, label_shape1, data_shape1[-1],
                                                    class_num1, batch_size, sess, net1, net_type1)
        pred1 =  np.asarray(np.argmax(prob1, axis = 3), np.uint16)
        print('the shape of pred1 is ', pred1.shape)
        pred1 = pred1 * temp_weight
        print('the shape of pred1 is ', pred1.shape)
        out_label = np.asarray(pred1, np.int16)
    test_time.append(time.time() - t0)
    final_label = np.zeros(temp_size, np.int16)
    print('temp_size is ', temp_size)
    print('the value of temp_bbox ', temp_bbox)
    final_label = set_ND_volume_roi_with_bounding_box_range(final_label, temp_bbox[0], temp_bbox[1], out_label)
    print('final_label is ', final_label.shape)
    print('final label element :', np.unique(final_label))
    save_array_as_nifty_volume(final_label, save_folder + "/{0:}.nii.gz".format(temp_name), img_names[0])
    print(temp_name)

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('python unet_test_step.py config17/UNet3D_test_step_wt.txt')
        exit()
    # Disable the debug information of tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    test(config_file)