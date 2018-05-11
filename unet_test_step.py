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
    print('Load Configure File')
    config = parse_config(config_file)
    config_data = config['data']
    config_net1 = config.get('network1', None)
    config_test = config['testing']
    batch_size  = config_test.get('batch_size', 1)

    # 2.1, network for whole tumor
    print('Construct Network Graph for Brain Tumor')
    if (config_net1):
        net_type1 = config_net1['net_type']
        net_name1 = config_net1['net_name']
        data_shape1 = config_net1['data_shape']
        label_shape1 = config_net1['label_shape']
        class_num1 = config_net1['class_num']

        # construct graph for 1st network
        full_data_shape1 = [batch_size] + data_shape1

        x1 = tf.placeholder(tf.float32, shape=full_data_shape1)
        net_class1 = NetFactory.create(net_type1)
        net1 = net_class1(num_classes=class_num1, w_regularizer=None,
                          b_regularizer=None, name=net_name1)
        net1.set_params(config_net1)
        predicty1 = net1(x1, is_training=True)
        proby1 = tf.nn.softmax(predicty1)

    # 3, create session and load trained models
    print('Create Session and Load Trained Models')
    all_vars = tf.global_variables()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    if(config_net1):
        net1_vars = [x for x in all_vars if x.name[0:len(net_name1) + 1]==net_name1 + '/']
        saver1 = tf.train.Saver(net1_vars)
        saver1.restore(sess, config_net1['model_file'])

    # 4, load test images
    print('Load Test Images')
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    image_num = dataloader.get_total_image_number()
    print('image_num is ', image_num)

    # 5, start to test
    test_slice_direction = config_test.get('test_slice_direction', 'all')
    save_folder = config_data['save_folder']
    test_time = []
    struct = ndimage.generate_binary_structure(3, 2)
    margin = config_test.get('roi_patch_margin', 5)

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python test.py config17/test_all_class.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    test(config_file)