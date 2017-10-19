# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import threading
import signal
import random
import utils
import math
import time
import os

from a3c.config import *
from futuresData import Futures_cn
from direct_sharing_lstm_ACNetwork import Direct_Sharing_LSTM_ACNetwork

path_list = [
    'data/IC00.csv',
    'data/TF.csv',
]
data = Futures_cn()
data.load_tranform(path_list)

network = Direct_Sharing_LSTM_ACNetwork(data.future_num, data.info_field_num)

grad_applier = tf.train.RMSPropOptimizer(
    learning_rate=args.learning_rate,
    decay=args.rmsp_alpha,
    momentum=0.0,
    epsilon=args.rmsp_epsilon)

with tf.name_scope('direct_train') as vs:
    optimizer = grad_applier.minimize(-network.totallogreward)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()

training_iters = 10000
display_step = 10
# Launch the graph
with tf.Session() as sess:
    tf.summary.scalar('tt', tf.constant(0, dtype=tf.float32))
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('tb_logs', sess.graph)

    sess.run(init_op)
    # Keep training until reach max iterations
    i = 0
    while i < training_iters:
        choice, firstprice, nextprice, data_nolast = data.extract_day_for_directTrain()
        # Run optimization op (backprop)
        feed_dict = {
            network.s: data_nolast,
            network.first_price: firstprice,
            network.next_price: nextprice
        }
        _ = sess.run([optimizer], feed_dict=feed_dict)

        if i % display_step == 0:
            # Calculate reward of the day
            totalreward = sess.run(network.totalreward, feed_dict=feed_dict)

            print("Iter " + str(i) + ", daily reward = " + \
                  "{:.6f}".format(totalreward))

            summary_str = sess.run(merged_summary_op)
            summary_writer.add_summary(summary_str, totalreward)

        i = i + 1
    print("Optimization Finished!")
    print('model saved in :', saver.save(sess, 'vars'))

    # Calculate accuracy
    # test_data = testset.data
    # test_label = testset.labels
    # test_seqlen = testset.seqlen
    # print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                                  seqlen: test_seqlen}))
