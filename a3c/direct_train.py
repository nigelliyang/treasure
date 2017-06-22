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

from lstm_ACNetwork import LSTM_ACNetwork
from thread import TrainingThread, TestThread
from datetime import datetime
from config import *
from futuresData import Futures_cn
from direct_sharing_lstm_ACNetwork import Direct_Sharing_LSTM_ACNetwork

path_list = [
    'data/IC00.csv',
    'data/TF.csv',
]
data = Futures_cn()
data.load_tranform(path_list)

network = Direct_Sharing_LSTM_ACNetwork(data.future_num, data.info_field_num, 0)

grad_applier = tf.train.RMSPropOptimizer(
        learning_rate = args.learning_rate,
        decay = args.rmsp_alpha,
        momentum = 0.0,
        epsilon = args.rmsp_epsilon)

optimizer = grad_applier.minimize(-network.totallogreward)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

training_iters = 10000
display_step = 10
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    i = 0
    while i < training_iters:
        choice,firstprice,nextprice,data_nolast = data.extract_day_for_directTrain()
        # Run optimization op (backprop)
        feed_dict = {
                network.s : data_nolast,
                network.first_price : firstprice,
                network.next_price : nextprice
                }
        sess.run(optimizer, feed_dict=feed_dict)
        
        if i % display_step == 0:
            # Calculate reward
            totalreward = sess.run(network.totalreward, feed_dict=feed_dict)

            print("Iter " + str(i) + ", daily reward = " + \
                  "{:.6f}".format(totalreward))

        i = i + 1
    print("Optimization Finished!")

    # Calculate accuracy
    #test_data = testset.data
    #test_label = testset.labels
    #test_seqlen = testset.seqlen
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                                  seqlen: test_seqlen}))
