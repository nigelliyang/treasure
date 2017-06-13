# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np
import random
import math
import os
import time
import utils
import signal

from datetime import datetime
from independentlstm import Independent_LSTM_ACNetwork
from network import LSTM_ACNetwork
from thread import TrainingThread
from config import *


global_t = 0

stop_requested = False

if args.share_variable:
    global_network = Independent_LSTM_ACNetwork(args.action_size, -1)
else:
    global_network = LSTM_ACNetwork(args.action_size, -1)

grad_applier = tf.train.RMSPropOptimizer(
        learning_rate = args.learning_rate,
        decay = args.rmsp_alpha,
        momentum = 0.0,
        epsilon = args.rmsp_epsilon)

local_networks = []
for i in range(args.thread_num):
    thread = TrainingThread(i, global_network, grad_applier, args.max_time_step)
    local_networks.append(thread)

test_determinate_network = TrainingThread(-2, global_network, grad_applier, args.max_time_step, use_test_data=True)
test_determinate_network.monitor = utils.invest_monitor(save_dir='determinate_log')

# prepare session
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto(allow_soft_placement = True)
with tf.Session(config=config) as sess:

    sess.run(init_op)
    def train(thread_index):
        global global_t
        network = local_networks[thread_index]
        # set start_time
        start_time = time.time()

        while True:
            if stop_requested:
                break
            if global_t > args.max_time_step:
                break
            diff_global_t = network.process(sess, global_t)
            global_t += diff_global_t

    def determinate_test(network):
        global global_t
        tic = datetime.now()
        # give the benchmark return first
        random_invest_return = network.determinate_test(sess, lazy=True)
        print("benchmark return: %.3f" %random_invest_return)
        # test the model every 30 seconds
        while True:
            determinate_invest_return = network.determinate_test(sess, lazy=False)
            toc = datetime.now()
            print("%s determinate policy return: %.3f" %(toc-tic, determinate_invest_return))
            if stop_requested or global_t > args.max_time_step:
                network.monitor.save(file_name='test_log')
                break
            time.sleep(30)

    def signal_handler(signal, frame):
        global stop_requested
        print('You pressed Ctrl+C!')
        stop_requested = True

    train_threads = []
    for i in range(args.thread_num):
        train_threads.append(threading.Thread(target=train, args=(i,)))
    test_thread = threading.Thread(target=determinate_test, args=(test_determinate_network,))

    for t in train_threads:
        t.start()
    test_thread.start()

    print('Press Ctrl+C to stop')
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
    print('Now saving data. Please wait')
    print('It may take a few seconds to finish the last test')

    for t in train_threads:
        t.join()
    test_thread.join()


