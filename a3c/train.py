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

from sharing_lstm_ACNetwork import Sharing_LSTM_ACNetwork
from lstm_ACNetwork import LSTM_ACNetwork
from thread import TrainingThread, TestThread
from datetime import datetime
from config import *

global_t = 0

stop_requested = False

if args.share_variable:
    global_network = Sharing_LSTM_ACNetwork(args.action_size, -1)
else:
    global_network = LSTM_ACNetwork(args.action_size, -1)
saver = tf.train.Saver(global_network.vars)

grad_applier = tf.train.RMSPropOptimizer(
        learning_rate = args.learning_rate,
        decay = args.rmsp_alpha,
        momentum = 0.0,
        epsilon = args.rmsp_epsilon)

local_networks = []
for i in range(args.thread_num):
    thread = TrainingThread(i, global_network, grad_applier, args.max_time_step)
    local_networks.append(thread)

test_determinate_network = TestThread(-2, global_network, grad_applier, args.max_time_step, use_test_data=True)
test_determinate_network.monitor = utils.invest_monitor(save_dir='determinate_log')

# prepare session
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto(allow_soft_placement = True)
with tf.Session(config=config) as sess:

    sess.run(init_op)
    # if use checkpoint is required, restore the checkpoint
    if args.use_checkpoint:
        ckpt = tf.train.get_checkpoint_state(os.path.join(args.checkpoint_dir, args.test_name))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_t = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("Load ckpt form ", ckpt.model_checkpoint_path)

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
        prev_global_t = global_t
        tic = datetime.now()
        # give the benchmark return first
        benchmark_return = network.determinate_test(sess, lazy=True)
        print("benchmark return: %.3f" %benchmark_return)
        # test the model every 30 seconds
        while True:
            if stop_requested or global_t > args.max_time_step:
                network.monitor.save(file_name=args.test_name)
                break
            if global_t - prev_global_t > args.test_steps:
                if args.short_sight:
                    determinate_invest_return = network.short_sight_test(sess)
                else:
                    determinate_invest_return = network.determinate_test(sess)
                toc = datetime.now()
                print("%s step %d determinate policy return: %.3f" %(toc-tic, global_t, determinate_invest_return))
            prev_global_t = global_t
            time.sleep(3)

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

    if not tf.gfile.Exists(os.path.join(args.checkpoint_dir, args.test_name)):
        tf.gfile.MkDir(os.path.join(args.checkpoint_dir, args.test_name))
    saver.save(sess, os.path.join(args.checkpoint_dir, args.test_name, args.test_name), global_step=global_t)
    print('Saved checkpoint at', os.path.join(args.checkpoint_dir, args.test_name))
    with open(os.path.join(args.checkpoint_dir, args.test_name, 'config_list.txt'), 'a') as f:
        f.write(str(args))
        print("Saved config at ", os.path.join(args.checkpoint_dir, args.test_name, 'config_list.txt'))

    for t in train_threads:
        t.join()
    test_thread.join()


