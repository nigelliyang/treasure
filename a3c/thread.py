# -*- coding: utf-8 -*-
from network import LSTM_ACNetwork
from config import *
from environment import *
from futuresData import *

import tensorflow as tf
import numpy as np
import random

class TrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 optimizer,
                 max_global_steps):
        self.thread_index = thread_index
        self.max_global_steps = max_global_steps
        self.local_network = LSTM_ACNetwork(args.action_size, self.thread_index)
        self.local_network.prepare_loss(args.entropy_beta, args.risk_beta)

        self.opt = optimizer
        local_gradients = self.opt.compute_gradients(self.local_network.total_loss, self.local_network.vars)
        self.gradients = local_gradients
        # TODO: fix the clip by norm
        # The clip by norm function has been turned off
        # because this operator will cause nan output
        # self.gradients = [(tf.clip_by_norm(local_gradients[i][0], args.grad_norm_clip), global_network.vars[i]) for i in range(len(local_gradients))]
        self.apply_gradients = self.opt.apply_gradients(self.gradients)

        self.sync = self.local_network.sync_from(global_network)

        data = futuresData()
        data.loadData_moreday0607()
        self.env = futuresGame(data)
        self.terminal = True
        self.episode_reward = 1.0 # use multiplication model in futures game
        self.init_allocation = np.zeros(args.action_size)
        self.init_allocation[-1] = 1
        self.allocation = self.init_allocation

        self.local_t = 0

    def choose_action(self, gauss_mean, gauss_sigma):
        '''
        :param guass_mean:  array [] ndarray
        :param guass_sigma: matrix like [[],[],[]] ndarray
        :return: ndarray
        '''
        max_times = 1000
        def check(values):
            for a in values:
                if abs(a) > 3:
                    return False
            return True

        for i in range(max_times):
            values = np.random.multivariate_normal(gauss_mean,gauss_sigma)
            values = np.append(values, 1-np.sum(values))
            if check(values):
                return values
        print('bad luck for choosing %d times not find a good assignment, so return the guass_mean' % max_times)
        return np.append(gauss_mean + 1-np.sum(gauss_mean))

    def process(self, sess, global_t):
        previous_t = self.local_t

        states = []
        allocations = []
        actions = []
        rewards = []
        values = []

        sess.run(self.sync)

        if self.terminal:
            self.state = self.env.reset()
            self.local_network.reset_state_value()
            self.allocation = self.init_allocation

        for i in range(args.local_t_max):
            gauss_mean, value_ = self.local_network.run_policy_and_value(sess, self.state, self.allocation)
            action = self.choose_action(gauss_mean, args.gauss_sigma)

            states.append(self.state)
            allocations.append(self.allocation)
            actions.append(action)
            values.append(value_)

            # reward is the neat return rate of capital, like 0.03
            self.state, self.allocation, reward, self.terminal, _ = self.env.step(action)
            self.episode_reward *= (1.0+reward)
            # print(self.episode_reward)
            rewards.append(reward)
            self.local_t += 1
            if self.terminal:
                print("return rate = {}".format(self.episode_reward))
                self.episode_reward = 1.0
                break

        if self.terminal:
            R = 1.0
        else:
            R = self.local_network.run_value(sess, self.state, self.allocation)

        rewards.reverse()
        values.reverse()
        # compute and accmulate gradients
        # FROM LATS TO FIRST
        batch_td = []
        batch_R = []
        for(ri, Vi) in zip(rewards, values):
            # args.gamma is the discount
            # the trade period is very short, the discount should be a really small value
            R = (1+ri) * args.gamma * R
            td = R - Vi
            batch_td.append(td)
            batch_R.append(R)

        batch_td.reverse()
        batch_R.reverse()
        batch_si = states
        batch_allo = allocations
        batch_a = actions
        # reverse back the values
        values.reverse()
        batch_vi = values

        feed_dict = {
            self.local_network.s: batch_si,
            self.local_network.allo: batch_allo,
            self.local_network.a: batch_a,
            self.local_network.td: batch_td,
            self.local_network.r: batch_R,
            self.local_network.gauss_sigma: args.gauss_sigma,
            self.local_network.c_in: self.local_network.state_init[0],
            self.local_network.h_in: self.local_network.state_init[1],
            }
        sess.run(self.apply_gradients, feed_dict = feed_dict)

        return self.local_t-previous_t





















