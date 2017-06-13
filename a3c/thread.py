# -*- coding: utf-8 -*-
from network import LSTM_ACNetwork
from independentlstm import Independent_LSTM_ACNetwork
import utils
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
                 max_global_steps,
                 use_test_data=False):
        self.thread_index = thread_index
        self.max_global_steps = max_global_steps
        if args.share_variable:
            self.local_network = Independent_LSTM_ACNetwork(args.action_size, self.thread_index)
        else:
            self.local_network = LSTM_ACNetwork(args.action_size, self.thread_index)
        #self.local_network = LSTM_ACNetwork(args.action_size, self.thread_index)
        self.local_network.prepare_loss(args.entropy_beta, args.risk_beta)

        self.opt = optimizer
        local_gradients = self.opt.compute_gradients(self.local_network.total_loss, self.local_network.vars)
        # self.gradients = local_gradients
        self.gradients = [(tf.clip_by_norm(local_gradients[i][0], args.grad_norm_clip), global_network.vars[i]) for i in range(len(local_gradients))]
        self.apply_gradients = self.opt.apply_gradients(self.gradients)

        self.sync = self.local_network.sync_from(global_network)

        data = futuresData()
        data.loadData_moreday0607(use_test_data)
        self.env = futuresGame(data)
        self.terminal = True
        self.episode_reward = 1.0 # use multiplication model in futures game
        self.init_allocation = np.zeros(args.action_size)
        self.init_allocation[-1] = 1
        self.allocation = self.init_allocation

        self.local_t = 0

        self.monitor = utils.invest_monitor(max_len = 10)

    def choose_action(self, gauss_mean, gauss_sigma, determinate_action=False):
        '''
        :param guass_mean:  array [] ndarray
        :param guass_sigma: matrix like [[],[],[]] ndarray
        :return: ndarray
        '''

        # if use determinate policy, return the mean value directly
        if determinate_action:
            return np.append(gauss_mean, 1-np.sum(gauss_mean))

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
        print('thread %d bad luck for choosing %d times not find a good assignment, so return the guass_mean' % (self.thread_index, max_times))
        print('gaussian mean', gauss_mean)
        return np.append(gauss_mean, 1-np.sum(gauss_mean))

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
                self.monitor.insert(self.episode_reward)
                # print("action = ", action, " value = ", value_)
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

        feed_dict = {}
        if type(self.local_network) == LSTM_ACNetwork:
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
        elif type(self.local_network) == Independent_LSTM_ACNetwork:
            feed_dict = {
                self.local_network.s: batch_si,
                self.local_network.allo: batch_allo,
                self.local_network.a: batch_a,
                self.local_network.td: batch_td,
                self.local_network.r: batch_R,
                self.local_network.gauss_sigma: args.gauss_sigma,
                self.local_network.lstm1_c_in: self.local_network.lstm1_c_init,
                self.local_network.lstm1_h_in: self.local_network.lstm1_h_init,
                }
        else:
            print ('Error:Unknown Network Type!')
        if args.dropout:
            feed_dict[self.local_network.keep_prob] = 0.5

        sess.run(self.apply_gradients, feed_dict = feed_dict)
        # print("gradient", sess.run(self.gradients,feed_dict = feed_dict))

        return self.local_t-previous_t

class TestThread(TrainingThread):
    """docstring for testThread"""
    def __init__(self,
                 thread_index,
                 global_network,
                 optimizer,
                 max_global_steps,
                 use_test_data=False):
        TrainingThread.__init__(self, thread_index, global_network, optimizer, max_global_steps, use_test_data)

    def lazy_choose_action(self):
        # return uniform asset distribution
        action = np.ones([args.action_size])
        action = action/np.sum(action)
        return action

    def determinate_test(self, sess, lazy = False):
        # random = False -> use the determinate_action
        # random = True -> use the totally random action, not the Gaussian distribution
        sess.run(self.sync)
        self.state = self.env.reset()
        self.local_network.reset_state_value()
        self.allocation = self.init_allocation
        self.terminal = False
        episode_reward = 1
        log_count = 0
        records = []
        while not self.terminal:
            gauss_mean, _ = self.local_network.run_policy_and_value(sess, self.state, self.allocation)
            if lazy:
                action = self.lazy_choose_action()
            else:
                action = self.choose_action(gauss_mean, args.gauss_sigma, determinate_action=True)
            if log_count%10==0:
                # print("determinate test", gauss_mean)
                log_count+=1
            # reward is the neat return rate of capital, like 0.03
            self.state, self.allocation, reward, self.terminal, _ = self.env.step(action)
            episode_reward *= (1.0+reward)
            # recording
            # the first action_size elements are action
            # -1 element is the net return
            # -2 element is the leverage level
            leverage = np.sum(np.abs(action))
            record = np.append(np.append(action, leverage), reward)
            records.append(record)
        # record in shape [steps, action_size+2]
        # invest_monitor._ovservation in shape [test_num, steps, action_size+2]
        self.monitor.insert(records)
        return episode_reward

    def short_sight_test(self, sess, short_sight_step=args.short_sight_step):
        # In this test, the network ouly consider the most recent states
        sess.run(self.sync)
        self.init_state = self.env.reset()
        self.states = np.array([self.init_state for _ in range(short_sight_step)])
        self.allocations = np.array([self.init_allocation for _ in range(short_sight_step)])
        self.terminal = False
        episode_reward = 1
        log_count = 0
        records = []
        while not self.terminal:
            gauss_mean, _ = self.local_network.short_sight_run_policy_and_value(sess, self.states, self.allocations)
            action = self.choose_action(gauss_mean, args.gauss_sigma, determinate_action=True)
            # if log_count%10==0:
            #     print("determinate test", gauss_mean)
            #     log_count+=1
            # reward is the neat return rate of capital, like 0.03
            self.state, self.allocation, reward, self.terminal, _ = self.env.step(action)
            self.states = np.append(self.states[1:,:], self.state[np.newaxis,:], axis=0)
            self.allocations = np.append(self.allocations[1:,:], self.allocation[np.newaxis,:], axis=0)
            episode_reward *= (1.0+reward)
            # recording
            # the first action_size elements are action
            # -1 element is the net return
            # -2 element is the leverage level
            leverage = np.sum(np.abs(action))
            record = np.append(np.append(action, leverage), reward)
            records.append(record)
        # record in shape [steps, action_size+2]
        # invest_monitor._ovservation in shape [test_num, steps, action_size+2]
        self.monitor.insert(records)
        return episode_reward






















