# -*- coding: utf-8 -*-
from network import LSTM_ACNetwork
from config import *
import gym

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
        self.local_network.prepare_loss(args.entropy_beta)

        self.opt = optimizer
        local_gradients = self.opt.compute_gradients(self.local_network.total_loss, self.local_network.vars)
        self.gradients = [(tf.clip_by_norm(local_gradients[i][0], args.grad_norm_clip), global_network.vars[i]) for i in range(len(local_gradients))]
        self.apply_gradients = self.opt.apply_gradients(self.gradients)

        self.sync = self.local_network.sync_from(global_network)

        self.env = gym.make(args.game)
        self.terminal = True
        self.episode_reward = 0

        self.local_t = 0

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p = pi_values)

    def process(self, sess, global_t):
        previous_t = self.local_t

        states = []
        actions = []
        rewards = []
        values = []

        sess.run(self.sync)

        if self.terminal:
            self.state = self.env.reset()
            self.local_network.reset_state_value()

        for i in range(args.local_t_max):
            pi_, value_ = self.local_network.run_policy_and_value(sess, self.state)
            action_index = self.choose_action(pi_)
            action = args.action_map[action_index]

            states.append(self.state)
            actions.append(action_index)
            values.append(value_)

            self.state, reward, self.terminal_end, _ = self.env.step(action)
            self.episode_reward += reward
            rewards.append( np.clip(reward, -1, 1) )
            self.local_t += 1

            if self.terminal_end:
                print("score={}".format(self.episode_reward))
                self.episode_reward = 0
                self.state = self.env.reset()
                break

        if self.terminal_end:
            R = 0.0
        else:
            R = self.local_network.run_value(sess, self.state)

        actions.reverse()
        states.reverse()
        rewards.reverse()
        values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
            R = ri + args.gamma * R
            td = R - Vi
            a = np.zeros([args.action_size])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        batch_si.reverse()
        batch_a.reverse()
        batch_td.reverse()
        batch_R.reverse()
        feed_dict = {
            self.local_network.s: batch_si,
            self.local_network.a: batch_a,
            self.local_network.td: batch_td,
            self.local_network.r: batch_R,
            self.local_network.c_in: self.local_network.state_init[0],
            self.local_network.h_in: self.local_network.state_init[1],
            }
        sess.run(self.apply_gradients, feed_dict = feed_dict)

        return self.local_t-previous_t





















