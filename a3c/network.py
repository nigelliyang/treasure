import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim
import numpy as np
from a3c.config import *

class BasicACNetwork(object):
    def __init__(self,
                 action_size,
                 thread_index):
        self._action_size = action_size
        self._thread_index = thread_index
        self._name = "ACnet_" + str(self._thread_index)

    def prepare_loss(self, entropy_beta, risk_beta):
        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder("float", [None])

        # avoid NaN with clipping when value in pi becomes zero
        log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

        # Gaussian distribution entropy
        # gauss_sigma is a constant now, entropy is not trainable
        # let gauss_sigma depends on state to train gauss_sigma
        # policy entropy
        # entropy = -tf.reduce_sum(0.5*self._action_size*(1+np.log(2*np.pi))+0.5*tf.log(tf.matrix_determinant(self.gauss_sigma)))

        # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
        # policy_loss = - tf.reduce_sum( log_pi * self.td + entropy * args.entropy_beta )
        policy_loss = - tf.reduce_sum(log_pi * self.td)

        # R (input for value)
        self.r = tf.placeholder("float", [None])

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
        value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

        # l1 loss for gauss_mean
        # self.gauss_mean shape [steps, action_size-1]
        action_mean = tf.concat([self.gauss_mean, 1-tf.reduce_sum(self.gauss_mean, axis=1, keep_dims=True)], axis=1)
        risk_loss = risk_beta * tf.reduce_sum(tf.abs(action_mean))

        # gradienet of policy and value are summed up
        # self.total_loss = policy_loss+value_loss
        self.total_loss = policy_loss + value_loss + risk_loss

    def sync_from(self, src_netowrk, name=None):
        '''
        return a list of ops
        run the list will sync self from src_network
        '''
        src_vars = src_netowrk.vars
        dst_vars = self.vars

        sync_ops = []

        with tf.name_scope(name, "GameACNetwork", []) as name:
            for(src_var, dst_var) in zip(src_vars, dst_vars):
                sync_op = tf.assign(dst_var, src_var)
                sync_ops.append(sync_op)

            return tf.group(*sync_ops, name=name)

    def _fc_variable(self, weight_shape):
        d = 1.0 / np.sqrt(weight_shape[0])
        bias_shape = [weight_shape[1]]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d), name='weights')
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d), name='bias')
        return weight, bias























