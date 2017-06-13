import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim
import numpy as np
from config import *

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
        entropy = -tf.reduce_sum(0.5*self._action_size*(1+np.log(2*np.pi))+0.5*tf.log(tf.matrix_determinant(self.gauss_sigma)))

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
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

class LSTM_ACNetwork(BasicACNetwork):
    def __init__(self,
                 action_size,
                 thread_index):
        BasicACNetwork.__init__(self, action_size, thread_index)
        with tf.variable_scope(self._name) as scope:
            # s shape [steps, len]
            self.s = tf.placeholder(tf.float32, [None, args.input_size])
            # lstm_in shape [batch, steps, len], where batch=1
            lstm_in = tf.expand_dims(self.s, [0])
            with tf.variable_scope('LSTM') as vs:
                lstm_cell = rnn.BasicLSTMCell(num_units=args.lstm_unit, state_is_tuple=True)
                c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
                h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
                self.state_init = [c_init, h_init]
                self.c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
                self.h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
                state = rnn.LSTMStateTuple(self.c_in,self.h_in)
                # lstm_outputs shape [1, steps, lstm_unit]
                # lstm_state shape [(lstm_c, lstm_h)]
                lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                      lstm_cell,
                      inputs = lstm_in,
                      initial_state = state,
                      time_major = False)
                # lstm_outputs shape [steps, lstm_unit]
                lstm_outputs = tf.reshape(lstm_outputs, [-1, args.lstm_unit])
            # self.c_in = tf.placeholder(tf.float32, [1, 10])
            # self.h_in = tf.placeholder(tf.float32, [1, 10])
            # self.state_init = 0
            # W_fc,b_fc = self._fc_variable([args.input_size, args.lstm_unit])
            # lstm_outputs = tf.matmul(self.s, W_fc)+b_fc

            with tf.variable_scope('Allocation_state') as vs:
                self.allo = tf.placeholder(tf.float32, [None, self._action_size])
                all_state = tf.concat([lstm_outputs, self.allo], axis=1)
                W_fc0, b_fc0 = self._fc_variable([args.lstm_unit+self._action_size, args.state_feature_num])
                self.state_feature = tf.nn.relu(tf.matmul(all_state, W_fc0) + b_fc0)
                if args.dropout:
                    self.keep_prob = tf.placeholder(tf.float32, [])
                    self.state_feature = tf.nn.dropout(self.state_feature, self.keep_prob)

            with tf.variable_scope('FC_policy') as vs:
                # the network will output the gaussian mean of size action_size-1
                W_fc1, b_fc1 = self._fc_variable([args.state_feature_num, action_size-1])
                # get the mean of gauss distribution
                # self.gauss_mean has shape [steps, action_size-1]
                self.gauss_mean = tf.matmul(self.state_feature, W_fc1) + b_fc1

                # Now calculate the pi for a given action
                # a has shape [steps, action_size]
                # given a, calculate the probability of a
                self.a = tf.placeholder(tf.float32, [None, self._action_size])
                # a_gauss_part has shape [steps, action_size-1]
                a_gauss_part = self.a[:,:-1]
                self.gauss_sigma = tf.placeholder("float",shape=[self._action_size-1,self._action_size-1])
                gauss_coefficient = 1.0 / (((2.0 * np.pi) ** (self._action_size / 2.0 )) * tf.sqrt(tf.reduce_sum(tf.square(self.gauss_sigma))))
                # gauss_bias shape [steps, action_size-1]
                gauss_bias = a_gauss_part - self.gauss_mean
                # shape [steps, action_size-1]
                temp = tf.matmul(gauss_bias, tf.matrix_inverse(self.gauss_sigma))
                # shape [steps]
                temp = tf.reduce_sum(tf.multiply(temp, gauss_bias),axis=1)
                # shape [steps]
                self.pi = gauss_coefficient * tf.exp(-0.5*temp)

            with tf.variable_scope('FC_value') as vs:
                W_fc2, b_fc2 = self._fc_variable([args.state_feature_num, 1])
                v_ = tf.matmul(self.state_feature,W_fc2) + b_fc2
                # self.v has shape [steps, ]
                self.v = tf.reshape(v_, [-1])

            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
            self.reset_state_value()

    def reset_state_value(self):
        self.state_value = self.state_init

    def run_policy_and_value(self, sess, s_t, allocation):
        # forward propagation, use the state in last step
        feed_dict = {
            self.s : [s_t],
            self.allo : [allocation],
            self.c_in : self.state_value[0],
            self.h_in : self.state_value[1]
        }
        if args.dropout:
            feed_dict[self.keep_prob] = 1.0

        gauss_mean_value, v_value, self.state_value = sess.run([self.gauss_mean, self.v, self.lstm_state],feed_dict = feed_dict)
        return (gauss_mean_value[0], v_value[0])
    def run_value(self, sess, s_t, allocation):
        # call this function when calculate the value of a certain state
        # this funcation won't update the state
        feed_dict = {
            self.s : [s_t],
            self.allo : [allocation],
            self.c_in : self.state_value[0],
            self.h_in : self.state_value[1]
        }
        if args.dropout:
            feed_dict[self.keep_prob] = 1.0

        v_value, _ = sess.run([self.v, self.lstm_state],feed_dict = feed_dict)
        return v_value[0]





















