from network import LSTM_ACNetwork
import utils
from config import *
from environment import *
from futuresData import *

import tensorflow as tf
import numpy as np
import random

local_network = LSTM_ACNetwork(args.action_size, 0)
local_network.prepare_loss(args.entropy_beta, args.risk_beta)
data = futuresData()
data.loadData_moreday0607()
env = futuresGame(data)
s = env.reset()
allo = np.random.random(args.action_size)
a = np.random.random(args.action_size)

grad_applier = tf.train.RMSPropOptimizer(
        learning_rate = args.learning_rate,
        decay = args.rmsp_alpha,
        momentum = 0.0,
        epsilon = args.rmsp_epsilon)
# 0 ['ACnet_0/LSTM/rnn/basic_lstm_cell/weights:0',
# 1  'ACnet_0/LSTM/rnn/basic_lstm_cell/biases:0',
# 2  'ACnet_0/Allocation_state/Variable:0',
# 3  'ACnet_0/Allocation_state/Variable_1:0',
# 4  'ACnet_0/FC_policy/Variable:0',
# 5  'ACnet_0/FC_policy/Variable_1:0',
# 6  'ACnet_0/FC_value/Variable:0',
# 7  'ACnet_0/FC_value/Variable_1:0']
var_index = 2

grad_op = grad_applier.compute_gradients(local_network.total_loss, [local_network.vars[var_index]])

feed_dict = {
    local_network.s: [s],
    local_network.allo: [allo],
    local_network.a: [a],
    local_network.td: [0.2],
    local_network.r: [1.2],
    local_network.gauss_sigma: args.gauss_sigma,
    }

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    grad = sess.run(local_network.temp, feed_dict=feed_dict)
print(local_network.vars[var_index].name)
print(local_network.vars[var_index].get_shape())
print(grad)