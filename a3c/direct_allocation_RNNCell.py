
from tensorflow.contrib import rnn
from a3c.network import *
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope

_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"
class direct_allocation_RNNCell(rnn.RNNCell):

    def __init__(self, lstminfo_dim, asset_num):
        self._info_dim = lstminfo_dim
        self._asset_num = asset_num
        self._price_dim = self._asset_num
        self._allo_dim = self._asset_num + 1

    @property
    def state_size(self):
        return self._price_dim + self._allo_dim

    @property
    def output_size(self):
        return 2 + self._asset_num

    # inputs as (next_price, this_infor) in shape (1, price_dim + info_dim)
    # state as (this_price, last_allocation) in shape(1, price_dim + allo_dim)
    def __call__(self, inputs, state, scope=None):
        price_dim = self._price_dim
        allo_dim = self._allo_dim
        info_dim = self._info_dim
        next_price, this_infor = tf.split(inputs,[price_dim, info_dim], axis = 1)
        this_price, last_allo = tf.split(state,[price_dim, allo_dim], axis = 1)

        with tf.variable_scope('FC0') as vs:
            input_fc0 = tf.concat([this_infor, last_allo], axis=1)
            output_dim_fc0 = args.alloRNN_fc0_output_dim
            W_fc0, b_fc0 = self._fc_variable([info_dim + allo_dim, output_dim_fc0])
            output_fc0 = tf.nn.relu(tf.matmul(input_fc0, W_fc0) + b_fc0)
            if args.dropout:
                keep_prob0 = tf.placeholder(tf.float32, [])
                output_fc0 = tf.nn.dropout(output_fc0, keep_prob0)

        with tf.variable_scope('FC1') as vs:
            W_fc1, b_fc1 = self._fc_variable([output_dim_fc0, allo_dim])
            output_fc1 = tf.nn.relu(tf.matmul(output_fc0, W_fc1) + b_fc1)
            if args.dropout:
                keep_prob1 = tf.placeholder(tf.float32, [])
                output_fc1 = tf.nn.dropout(output_fc1, keep_prob1)

        # Avoid the [0,0,...,0] condition leaded 0/0 error
        normal_item = tf.concat(
                    [tf.constant(0,dtype = tf.float32, shape = [1,self._asset_num]),
                    tf.constant(0.1,dtype = tf.float32, shape = [1,1])],
                    axis = 1
                )
        output_fc1 = output_fc1 + normal_item
        this_allo = tf.reduce_sum(output_fc1) * output_fc1

        newstate = tf.concat([next_price, this_allo], axis = 1)

        # deposit is the value of the cash  (1)
        deposit = tf.constant([[1]], dtype = tf.float32, shape = [1,1])
        # in shape (1, asset_num+1) while this_price in shape (1, asset_num)
        this_price_with_deposit = tf.concat([this_price,deposit], axis=1)
        next_price_with_deposit = tf.concat([next_price,deposit], axis=1)
        this_property = tf.reduce_sum(this_allo * this_price_with_deposit)
        next_property = tf.reduce_sum(this_allo * next_price_with_deposit)

        reward = this_property / next_property
        logreward = tf.log(reward)
        logreward = tf.expand_dims(tf.expand_dims(logreward, [0]),[0])
        # output is [[logreward, this_allocation[0], this_allocation[1], ..., this_allocation[asset_num]]
        # output in shape (1 , 2 + args.asset_num)
        output = tf.concat([logreward, this_allo], axis=1)

        return output, newstate

    def _fc_variable(self, weight_shape):
        with variable_scope.variable_scope('FC_var') as vs:
            d = 1.0 / np.sqrt(weight_shape[0])
            bias_shape = [weight_shape[1]]
            bias = variable_scope.get_variable(_BIAS_VARIABLE_NAME, shape = bias_shape,initializer = init_ops.constant_initializer(0, dtype=tf.float32))
            weight = variable_scope.get_variable(_WEIGHTS_VARIABLE_NAME, shape = weight_shape,initializer = init_ops.constant_initializer(0, dtype=tf.float32))
        return weight, bias
