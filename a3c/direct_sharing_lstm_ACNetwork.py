from network import *
import tensorflow as tf
from tensorflow.contrib import rnn
from direct_allocation_RNNCell import direct_allocation_RNNCell

class Direct_Sharing_LSTM_ACNetwork(BasicACNetwork):
    def __init__(self,
                 asset_num,
                 info_num,
                 thread_index):
        action_size = asset_num + 1
        BasicACNetwork.__init__(self, action_size, thread_index)
        with tf.variable_scope(self._name) as scope:
            # s shape [steps, len]
            self.s = tf.placeholder(tf.float32, [None, asset_num * info_num])
            # lstm1_in shape [batch, steps, len], where batch=1, len=info_num*asset_num
            lstm1_in = tf.expand_dims(self.s, [0])
            # lstm1_in_split[i] in shape [batch, steps, info_num], where batch=1
            self.lstm1_in_split = tf.split(lstm1_in, asset_num, axis=2)
            self.first_price = tf.placeholder(tf.float32, [1,asset_num])
            self.next_price = tf.placeholder(tf.float32, [None, asset_num])
            with tf.variable_scope('LSTM1') as vs:
                lstm1_cell = rnn.BasicLSTMCell(num_units=args.lstm1_unit, state_is_tuple=True)
                # asset_num inputs using the same variable
                # but the state are independent
                # init state, np array
                #self.lstm1_c_init = np.zeros((asset_num, lstm1_cell.state_size.c), np.float32)
                #self.lstm1_h_init = np.zeros((asset_num, lstm1_cell.state_size.h), np.float32)
                # state for the LSTM network
                # they are placeholder because we need to define the inference later
                #self.lstm1_c_in = tf.placeholder(tf.float32, [args.asset_num, lstm1_cell.state_size.c])
                self.lstm1_c_in = tf.constant(0, dtype=tf.float32, shape=[asset_num, lstm1_cell.state_size.c])
                #self.lstm1_h_in = tf.placeholder(tf.float32, [args.asset_num, lstm1_cell.state_size.h])
                self.lstm1_h_in = tf.constant(0, dtype=tf.float32, shape=[asset_num, lstm1_cell.state_size.h])
                # a list of tensors
                # correspond to different asset
                lstm1_c_in_split = tf.split(self.lstm1_c_in,asset_num,axis=0)
                lstm1_h_in_split = tf.split(self.lstm1_h_in,asset_num,axis=0)
                self.lstm1_c_split = []
                self.lstm1_h_split = []
                self.lstm1_output_split = []

                for iAsset in range(asset_num):
                    # every time call dynamic_rnn, will redefine the variables, so use reuse_variables to implement share variables
                    if iAsset > 0:
                        tf.get_variable_scope().reuse_variables()
                    c_in_i = lstm1_c_in_split[iAsset]
                    h_in_i = lstm1_h_in_split[iAsset]
                    # concat c and h into a statetuple
                    statei_tuple = rnn.LSTMStateTuple(c_in_i ,h_in_i)
                    # lstm1_outputi in shape [1, steps, lstm1_unit]
                    # lstm1_statetuplei in shape [(lstm1_c, lstm1_h)]
                    lstm1_outputi, lstm1_statetuplei = tf.nn.dynamic_rnn(
                        lstm1_cell,
                        inputs = self.lstm1_in_split[iAsset],
                        initial_state = statei_tuple,
                        time_major = False
                    )
                    # the index of list is the num of asset
                    # lstm1_output_split shape [asset_num, lstm1_outputi]
                    # lstm1_outputi in shape [1, steps, lstm1_unit]
                    self.lstm1_c_split.append(lstm1_statetuplei[0])
                    self.lstm1_h_split.append(lstm1_statetuplei[1])
                    self.lstm1_output_split.append(lstm1_outputi)
                # concat states
                # self.lstm1_c and self.lstm1_h are operators, not value
                self.lstm1_c = tf.concat(self.lstm1_c_split,0)
                self.lstm1_h = tf.concat(self.lstm1_h_split,0)

                # lstm1_outpust in shape [1, steps, lstm1_unit * asset_num]
                self.lstm1_outputs = tf.concat(self.lstm1_output_split,2)
                # lstm1_outputs in shape [steps, lstm1_unit * asset_num]
                self.lstm1_outputs = tf.reshape(self.lstm1_outputs, [-1, args.lstm1_unit * asset_num])

            with tf.variable_scope('Allocation_RNN') as vs:
                allo_rnn = direct_allocation_RNNCell(args.lstm1_unit * asset_num, asset_num)
                
                self.allo_init = tf.concat(
                    [tf.constant(0,dtype = tf.float32, shape = [1,asset_num]),
                    tf.constant(1,dtype = tf.float32, shape = [1,1])],
                    axis = 1
                )
                allo_rnn_state_init = tf.concat([self.first_price, self.allo_init], axis=1)
                allo_rnn_input = tf.concat([self.next_price, self.lstm1_outputs], axis = 1)
                allo_rnn_input = tf.expand_dims(allo_rnn_input, [0])

                self.allo_rnn_output,self.allo_rnn_state = tf.nn.dynamic_rnn(
                    allo_rnn,
                    inputs = allo_rnn_input,
                    initial_state = allo_rnn_state_init,
                    time_major= False
                )
                self.logrewards, self.actions = tf.split(self.allo_rnn_output, [1,asset_num + 1], axis = 2)

            self.totallogreward = tf.reduce_sum(self.logrewards)
            self.totalreward = tf.exp(self.totallogreward)

            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
            #self.reset_state_value()
