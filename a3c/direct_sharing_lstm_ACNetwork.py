from network import *
from direct_allocation_RNNCell import direct_allocation_RNNCell

class Sharing_LSTM_ACNetwork(BasicACNetwork):
    def __init__(self,
                 action_size,
                 thread_index):
        BasicACNetwork.__init__(self, action_size, thread_index)
        with tf.variable_scope(self._name) as scope:
            # s shape [steps, len]
            self.s = tf.placeholder(tf.float32, [None, args.input_size])
            assert args.info_num * args.asset_num == args.input_size
            # lstm1_in shape [batch, steps, len], where batch=1, len=info_num*asset_num
            lstm1_in = tf.expand_dims(self.s, [0])
            # lstm1_in_split[i] in shape [batch, steps, info_num], where batch=1
            lstm1_in_split = tf.split(lstm1_in, args.asset_num, axis=2)
            with tf.variable_scope('LSTM1') as vs:
                lstm1_cell = rnn.BasicLSTMCell(num_units=args.lstm1_unit, state_is_tuple=True)
                # asset_num inputs using the same variable
                # but the state are independent
                # init state, np array
                self.lstm1_c_init = np.zeros((args.asset_num, lstm1_cell.state_size.c), np.float32)
                self.lstm1_h_init = np.zeros((args.asset_num, lstm1_cell.state_size.h), np.float32)
                # state for the LSTM network
                # they are placeholder because we need to define the inference later
                self.lstm1_c_in = tf.placeholder(tf.float32, [args.asset_num, lstm1_cell.state_size.c])
                self.lstm1_h_in = tf.placeholder(tf.float32, [args.asset_num, lstm1_cell.state_size.h])
                # a list of tensors
                # correspond to different asset
                lstm1_c_in_split = tf.split(self.lstm1_c_in,args.asset_num,axis=0)
                lstm1_h_in_split = tf.split(self.lstm1_h_in,args.asset_num,axis=0)
                lstm1_c_split = []
                lstm1_h_split = []
                lstm1_output_split = []

                for iAsset in range(args.asset_num):
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
                        inputs = lstm1_in_split[iAsset],
                        initial_state = statei_tuple,
                        time_major = False
                    )
                    # the index of list is the num of asset
                    # lstm1_output_split shape [asset_num, lstm1_outputi]
                    # lstm1_outputi in shape [1, steps, lstm1_unit]
                    lstm1_c_split.append(lstm1_statetuplei[0])
                    lstm1_h_split.append(lstm1_statetuplei[1])
                    lstm1_output_split.append(lstm1_outputi)
                # concat states
                # self.lstm1_c and self.lstm1_h are operators, not value
                self.lstm1_c = tf.concat(lstm1_c_split,0)
                self.lstm1_h = tf.concat(lstm1_h_split,0)

                # lstm1_outpust in shape [1, steps, lstm1_unit * asset_num]
                lstm1_outputs = tf.concat(lstm1_output_split,2)
                # lstm1_outputs in shape [steps, lstm1_unit * asset_num]
                lstm1_outputs = tf.reshape(lstm1_outputs, [-1, args.lstm1_unit * args.asset_num])

            with tf.variable_scope('Allocation_RNN') as vs:
                allo_rnn = direct_allocation_RNNCell(args.lstm1_unit * args.asset_num)

                self.allo_init = tf.concat(
                    tf.constant(0,dtype = tf.float32, shape = [1,args.asset_num]),
                    tf.constant(0,dtype = tf.float32, shape = [1,1]),
                    axis = 1
                )
                self.first_price = tf.placeholder(tf.float32, [1,args.asset_num])
                self.next_price = tf.placeholder(tf.float32, [None, args.asset_num])
                allo_rnn_input = tf.concat([self.next_price, lstm1_outputs], axis = 1)
                allo_rnn_state_init = tf.concat([self.first_price, self.allo_init])

                self.allo_rnn_output,self.allo_rnn_state = tf.nn.dynamic_rnn(
                    allo_rnn,
                    inputs = allo_rnn_input,
                    initial_state = allo_rnn_state_init,
                    time_major= False
                )
                self.logrewards, self.actions = tf.split(self.allo_rnn_output, [1,args.asset_num + 1], axis = 1)

            self.totallogreward = tf.reduce_sum(self.logrewards)

            self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
            self.reset_state_value()

    def reset_state_value(self):
        # self.lstm1_c_value and self.lstm1_h_value
        # are used to feed lstm state
        self.lstm1_c_value = self.lstm1_c_init
        self.lstm1_h_value = self.lstm1_h_init

    def run_opt(self, sess, s_t, allocation):
        # forward propagation, use the state in last step
        feed_dict = {
            self.s : [s_t],
            self.allo : [allocation],
            self.lstm1_c_in: self.lstm1_c_value,
            self.lstm1_h_in: self.lstm1_h_value
        }
        if args.dropout:
            feed_dict[self.keep_prob] = 1.0

        gauss_mean_value, v_value, self.lstm1_c_value, self.lstm1_h_value = sess.run(
            [self.gauss_mean, self.v, self.lstm1_c, self.lstm1_h],
            feed_dict=feed_dict)
        return (gauss_mean_value[0], v_value[0])

    def short_sight_run_policy_and_value(self, sess, s, allocations):
        # forward propagation, s is the series of the most recent steps
        # allocations are the most recent allocations
        # but the output of this function only depends on the last allocation
        feed_dict = {
            self.s : s,
            self.allo : allocations,
            self.lstm1_c_in: self.lstm1_c_init,
            self.lstm1_h_in: self.lstm1_h_init
        }
        if args.dropout:
            feed_dict[self.keep_prob] = 1.0
        gauss_mean_value, v_value = sess.run([self.gauss_mean, self.v],feed_dict=feed_dict)
        # gauss_mean_value and v_value are lists contains all steps' results, so return the lastest one
        return (gauss_mean_value[-1], v_value[-1])

    def run_value(self, sess, s_t, allocation):
        # when calculate the value of a certain state
        # this funcation won't update the state
        feed_dict = {
            self.s : [s_t],
            self.allo : [allocation],
            self.lstm1_c_in : self.lstm1_c_value,
            self.lstm1_h_in : self.lstm1_h_value
        }
        if args.dropout:
            feed_dict[self.keep_prob] = 1.0

        v_value, _, _ = sess.run([self.v, self.lstm1_c, self.lstm1_h], feed_dict=feed_dict)
        return v_value[0]
