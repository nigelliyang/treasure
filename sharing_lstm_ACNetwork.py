from network import *

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

            with tf.variable_scope('Allocation_state') as vs:
                self.allo = tf.placeholder(tf.float32, [None, self._action_size])
                # all_state in shape [steps, lstm1_unit * asset_num + action_size]
                all_state = tf.concat([lstm1_outputs, self.allo], axis=1)
                W_fc0, b_fc0 = self._fc_variable([args.lstm1_unit * args.asset_num +self._action_size, args.state_feature_num])
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
        # self.lstm1_c_value and self.lstm1_h_value
        # are used to feed lstm state
        self.lstm1_c_value = self.lstm1_c_init
        self.lstm1_h_value = self.lstm1_h_init

    def run_policy_and_value(self, sess, s_t, allocation):
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