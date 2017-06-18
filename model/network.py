#/usr/bin/python
#-*- coding:utf-8 -*-

#defination of the network



import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import data_provider as dp

from config import args
import os

prefix_path = '../summary/1'
path_train = prefix_path + '/summary_train'
path_test = prefix_path + '/summary_test'
path_allocation = prefix_path + '/allocation'

def checkpath():
    path_list = [path_train,path_test,path_allocation]
    for path in path_list:
        os.makedirs(path)


class NetWork(object):
    def __init__(self):
        self.lstm_c_size = None
        self.lstm_h_size = None


    def get_init_lstm_state(self):
        return np.zeros((1,self.lstm_c_size)),np.zeros((1,self.lstm_h_size))

    def _get_fc_variable(self,weight_shape):
        d = 1.0 / np.sqrt(weight_shape[0])
        bias_shape = [weight_shape[1]]
        weight = tf.Variable(tf.random_uniform(shape=weight_shape,minval=-d,maxval=d,name='weights'))
        bias = tf.Variable(tf.random_uniform(shape=bias_shape,minval=-d,maxval=d,name='bias'))
        return weight,bias

    # the inference of the network
    def inference(self):

        self.input = tf.placeholder(tf.float32,shape=(None,args.input_size))
        # lstm_in shape (1,step, input_size)
        lstm_in = tf.expand_dims(self.input,[0])
        # each element in lstm_in shape is (1,step,infofield_num) where infofield_num = input_size / split_size
        lstm_in_split_list = tf.split(lstm_in,args.futures_num,axis=2)

        with tf.variable_scope('lstm') as vs_lstm:
            lstm_cell = rnn.BasicLSTMCell(args.lstm_num_units,state_is_tuple=True)
            self.lstm_c_size = lstm_cell.state_size.c
            self.lstm_h_size = lstm_cell.state_size.h
            self.lstm_c_in = tf.placeholder(tf.float32,shape=(1,self.lstm_c_size))
            self.lstm_h_in = tf.placeholder(tf.float32,shape=(1,self.lstm_h_size))
            lstm_state_in = rnn.LSTMStateTuple(self.lstm_c_in,self.lstm_h_in)

            lstm_out_list = []
            with tf.variable_scope('shard_lstm') as vs_shared_lstm:

                for i in range(args.futures_num):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    # lstm_out shape (1,step,num_units)
                    lstm_out,lstm_state_tuple = tf.nn.dynamic_rnn(lstm_cell,lstm_in_split_list[i],
                                      initial_state=lstm_state_in,
                                      time_major=False)
                    lstm_out_list.append(lstm_out)
                # lstm_out_concat shape is (step,split_size * num_units) where step is bachsize for next FC layer
                lstm_out_concat = tf.reshape(tf.concat(lstm_out_list,axis=2),shape=(-1,args.futures_num * args.lstm_num_units))

        with tf.variable_scope('FC') as vs_fc:

            w1,b1 = self._get_fc_variable(weight_shape=(args.futures_num * args.lstm_num_units,args.futures_num))
            f1 = tf.nn.relu(tf.matmul(lstm_out_concat,w1) + b1)
            tf.summary.histogram('w1',w1)
            tf.summary.histogram('b1',b1)
            tf.summary.histogram('f1',f1)
        return f1



    # define loss for the network
    def get_loss(self,logits):
        # logits and labels label is (batch,num_units)
        # 简单地认为第0天的信息包含（开盘，收盘等），根据第0天的信息在第0天结束的时候一次性买，且按照收盘价买
        # 收益按照一天一结算，第1天结束时按照第1天的收盘价卖出，然后再按照logits[1]给出第一天结束时的购买配比
        # 即一天末结束清空该天收益，并计算该天收益，同时购买下一天的配比 logist[0]收益由price[0]和price[1]共同给出
        soft_logits = tf.nn.softmax(logits)
        print(soft_logits.shape)
        self.price = tf.placeholder(dtype=tf.float32,shape=(None,args.futures_num))
        # assert soft_logits.shape[0].value == self.price.shape[0].value + 1
        step_property = tf.reduce_sum((soft_logits * self.price[1:,:]) / self.price[0:-1,:],axis=1)
        episode_property = tf.reduce_prod(step_property,axis=0)
        tf.summary.scalar('reward',episode_property)
        return -episode_property,soft_logits




    def train(self):

        with tf.variable_scope('global_scope') as scope:

            logits = self.inference()
            loss,allocation = self.get_loss(logits)
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope.name)
            opt = tf.train.AdamOptimizer(learning_rate=0.1)
            gradient_tuples = opt.compute_gradients(loss,train_vars)
            train_op = opt.apply_gradients(gradient_tuples)


        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(path_train)
        summary_writer_test = tf.summary.FileWriter(path_test)

        sess = tf.Session()
        c = dp.futuresData()
        c.loadData_moreday0607(False)
        c_test = dp.futuresData()
        c_test.loadData_moreday0607(True)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # start to train
        for i in range(args.batch_num):
            temp_input,temp_price = c.next_batch(args.batch_size)
            temp_lstm_c_in,temp_lstm_h_in = self.get_init_lstm_state()
            feed_dict = {
                self.input:temp_input,
                self.lstm_c_in:temp_lstm_c_in,
                self.lstm_h_in:temp_lstm_h_in,
                self.price:temp_price,
            }
            _,loss_value,summary_str,allo_ndarray_train = sess.run([train_op,loss,summary,allocation],feed_dict=feed_dict)
            summary_writer.add_summary(summary_str,i)
            summary_writer.flush()
            print('batch num is %d cur_loss is %f' % (i,loss_value))

            # check test
            if i % 100 ==0: #test
                temp_input_test, temp_price_test = c_test.get_all()
                temp_lstm_c_in, temp_lstm_h_in = self.get_init_lstm_state()
                feed_dict = {
                    self.input: temp_input_test,
                    self.lstm_c_in: temp_lstm_c_in,
                    self.lstm_h_in: temp_lstm_h_in,
                    self.price: temp_price_test,
                }
                _, loss_value, summary_str,allo_ndarray_test = sess.run([train_op, loss, summary,allocation], feed_dict=feed_dict)
                if i % 500 == 0:
                    summary_writer_test.add_summary(summary_str, i)
                    summary_writer_test.flush()
                    print('TEST: batch num is %d cur_loss is %f' % (i, loss_value))
                    np.save(path_allocation+'/'+str(i)+'_train',allo_ndarray_train)
                    np.save(path_allocation + '/' + str(i) + '_test', allo_ndarray_test)





if __name__=='__main__':
    checkpath()
    net = NetWork()
    net.train()


























