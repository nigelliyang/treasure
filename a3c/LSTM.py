import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

lr = 1e-3
batch_size_train = 128
hidden_num = 256
layer_num = 2
class_num = 10

def inference(images, labels, batch_size, keep_prob):
    with tf.name_scope('LSTM'):
        lstm_cell = rnn.BasicLSTMCell(num_units=hidden_num, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        mlstm_cell = rnn.MultiRNNCell([lstm_cell]*layer_num, state_is_tuple=True)
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=images, initial_state=init_state, time_major=False)
        h_state = outputs[:, -1, :]

    with tf.name_scope('CF'):
        w = tf.Variable(tf.truncated_normal([hidden_num, class_num], stddev=0.1), dtype=tf.float32)
        b = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
        y = tf.nn.softmax(tf.matmul(h_state, w)+b)

    with tf.name_scope('Eval'):
        cross_entropy = -tf.reduce_mean(labels*tf.log(y))
        correct = tf.equal(tf.argmax(y,1), tf.argmax(labels,1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        result = tf.argmax(y,1)

    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    return train_op, accuracy, correct, cross_entropy

def run_training():
    with tf.Graph().as_default():
        raw_images = tf.placeholder("float", shape=[None, 28*28])
        images = tf.reshape(raw_images, [-1,28,28])
        labels = tf.placeholder("float", shape=[None, class_num])
        batch_size = tf.placeholder(tf.int32)
        keep_prob = tf.placeholder(tf.float32)
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_op, accuracy, result, loss_op = inference(images, labels, batch_size, keep_prob)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            step = 1
            while step<1000:
                batch = mnist.train.next_batch(batch_size_train)
                _, loss_value = sess.run([train_op, loss_op], feed_dict={raw_images: batch[0], labels: batch[1], batch_size: batch_size_train, keep_prob: 1.0})
                if step%200 == 0:
                    batch_size_test = len(mnist.test.labels)
                    accuracy_value = sess.run(accuracy, feed_dict={raw_images:mnist.test.images, labels:mnist.test.labels, batch_size: batch_size_test, keep_prob:1.0})
                    print("step %d  test accuracy is:%.3f" % (step, accuracy_value))

                step += 1

if __name__ == '__main__':
    run_training()