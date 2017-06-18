#



import tensorflow as tf


def inner():
    with tf.variable_scope('vs2'):
        b = tf.get_variable('b',[1])
        return b


with tf.variable_scope('vs1'):
    a = tf.random_uniform([5,5,5],minval=-1,maxval=1)
    sess = tf.Session()
    na = sess.run(a[:,:,2:5])
    # print(na.shape)
    # print(na)
    c = tf.reduce_sum(na,axis=2)
    print(sess.run(c))