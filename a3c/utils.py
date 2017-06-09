import tensorflow as tf
import math

def activation_tensor_summary(x):
    tf.summary.histogram(x.op.name+'/activations', x)
    tf.summary.scalar(x.op.name+'/sparsity', tf.nn.zero_fraction(x))

def prepare_dir(directory):
    if tf.gfile.Exists(directory):
        tf.gfile.DeleteRecursively(directory)
    tf.gfile.MakeDirs(directory)

def lr_anneal(ini_lr, end_lr, steps, global_step, type = 'linear'):
    if type == 'linear':
        lr = ini_lr-(ini_lr-end_lr)/steps*global_step
        return lr
    else:
        raise KeyError('learning anneal type %s not supported' %type)

def SCOT(f, len_a, beta):
    '''
    This function solve the following optimal problem using shrink
    min(alpha) ||Df-alpha||2 + beta*||alpha||1
    s.t. sum(Dj)>=1
    where Dj is the row vector of D
    the optimal solution alpha is a sparse representation of f
    the length of a, len_a, is usually longer than len_f
    When D is given, the optimal solution alpha is fixed.
    alpha is the input of A3Cnetwork, and the variable D should be trained together with A3C network

    Input: f->shape [batch, len_f]
    '''
    D = tf.Variable(tf.truncated_normal([len_f, len_a]), stddev =
            1.0/math.sqrt(float(len_f)), trainable=True)
    # garantee every row of D has a l2 norm less than 1
    D = D/tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(D), axis=1)), axis=1)
    # positive shape [Batch, len_a]
    positive = tf.matmul(f, D)-beta/2
    negative = positive+beta
    full_zeros = tf.zeros(positive.shape)
    pos_index = tf.cast(tf.greater(positive, full_zeros), tf.float32)
    neg_index = tf.cast(tf.less(negative, full_zeros), tf.float32)
    # alpha shape [Batch, len_a]
    alpha = tf.add(full_zeros, tf.multiply(pos_index, positive))
    alpha = tf.add(alpha, tf.multiply(neg_index, negative))

    return alpha, D

class invest_monitor(object):
    def __init__(self, max_len):
        self._observation = []
        self._max_len = max_len
    def insert(self, observation):
        self._observation.append(observation)
        if len(self._observation) >= self._max_len:
            print("resent %d investments return: %.3f" %(len(self._observation), sum(self._observation)/len(self._observation)))
            self._observation = []

# summary and ckpt saver
# merged = tf.summary.merge_all()
# summaryWriter = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
# summary = sess.run(merged)
# summaryWriter.add_summary(summary, step)
# saver = tf.train.Saver(tf.all_variables())
# saver.save(sess, checkpoint_dir+'/MLP', global_step=step)


# def SCOT(f, len_a, beta):
#     '''
#     based on Sparse Coding-Inspired Optimal Trading System for HFT Industry
#     Digital Object Identifier 10.1109/TII.2015.2404299
#     Solve the strict of eq.4
#     alpha = invDTD*(DTf-lambda) if > 0
#             invDTD*(DTf+lambda) if < 0
#             0                   otherwise
#     in this function, we assume D is indepent from f
#     '''
#     # f is a placeholder shape [Batch, len_f]
#     # change it to shape [len_f, Batch]
#     f = tf.transpose(f, [1,0])
#     len_f = f.shape.as_list()[0]
#     D = tf.Variable(tf.random_uniform([len_f, len_a]), trainable=True)
#     # garantee every row of D has a l2 norm less than 1
#     D = D/tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(D), axis=1)), axis=1)
#     DT = tf.transpose(D, [1,0])
#     DTD = tf.matmul(DT, D)
#     invDTD = tf.matrix_inverse(DTD)
#     # beta is the multiplier on l1 loss
#     # positive and negative have the same shape [len_a, batch]
#     positive = tf.matmul(invDTD, tf.matmul(DT, f)-beta)
#     negative = tf.matmul(invDTD, tf.matmul(DT, f)+beta)
#     full_zeros = tf.zeros(positive.shape)
#     pos_index = tf.cast(tf.greater(positive, full_zeros), tf.float32)
#     neg_index = tf.cast(tf.less(negative, full_zeros), tf.float32)
#     alpha = tf.add(full_zeros, tf.multiply(pos_index, positive))
#     alpha = tf.add(alpha, tf.multiply(neg_index, negative))
#     # now alpha shape [len_a, batch]
#     # change it to [batch, len_a]
#     alpha = tf.transpose(alpha, [1,0])
#     # the only variable in SCOT is D
#     return alpha, D

# def SCOT(f, len_a, beta):
#     '''
#     based on Sparse Coding-Inspired Optimal Trading System for HFT Industry
#     Digital Object Identifier 10.1109/TII.2015.2404299
#     Solve the strict of eq.4
#     alpha = invDTD*(DTf-lambda) if > 0
#             invDTD*(DTf+lambda) if < 0
#             0                   otherwise
#     in this function, we assume the relationship between D and f can be presented by a FC layer
#     '''
#     # f is a placeholder shape [Batch, len_f]
#     len_f = f.shape.as_list()[1]
#     f_to_D_w = tf.Variable(tf.truncated_normal([len_f, len_f, len_a]), stddev =
#             1.0/math.sqrt(float(len_f*len_a)), trainable=True)
#     f_to_D_b = tf.Variable(tf.zeros([len_f, len_a]), trainable=True)
#     # D shape [Batch, len_f, len_a]
#     D = tf.matmul(f, f_to_D_w)+f_to_D_b
#     # garantee every row of D has a l2 norm less than 1
#     D = D/tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(D), axis=2)), axis=2)
#     DT = tf.transpose(D, [0,2,1])
#     # the matmul will treat the first dim as batch
#     # and run matmul at each 2-D matrix
#     DTD = tf.matmul(DT, D)
#     # matrix_inverse also automatically
#     invDTD = tf.matrix_inverse(DTD)
#     # beta is the multiplier on l1 loss
#     # positive and negative have the same shape [len_a, batch]
#     positive = tf.matmul(invDTD, tf.matmul(DT, f)-beta)
#     negative = tf.matmul(invDTD, tf.matmul(DT, f)+beta)
#     full_zeros = tf.zeros(positive.shape)
#     pos_index = tf.cast(tf.greater(positive, full_zeros), tf.float32)
#     neg_index = tf.cast(tf.less(negative, full_zeros), tf.float32)
#     alpha = tf.add(full_zeros, tf.multiply(pos_index, positive))
#     alpha = tf.add(alpha, tf.multiply(neg_index, negative))
#     # now alpha shape [len_a, batch]
#     # change it to [batch, len_a]
#     alpha = tf.transpose(alpha, [1,0])
#     # the only variable in SCOT is D
#     return alpha, D

# # alpha, D = SCOT(a, 5, 1)