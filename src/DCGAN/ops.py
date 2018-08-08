import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from utils import *


def linear():
    return z_, h0_w, h0b


def linear(input_, output_size, scope=None, 
    stddev=0.02, bias_start=0.0, with_w=False):
    '''
    Description: 
    - This function provide a random linear 

    Args:
    - 
    '''

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def generator(self, z):
    self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4,
                                           'g_h0_lin', with_w=True)

    self.h0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
    h0 = tf.nn.relu(self.g_bn0(self.h0))

    self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,
        [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
    h1 = tf.nn.relu(self.g_bn1(self.h1))

    h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
        [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
    h2 = tf.nn.relu(self.g_bn2(h2))

    h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
        [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
    h3 = tf.nn.relu(self.g_bn3(h3))

    h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
        [self.batch_size, 64, 64, 3], name='g_h4', with_w=True)

    return tf.nn.tanh(h4)

