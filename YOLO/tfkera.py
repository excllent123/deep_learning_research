import tensorflow as tf
sess = tf.Session()


from keras import backend as K
K.set_session(sess)

import numpy as np

 # import the necessary packages

from keras.layers import Input

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential , Model

from tf_yolo import YoloDetector
from yolo_cnn import YoloNetwork
#

#img = tf.placeholder(tf.float32, shape=(None, 784))#

#from keras.layers import Dense#

## Keras layers can be called on TensorFlow tensors:
#x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
#x = Dense(128, activation='relu')(x)
#preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation#
#

#labels = tf.placeholder(tf.float32, shape=(None, 10))#

#from keras.objectives import categorical_crossentropy#

#loss = tf.reduce_mean(categorical_crossentropy(labels, preds))#

#from tensorflow.examples.tutorials.mnist import input_data
#mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)#

#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#with sess.as_default():
#    for i in range(100):
#        batch = mnist_data.train.next_batch(50)
#        train_step.run(feed_dict={img: batch[0],
#                                  labels: batch[1]})#
#
#

#from keras.metrics import categorical_accuracy as accuracy#

#acc_value = accuracy(labels, preds)
#with sess.as_default():
#    print acc_value.eval(feed_dict={img: mnist_data.test.images,})


W = 448
H = 448
S = 7
B = 2
C = 20


input_X = tf.placeholder(tf.float32, shape=(None, W,H,3))
true_y = tf.placeholder(tf.float32, shape = (None , S*S*(B*5+C)))


yolooo = YoloNetwork()
A = YoloDetector()

model = yolooo.yolo_small()
# this works! 

pred_y = model(input_X)
print true_y[0,:].get_shape()

#tf.py_func(func, inp, Tout, name=None)

# Wraps a python function and uses it as a tensorflow op.

# Given a python function func, which takes numpy arrays as its inputs a
# nd returns numpy arrays as its outputs. E.g.,



loss = A.loss(true_y[0,:], pred_y[0,:] ) # tf-stle slice must have same rank

#loss = tf.py_func(A.loss, true_y[0,:], pred_y[0,:])

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess.run(Output, feed_dict = {x: images_feed, y_:labels_feed})

#with tf.Session():
#  a = tf.random_uniform((3, 3))
#  b = a.eval()  

  # Runs to get the output of 'a' and converts it to a numpy array
'''
K.set_learning_phase(0)

x = Convolution2D(16, 3, 3, border_mode='same' )(input_X)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' , strides=(2,2))(x)

x = Convolution2D(32,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' , strides=(2,2))(x)

x = Convolution2D(64,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
    strides=(2,2))(x)

x = Convolution2D(128,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
    strides=(2,2))(x)

x = Convolution2D(256,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
    strides=(2,2))(x)

x = Convolution2D(512,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
    strides=(2,2))(x)

x = Convolution2D(1024,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)

x = Convolution2D(1024,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)

x = Convolution2D(1024,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)

# x = Flatten()(x) # Cause some problem
# replace by flowing 
# https://github.com/fchollet/keras/issues/4207
x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])

x = Dense(256, activation='linear')(x)

x = Dense(4096)(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.5)(x)
pred = Dense(S*S*(B*5+C), activation='linear')(x)

model = Model(input=Input(shape=input_X.get_shape()), output=Input(shape=pred.get_shape()))
model.summary()
'''

        