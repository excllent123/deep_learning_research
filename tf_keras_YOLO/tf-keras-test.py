import tensorflow as tf
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import tensorflow.examples.tutorials.mnist.input_data as input_data


def MNIST(one_hot=True):
    """Returns the MNIST dataset.
    Returns
    -------
    mnist : DataSet
        DataSet object w/ convenienve props for accessing
        train/validation/test sets and batches.
    """
    return input_data.read_data_sets('MNIST_data/', one_hot=one_hot)

batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets

mnist = MNIST()


input_shape = (28,28,1)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# %% We add a new type of placeholder to denote when we are training.
# This will be used to change the way we compute the network during
# training/testing.
is_training = tf.placeholder(tf.bool, name='is_training')


x_tensor = tf.reshape(x, [-1, 28, 28, 1])

x_tensor = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],border_mode='valid',
    input_shape=input_shape)(x_tensor)
x_tensor = Activation('relu')(x_tensor)
x_tensor = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(x_tensor)
x_tensor = Activation('relu')(x_tensor)
x_tensor = MaxPooling2D(pool_size=pool_size)(x_tensor)
x_tensor = Dropout(0.25)(x_tensor)

x_tensor = tf.reshape(x_tensor, [-1, np.prod(x_tensor.get_shape()[1:].as_list())])
x_tensor = Dense(128)(x_tensor)
x_tensor = Activation('relu')(x_tensor)
x_tensor = Dropout(0.5)(x_tensor)
x_tensor = Dense(nb_classes)(x_tensor)
y_pred = Activation('softmax')(x_tensor)



# %% Define loss/eval/training functions
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))





# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
K.set_session(sess)
sess.run(tf.initialize_all_variables())


# %% We'll train in minibatches and report accuracy:
n_epochs = 10
batch_size = 100
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, losss = sess.run([train_step, cross_entropy], feed_dict={
            x: batch_xs, y: batch_ys, is_training: True, K.learning_phase(): 0})
    print(sess.run(accuracy,
                   feed_dict={
                       x: mnist.validation.images,
                       y: mnist.validation.labels,
                       is_training: False,
                       K.learning_phase(): 0
                   }))
    print losss