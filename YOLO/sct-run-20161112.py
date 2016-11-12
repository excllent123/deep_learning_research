import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

# Reference : http://keras-cn.readthedocs.io/en/latest/blog/keras_and_tensorflow/
# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))