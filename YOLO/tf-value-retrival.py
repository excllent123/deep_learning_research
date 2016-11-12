
import tensorflow as tf


x = tf.constant([[1341,342,343],
                                  [4, 5, 6],
                                  [7, 8, 9]])
idx = tf.constant([1, 0, 2])
idx_flattened = tf.range(0, 3)
y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                            idx_flattened)  # use flattened indices

with tf.Session(''):
      print y.eval()  # [2 4 9]
