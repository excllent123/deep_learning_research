

import tensorflow as tf
v1 = tf.Variable(1.32, name="v1")
v2 = tf.Variable(1.33, name="v2")

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)
  save_path = saver.save(sess, "model-test.ckpt")

saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, "model-test.ckpt")
  print("Model restored.")

