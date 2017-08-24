import tensorflow as tf 


def tf_print(obj):
	''''''
	with tf.Session as sess:
		sess.run(tf.global_variables_initializer())
		print (sess.run(obj))

