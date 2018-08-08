import tensorflow as tf 


input_id = tf.placeholder(tf.int32, [None])
emb = tf.get_variable('emb', shape=(100,100))
oo  = tf.nn.embedding_lookup(emb, input_id)

with tf.Session() as s:
	s.run(tf.global_variables_initializer())
	print (s.run(oo, feed_dict={ input_id:[1,3,2] }))