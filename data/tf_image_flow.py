import tensorflow as tf

filename_queue = tf.train.string_input_producer(['/vatic_id2/6.jpg']) 
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
