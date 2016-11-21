
import tensorflow as tf

q = tf.FIFOQueue(3,'float32')

init = q.enqueue_many(([0.,0.,0.],))

x = q.dequeue()

y = x+1

q_inc = q.enqueue([y])


with tf.Session() as sess :
    sess.run(init)
    sess.run(q_inc)
    print q
    print q_inc

    sess.run(q_inc)
    print q
    print q_inc

    sess.run(q_inc)
    print q
    print q_inc
