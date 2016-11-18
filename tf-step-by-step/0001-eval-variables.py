

# learning tensorflow in hard ways 
# 

# Object : 
# python-objects with value 
# numpy-objects with value
# tensor-objects 
# tf.Variable with value 
# tf.constant with value 
# tf.operations without value 
# 
# numpy <-> tensorflow 
# 

import tensorflow as tf 
import numpy as np
np.random.seed(0)


x = np.random.random(size=(100,30,30,3))


# 2-D tensor `a`
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) #=> [[1. 2. 3.]
                                                  #    [4. 5. 6.]]
# 2-D tensor `b`
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) #=> [[7. 8.]
                                                     #    [9. 10.]
                                                     #    [11. 12.]]
# this is an operation, if the operation is based on tf-tensor-object with value 
# the object could be eval 
c = tf.matmul(a, b) #=> [[58 64]
                    #    [139 154]]


# Way 1
# use placeholder and feeding it with numpy, python value
# and just print it 
# once you use a place-holder 
# it is the container let the data flow in and out
# 
object_1 = tf.placeholder("float", [None, 30,30,3], name='input_x')
object_2 = tf.Variable(x)

# Way 2 
# Just define your tf tensor object & print it with eval


print (x.shape)

tf_x = tf.placeholder("float", [None, 30,30,3])

input1 = tf.placeholder(tf.float32) # place holder is not able to be eval
input2 = tf.Variable(x)
#output = tf.mul(input1, input2)


with tf.Session() as sess:
    
    sess.run(tf.initialize_all_variables())
    
    print (object_2.eval())
    print (c.eval)
    
    # Show in list or np array 
    x_ = sess.run([tf_x], feed_dict = { tf_x:x })
    print (x_ )