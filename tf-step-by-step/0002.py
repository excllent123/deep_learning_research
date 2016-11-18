import tensorflow as tf

def create_variables():
  with tf.variable_scope('model'):
    w1 = tf.get_variable('w1', [1, 2])
    b1 = tf.get_variable('b1', [2])

def inference(input):
  with tf.variable_scope('model', reuse=True):
    w1 = tf.get_variable('w1')
    b1 = tf.get_variable('b1')
    output = tf.matmul(input, w1) + b1
  return output

create_variables()

i1a = tf.placeholder(tf.float32, [3, 1])
o1a = inference(i1a)

i1b = tf.placeholder(tf.float32, [3, 1])
o1b = inference(i1b)

loss = tf.reduce_mean(o1a - o1b)


with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  loss3 = sess.run(loss, feed_dict={i1a: [[0.], [1.], [2.]], i1b: [[0.5], [1.5], [2.5]]})
  print loss3
  #print loss.eval()  # not work -> arrise the InvalueAugmentError  
                      # You must feed a value for placeholder tensor 
                      # 'Placeholder' with dtype float and shape [3,1]
  #print i1a.eval()
  #print i1b.eval()
  #print w1.eval()
  #print b1.eval()


t = tf.constant(42.0)
u = tf.constant(37.0)
tu = tf.mul(t, u)
ut = tf.mul(u, t)
with tf.Session() as sess:
   print (tu.eval() )  # runs one step
   print (ut.eval() ) # runs one step
   print (sess.run([tu, ut]) ) # runs a single step




input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)
with tf.Session() as sess:
  print (sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
  print input1

# output:
# [array([ 14.], dtype=float32)]

# The tf.placeholder() op is a way of defining a symbolic argument
# to the computation: it doesnt have any value itself
