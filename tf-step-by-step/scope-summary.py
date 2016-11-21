import tensorflow as tf



with tf.name_scope('T1'):
    with tf.name_scope('T2'):
        v = tf.get_variable("v1", [1])
        x = v+ 1.0
        print v.name
        print x.op.name

with tf.name_scope('T3') :
    with tf.variable_scope('T4'):
        v = tf.get_variable("v1", [1])
        x = v+ 1.0
        print v.name
        print x.op.name

with tf.name_scope('T5'):
    with tf.variable_scope('T6'):
        v = tf.get_variable("v1", [1])
        x = v+ 1.0
        print v.name
        print x.op.name

'''OUTPUT
Note name_scope only affect the operations and not affect the variables
but the variable_scope affect both
v1:0
T1/T2/add
T4/v1:0
T3/T4/add
T6/v1:0
T5/T6/add
'''


with tf.variable_scope("S1") as S1 :
    print S1.name
with tf.variable_scope("Q1") as Q1 :
    with tf.variable_scope("Q2") as Q2 :
        print Q2.name
    with tf.variable_scope(S1) as Q3:
        print Q3.name
'''
S1
Q1/Q2
S1

'''



with tf.name_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print (v1.name)  # var1:0
print (v2.name)  # my_scope/var2:0
print (a.name)   # my_scope/Add:0
print ('----------------------------------')


with tf.variable_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.get_variable("var2",[1],  dtype=tf.float32)
    a = tf.add(v1, v2)

print (v1.name)  # my_scope/var1:0
print (v2.name)  # my_scope/var2:0
print (a.name)   # my_scope/Add:0
print ('----------------------------------')

# tf.name_scope         creates namespace for operators in the default graph.
# tf.variable_scope     creates namespace for both variables and operators in the default graph.
# tf.op_scope           As tf.name_scope, but for the graph in which specified variables were created.
# tf.variable_op_scope  As tf.variable_scope, but for the graph in which specified variables were created.


