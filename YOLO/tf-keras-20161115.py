import tensorflow as tf



from keras.layers.core import K

import numpy as np



from keras.layers import Input

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential , Model

from tf_yolo import YoloDetector
from yolo_cnn import YoloNetwork
from yolo_preprocess import VaticPreprocess

# https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
# https://github.com/fchollet/keras/issues/2310

#img = tf.placeholder(tf.float32, shape=(None, 784))#

#from keras.layers import Dense#

## Keras layers can be called on TensorFlow tensors:
#x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
#x = Dense(128, activation='relu')(x)
#preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation#
#

#labels = tf.placeholder(tf.float32, shape=(None, 10))#

#from keras.objectives import categorical_crossentropy#

#loss = tf.reduce_mean(categorical_crossentropy(labels, preds))#

#from tensorflow.examples.tutorials.mnist import input_data
#mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)#

#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#with sess.as_default():
#    for i in range(100):
#        batch = mnist_data.train.next_batch(50)
#        train_step.run(feed_dict={img: batch[0],
#                                  labels: batch[1]})#
#
#

#from keras.metrics import categorical_accuracy as accuracy#

#acc_value = accuracy(labels, preds)
#with sess.as_default():
#    print acc_value.eval(feed_dict={img: mnist_data.test.images,})

 
W = 448
H = 448
S = 7
B = 2
C = 2
batch_size = 4
epoch_size = 25

model_name = __file__.split('\\')[-1].split('.')[0]
file_path = '../data_test/vatic_example.txt'
maplist = ['Rhand', 'ScrewDriver']

input_X = tf.placeholder(tf.float32, shape=(None, W,H,3))
true_y = tf.placeholder(tf.float32, shape = (None , S*S*(B*5+C)))



yolooo = YoloNetwork(C=C)
A =      YoloDetector(C = C)
processer = VaticPreprocess(file_path, maplist=maplist, detector=A)

model = yolooo.yolo_small()
# this works! 


pred_y = model(input_X)
print true_y[0,:].get_shape()

def batch_check(x, batch_size):
    while len(x) != batch_size : 
        x = list(x)
        x.append(x[0])
        x = np.asarray(x)
    return x


# where true_y [None, S*S*(B*5+C)]
loss = A.loss(true_y, pred_y, batch_size=batch_size) # tf-stle slice must have same rank

#loss = tf.py_func(A.loss, true_y[0,:], pred_y[0,:])

train_step = tf.train.GradientDescentOptimizer(0.000000000001).minimize(loss)
# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess : 
    sess.run(init)
    # test mode
    epoch = 1
    while epoch < epoch_size:
        model.load_weights('my_weights.h5')    

        step = 1
        DATAFLOW = processer.genYOLO_foler_batch('../data/vatic_id2', batch_size=batch_size)
        for images_feed, labels_feed in DATAFLOW :
            images_feed = batch_check(np.asarray(images_feed), batch_size)
            labels_feed = batch_check(np.asarray(labels_feed), batch_size)    

            sess.run(train_step, feed_dict = 
            	{input_X : images_feed, true_y :labels_feed, K.learning_phase(): 0})    

            if step % 2 ==0:
                lossN  =  sess.run([loss], feed_dict = 
                    {input_X : images_feed, true_y :labels_feed, K.learning_phase(): 1})
                print ('EP [{}] Iter {} Loss {}'.format(epoch,step, lossN))    

            step+=1
        epoch +=1
        model.save_weights('my_weights.h5')
