import tensorflow as tf
import keras.backend as K

import numpy as np

from keras.layers import Input

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential , Model

from tf_yolo import YoloDetector
from yolo_cnn import YoloNetwork
from yolo_preprocess import VaticPreprocess

# =========================================
W = 448
H = 448
S = 7
B = 2
C = 2 
nb_classes = C
batch_size = 2
epoch_size = 2500

model_name = __file__.split('\\')[-1].split('.')[0]
file_path = '../data_test/vatic_example.txt'
maplist = ['Rhand', 'ScrewDriver']

# =======================================

A =      YoloDetector(C = C)
processer = VaticPreprocess(file_path, maplist=maplist, detector=A)

yolooo = YoloNetwork(C=C)
model = yolooo.yolo_tiny_v2()

# =======================================
model_json = model.to_json()
with open("../hub/model/{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)
    print ('saving model struct as ' + "../hub/model/{}.json".format(model_name))

# =======================================

true_y = tf.placeholder(tf.float32, shape = (None , S*S*(B*5+C)))
input_tensor = Input(shape=(H, W, 3))

pred_y = model(input_tensor)



def batch_check(x, batch_size):
    while len(x) != batch_size : 
        x = list(x)
        x.append(x[0])
        x = np.asarray(x)
    return x


# where true_y [None, S*S*(B*5+C)]
loss = A.loss(true_y, pred_y, batch_size=batch_size) # tf-stle slice must have same rank
#loss = tf.py_func(A.loss, true_y[0,:], pred_y[0,:])

# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
train_step = tf.train.RMSPropOptimizer(1e-5, momentum=0.5).minimize(loss)
# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess : 
    sess.run(init)
    # test mode
    epoch = 1
    while epoch < epoch_size:
        if epoch == 1:
            try:
                model.load_weights('../hub/model/{}.h5'.format(model_name)) 
            except :
                pass
        else : 
            try:
                model.load_weights('../hub/model/{}.h5'.format(model_name))   
            except:
                print ('NOT LOAD WEIGHT')
        step = 1
        DATAFLOW = processer.genYOLO_foler_batch('../data/vatic_id2', batch_size=batch_size)
        for images_feed, labels_feed in DATAFLOW :
            images_feed = batch_check(np.asarray(images_feed), batch_size)
            labels_feed = batch_check(np.asarray(labels_feed), batch_size)    

            sess.run(train_step, feed_dict = 
            	{input_tensor : images_feed, true_y :labels_feed, K.learning_phase(): 0})    

            if step % 10 ==0:
                lossN  =  sess.run([loss], feed_dict = 
                    {input_tensor : images_feed, true_y :labels_feed, K.learning_phase(): 1})
                print ('EP [{}] Iter {} Loss {}'.format(epoch,step, lossN))    

            step+=1
        epoch +=1

        try:
            model.save_weights('../hub/model/{}.h5'.format(model_name)) 
            print ('SAVE WEIGHT')
        except:
            print ('NOT SAVE')


# logger 
# Due to the tf-keras-20161125 => since to have fixed output 
# 

