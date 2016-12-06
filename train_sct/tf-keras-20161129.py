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
from tf_keras_board import get_summary_op

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
log_dir = '../hub/logger_{}'.format(model_name)

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

#train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)
train_step = tf.train.RMSPropOptimizer(1e-10, momentum=0.9).minimize(loss)
# Initializing the variables
summary_op = get_summary_op(model, loss)

init = tf.global_variables_initializer()

MIN_LOSS = 9999
with tf.Session() as sess : 
    sess.run(init)
    

    writer = tf.train.SummaryWriter(log_dir, tf.get_default_graph())
    # test mode
    epoch = 1
    while epoch < epoch_size:
        SUM_LOSS= 0
        if epoch == 1:
            try:
                model.load_weights('../hub/model/{}-v1.h5'.format(model_name)) 
            except :
                pass
        else : 
            pass
        step = 1
        DATAFLOW = processer.genYOLO_foler_batch('../data/vatic_id2', batch_size=batch_size)
        for images_feed, labels_feed in DATAFLOW :
            images_feed = batch_check(np.asarray(images_feed), batch_size)
            labels_feed = batch_check(np.asarray(labels_feed), batch_size)    

            _, lossN, summary_log = sess.run([train_step,loss,summary_op], feed_dict = 
            	{input_tensor : images_feed, true_y :labels_feed, K.learning_phase(): 0})    

            
            SUM_LOSS+=lossN
            writer.add_summary(summary_log, epoch*step)
            step+=1


        SUM_LOSS = SUM_LOSS/(batch_size*step)

        
        print ('EP [{}] Iter {} Loss {}'.format(epoch,step,SUM_LOSS ))
        MIN_LOSS = min(SUM_LOSS, MIN_LOSS)
        if SUM_LOSS<=MIN_LOSS:
            
            model.save_weights('../hub/model/{}-v2.h5'.format(model_name)) 
            print ('SAVE WEIGHT')
        else:
            print ('NOT SAVE')
        epoch +=1


# logger 
# Due to the tf-keras-20161125 => since to have fixed output 
# init 1e-8 momentum = 0.9 batch_size=2
# 1e-7 ~a8ound 20

