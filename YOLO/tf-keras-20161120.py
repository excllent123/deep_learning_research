import tensorflow as tf


import keras.backend as K


from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import numpy as np

from keras.layers import Input

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential , Model

from tf_yolo import YoloDetector
# from yolo_cnn import YoloNetwork
from yolo_preprocess import VaticPreprocess


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


true_y = tf.placeholder(tf.float32, shape = (None , S*S*(B*5+C)))



# yolooo = YoloNetwork(C=C)
A =      YoloDetector(C = C)
processer = VaticPreprocess(file_path, maplist=maplist, detector=A)


# =======================================

input_tensor = Input(shape=(H, W, 3))
#base_model = ResNet50(input_tensor=input_tensor, include_top=False)
#base_model = InceptionV3(input_tensor=input_tensor, include_top=False)
base_model = VGG16(input_tensor=input_tensor, include_top=False)

x = base_model.output
# x = AveragePooling2D((8,8), strides=(8,8))(x)
x = AveragePooling2D((7,7))(x) # for VGG16
x = Flatten()(x)
x = Dense(2048)(x)
x = LeakyReLU(alpha=0.1)(x)
#x = Dense(2048)(x)
#x = LeakyReLU(alpha=0.1)(x)
pred_y = Dense(S*S*(5*B+C), activation='linear')(x)

# by pass the tf-style, to store weight in h5.py
model = Model(base_model.input, pred_y)

# # freeze base model layers
for layer in base_model.layers:
    layer.trainable = False


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
train_step = tf.train.RMSPropOptimizer(0.0000001, momentum=0.9).minimize(loss)
# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess : 
    sess.run(init)
    # test mode
    epoch = 1
    while epoch < epoch_size:
        if epoch == 1:
            model.load_weights('tf-keras-20161120.h5') 
        try:
            model.load_weights('tf-keras-20161120-v2.h5')    
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
            model.save_weights('tf-keras-20161120-v2.h5')
            print ('SAVE WEIGHT')
        except:
            print ('NOT SAVE')
# tf-keras-20161120.h5 for 36hrs traning 

# continue with saving v2
