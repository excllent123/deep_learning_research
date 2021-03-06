import tensorflow as tf



from keras import backend as K


import numpy as np



from keras.layers import Input

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential , Model

from tf_yolo import YoloDetector
from yolo_cnn import YoloNetwork
from yolo_preprocess import VaticPreprocess

def batch_check(x, batch_size):
    while len(x) != batch_size : 
        x = list(x)
        x.append(x[0])
        x = np.asarray(x)
    return x


W = 448
H = 448
S = 7
B = 2
C = 2
batch_size = 2
epoch_size = 2500

model_name = __file__.split('\\')[-1].split('.')[0]
file_path = '../data_test/vatic_example.txt'
maplist = ['Rhand', 'ScrewDriver']

input_X = tf.placeholder(tf.float32, shape=(None, W,H,3))
true_y = tf.placeholder(tf.float32, shape = (None , S*S*(B*5+C)))

processer = VaticPreprocess(file_path, maplist=maplist)

yolooo = YoloNetwork(C=C)
A =      YoloDetector(C = C)


# ==================
# model = yolooo.yolo_tiny_v2()
# this not works! 
x = Convolution2D(16, 3, 3, input_shape=(W,H,3), border_mode='same' )(input_X)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' ,strides=(2,2))(x)

x = Convolution2D(32,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' ,strides=(2,2))(x)

x = Convolution2D(64,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' ,strides=(2,2))(x)

x = Convolution2D(128,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' , strides=(2,2))(x)

x = Convolution2D(256,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
    strides=(2,2))(x)

x = Convolution2D(512,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
    strides=(2,2))(x)

x = Convolution2D(1024,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)

x = Convolution2D(1024,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)

x = Convolution2D(1024,3,3,border_mode='same')(x)
x = LeakyReLU(alpha=0.1)(x)

x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])
x = Dense(256, activation='linear')(x)
x = Dense(2048)(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.2)(x)
pred_y = Dense(S*S*(5*B+C), activation='linear')(x)



print true_y[0,:].get_shape()



# where true_y [None, S*S*(B*5+C)]
loss = A.loss(true_y, pred_y, batch_size=batch_size) # tf-stle slice must have same rank

#loss = tf.py_func(A.loss, true_y[0,:], pred_y[0,:])

train_step = tf.train.RMSPropOptimizer(0.0001, momentum=0.5).minimize(loss)

# Initializing the variables
init = tf.initialize_all_variables()

sess = tf.Session()
K.set_session(sess)

with sess.as_default():
    
    # test mode
    epoch = 1
    while epoch < epoch_size:
        try :
            model.load_weights('{}.h5'.format(model_name))    
        except:
            print ('No wiehgt to be loaded ')
        sess.run(init)
        step = 1
        DATAFLOW = processer.genYOLO_foler_batch('../data/vatic_id2', batch_size=batch_size)
        for images_feed, labels_feed in DATAFLOW :
            images_feed = batch_check(np.asarray(images_feed), batch_size)
            labels_feed = batch_check(np.asarray(labels_feed), batch_size)    

            sess.run(train_step, feed_dict = 
            	{input_X : images_feed, true_y :labels_feed, K.learning_phase(): 1})    

            if step % 2 ==0:
                lossN  =  sess.run([loss], feed_dict = 
                    {input_X : images_feed, true_y :labels_feed, K.learning_phase(): 1})
                print ('Iter : [{}] , Epoch : [{}], Loss : [{}]'.format(step, epoch, lossN)) 

            step+=1
        epoch +=1
        try:
            model.save_weights('{}.h5'.format(model_name))
        except:
            print ('Not saving the weight')