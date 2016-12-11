import tensorflow as tf


import keras.backend as K
from keras.utils import generic_utils

#from keras.applications.resnet50 import ResNet50, preprocess_input
#from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import numpy as np

from keras.layers import Input

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential , Model

import relat_import
from tf_keras_YOLO.yolo_layer import YoloDetector
from tf_keras_YOLO.yolo_cnn import YoloNetwork
from tf_keras_YOLO.yolo_preprocess import VaticPreprocess
from tf_keras_YOLO.tf_keras_board import get_summary_op

flags = tf.flags
flags.DEFINE_float('learning_rate', 1e-6, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
FLAGS = flags.FLAGS

W = 448
H = 448
S = 7
B = 2
C = 2 
nb_classes = C
batch_size = 2
epoch_size = 2500

model_name = __file__.split('\\')[-1].split('.')[0]
file_path = '../hub_data/vatic/vatic_id2/example_off_withSave4.txt'
maplist = ['Rhand', 'ScrewDriver']
log_dir = '../hub_logger/{}'.format(model_name)



# =======================================

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
x = AveragePooling2D((5,5))(x) # for VGG16
x = Flatten()(x)
x = Dense(3048)(x)
x = LeakyReLU(alpha=0.1)(x)
#x = Dense(2048)(x)
#x = LeakyReLU(alpha=0.1)(x)
pred_y = Dense(S*S*(5*B+C), activation='linear')(x)

# by pass the tf-style, to store weight in h5.py
model = Model(base_model.input, pred_y)

# =======================================
model_json = model.to_json()
with open("../hub_model/{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)
    print ('saving model struct as ' + "../hub_model/{}.json".format(model_name))

# =======================================


# # freeze base model layers

print (len(base_model.layers))
iii = 0
for layer in base_model.layers:
    if iii == len(base_model.layers)-2:
        layer.trainable = True
    else:
        layer.trainable = False
    iii+=1


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
train_step = tf.train.RMSPropOptimizer(FLAGS.learning_rate, momentum=0.9).minimize(loss)
# Initializing the variables

summary_op = get_summary_op(model, loss)
init = tf.global_variables_initializer()

MIN_LOSS = 9999
with tf.Session() as sess :
    sess.run(init)


    writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    # test mode
    epoch = 1
    while epoch < epoch_size:
        SUM_LOSS= 0
        if epoch == 1:
            try:
                model.load_weights('../hub_model/{}-v1.h5'.format(model_name))
            except :
                pass
        else :
            pass
        step = 1
        DATAFLOW = processer.genYOLO_foler_batch('../hub_data/vatic/vatic_id2', batch_size=batch_size)
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

            model.save_weights('../hub_model/{}-v2.h5'.format(model_name))
            print ('SAVE WEIGHT')
        else:
            print ('NOT SAVE')
        epoch +=1


# v1 : ~19 
# v2 : start to trian 1 one layer