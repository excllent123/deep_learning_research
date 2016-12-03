import tensorflow as tf 
import cv2
import numpy as np

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import numpy as np

from keras.layers import Input

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential , Model
import keras.backend as K

from tf_yolo import YoloDetector
# from yolo_cnn import YoloNetwork
from yolo_preprocess import VaticPreprocess

# =============================================
# Set Variables

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

model_json = model.to_json()
with open("../hub/model/{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)
    print ('saving model struct as ' + "../hub/model/{}.json".format(model_name))


input_tensor = Input(shape=(H, W, 3))
true_y = tf.placeholder(tf.float32, shape = (None , S*S*(B*5+C)))