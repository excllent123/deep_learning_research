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

import imageio 
import cv2
import numpy as np
from keras.models import model_from_json
from tf_yolo import YoloDetector

# ================================
# Set you parameters by config

import argparse 

parser = argparse.ArgumentParser()

parser.add_argument('-f','--frameid',  type=int)
arg=parser.parse_args()

file_path = '../data_test/vatic_example.txt'
maplist = ['Rhand', 'ScrewDriver']
W=448
H=448
S=7
B=2
C=2
nb_epoch = C


yolo_detect = YoloDetector(C=C)
yolo_detect.set_class_map(maplist)


def get_test_img(img, W=448, H=448):

    h,w,c = img.shape
    print h
    if h!=H or w!=W:
        img = cv2.resize(img, (H, W))
    img = np.resize(img,[1,H,W,c])
    return img

# load trained model 
def get_model(jsonPath,weightPath):
    with open(jsonPath, 'r') as f:
        loaded_model_json = f.read()
    # struc model
    model = model_from_json(loaded_model_json)
    # loading model weight
    model.load_weights(weightPath)
    # add model
    return model 


TFmodel = get_model(jsonPath='../hub/model/tf-keras-20161120.json',
               weightPath='../hub/model/tf-keras-20161120-v2.h5')

# ====================================================================

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
model = Model(base_model.input, pred_y)
model.load_weights('tf-keras-20161125-v7.h5')
# ====================================================================

#input_tensor = Input(shape=(H, W, 3))

#y = TFmodel(input_tensor)

init = tf.initialize_all_variables()

# get test data
vid = imageio.get_reader('~/data/10.167.10.159_01_20160121083216799_2.mp4')
vid = imageio.get_reader('../data/vatic_id2/output.avi')

if arg.frameid and arg.frameid< vid.get_length:
    frameid = arg.frameid
else : 
    frameid = 600
img = vid.get_data(frameid)
img = cv2.resize(img, (H, W))

test_img = get_test_img(img)


with tf.Session() as sess : 
    sess.run(init)
    output_tensor = sess.run(pred_y, feed_dict = 
                {input_tensor : test_img, K.learning_phase(): 0})   
    # output = nd array
    #print (output_tensor)

    bbx = yolo_detect.decode(output_tensor[0,:], threshold=1e-25)
    print (bbx)

img_copy = img.copy()
for item in bbx: 
    name, cX,cY,w,h , _= item
    def check_50(x):
        if x < 50 :
            x = 50 
        return x
    cX,cY,w,h = map(check_50,[cX,cY,w,h] )
    pt1= ( int(cX-0.5*w) ,int(cY-0.5*h) )
    pt2= ( int(cX+0.5*w) ,int(cY+0.5*h) )    
    cv2.rectangle(img_copy, pt1, pt2, (255,255,255), thickness=2)
print img.shape

cv2.imshow("Before",img)
cv2.imshow("After", img_copy)
cv2.waitKey()


