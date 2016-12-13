import tensorflow as tf
import keras.backend as K
from keras.layers import Input
from yolo_layer import YoloDetector
from yolo_preprocess import VaticPreprocess
import imageio
import cv2
import numpy as np
from keras.models import model_from_json

# ================================
# Set you parameters by config

import argparse

# ==========================================================
# get the parameters
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--frameid',  type=int)
parser.add_argument('-t', '--threshold', type=float)
parser.add_argument('-w', '--weight_file', type=str)
parser.add_argument('-j', '--json_file',type=str)
parser.add_argument('-v', '--vid_path',type=str)
arg=parser.parse_args()

# ===========================================================
# initialize the parameters

threshold   = arg.threshold if (arg.threshold and arg.threshold < 1) else 0.2

weight_file = arg.weight_file if arg.weight_file else 'tf-keras-20161125-v7.h5'

json_file   = arg.json_file if arg.json_file else '../hub/model/tf-keras-20161120.json'

vid_path    = arg.vid_path if arg.vid_path else '../hub_data/vatic/vatic_id2/output.avi'

vid  = imageio.get_reader(vid_path)

frameid = arg.frameid if (arg.frameid and arg.frameid< vid.get_length ) else 600



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
    if h!=H or w!=W:
        img = cv2.resize(img, (H, W))
    img = np.resize(img,[1,H,W,c])
    img *= int(255.0/img.max())
    return img

# load trained model
def get_model(jsonPath):
    with open(jsonPath, 'r') as f:
        loaded_model_json = f.read()
    # struc model
    model = model_from_json(loaded_model_json)
    # add model
    return model


TFmodel = get_model(jsonPath=json_file)

# =====================================
# Alternative way
# img = vid.get_data(frameid)
# img = cv2.resize(img, (H, W))
# test_img = get_test_img(img)
# print (TFmodel.predict(test_img)).shape
# ====================================================================

input_tensor = Input(shape=(H, W, 3))

pred_y = TFmodel(input_tensor)

init = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init)

    # this is the beauty of keras
    TFmodel.load_weights(weight_file)
    while frameid<vid.get_length():
        img = vid.get_data(frameid)
        img = cv2.resize(img, (H, W))
        test_img = get_test_img(img)

        output_tensor = sess.run(pred_y, feed_dict =
                {input_tensor : test_img, K.learning_phase(): 0})

        bbx = yolo_detect.decode(output_tensor[0,:], threshold=threshold)
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
        # print img.shape
        #cv2.imshow("Before",img)
        cv2.imshow("After", img_copy)
        cv2.waitKey()
        frameid+=5

