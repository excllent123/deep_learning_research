import tensorflow as tf
import numpy as np




 # import the necessary packages
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Sequential

from yolo_layer import YoloDetect




class YoloNetwork(object):
    def __init__(self, numCla=20, rImgW=448, rImgH=448, S=7, B=2):
        self.S = S
        self.B = B
        self.C = numCla
        self.W = rImgW
        self.H = rImgH
        self.iou_threshold=0.5
        self.classMap  =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
        "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

    def build(self):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        model = Sequential()

        model.add(Convolution2D(64, 7, 7, input_shape=(3,W,H) ,border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

        model.add(Convolution2D(192, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)  ))


        model.add(Convolution2D(128, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(256, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))   


        model.add(Convolution2D(256, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(256, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(256, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(256, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))

        model.add(Convolution2D(512, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(1024, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  

        model.add(Convolution2D(512, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(1024, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(512, 1, 1, border_mode='same', activation='relu'))
        model.add(Convolution2D(1024, 3, 3, border_mode='same', activation='relu'))

        model.add(Convolution2D(1024, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(1024, 3, 3, border_mode='same', activation='relu', strides=2))

        model.add(Convolution2D(1024, 3, 3, border_mode='same', activation='relu'))
        model.add(Convolution2D(1024, 3, 3, border_mode='same', activation='relu'))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
  
        model.add(Dense(S*S*(5*B+C), activation='linear'))
        return model

    def train(self, ):
        model = self.build()
        model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',metrics=['accuracy',f1])

    def predict(self, ):
        pass

    def save(self, ):
        pass


yolooo = YoloNetwork()
model = yolooo.build()
model.summary()





