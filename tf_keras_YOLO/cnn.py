import tensorflow as tf
import numpy as np




 # import the necessary packages

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dropout
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
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

    def yolo_small(self):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        model = Sequential()

        # Beaware of input shape might be 448,448,3 or 3,448,448 depend on keras.backend 
        model.add(Convolution2D(64, 7, 7, input_shape=(W,H,3), 
            border_mode='same' , subsample=(2,2)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
            strides=(2,2)))


        model.add(Convolution2D(192, 3, 3, border_mode='same'))
        # subsample default = 1,1 
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2) ))


        model.add(Convolution2D(128, 1, 1, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        #
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))   


        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))  
        #

        model.add(Convolution2D(512, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        ###
        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.summary()
        model.add(Convolution2D(1024, 3, 3, border_mode='same',
            subsample=(2,2)))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))  

        model.add(Dense(S*S*(5*B+C), activation='linear'))
        return model




    def train(self, ):
        model = self.yolo_small()
        model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',metrics=['accuracy',f1])

    def predict(self, ):
        pass

    def save(self, ):
        pass


# From DokerKeras

def SimpleNet(yoloNet):
    model = Sequential()

    #Convolution Layer 2 & Max Pooling Layer 3
    model.add(ZeroPadding2D(padding=(1,1),input_shape=(448,448,3)))
    model.add(Convolution2D(16, 3, 3, weights=[yoloNet.layers[1].weights,yoloNet.layers[1].biases],border_mode='valid',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Use a for loop to replace all manually defined layers
    for i in range(3,yoloNet.layer_number):
        l = yoloNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            model.add(ZeroPadding2D(padding=(l.size//2,l.size//2,)))
            model.add(Convolution2D(l.n, l.size, l.size, weights=[l.weights,l.biases],border_mode='valid',subsample=(1,1)))
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "MAXPOOL"):
            model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        elif(l.type == "FLATTEN"):
            model.add(Flatten())
        elif(l.type == "CONNECTED"):
            model.add(Dense(l.output_size, weights=[l.weights,l.biases]))
        elif(l.type == "LEAKY"):
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "DROPOUT"):
            pass
        else:
            print "Error: Unknown Layer Type",l.type
    return model

yolooo = YoloNetwork()
model = yolooo.yolo_small()






