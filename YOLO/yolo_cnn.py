 # import the necessary packages

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

from yolo_layer import YoloDetect

# =============================================================================
# This module contains following network architecture 
# Here I give it in an explicity way writing in plan keras-style
# - yolo-small : 
# - yolo-tiny :
# =============================================================================


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

        model.add(Convolution2D(64, 7, 7, input_shape=(W,H,3), 
            border_mode='same' , subsample=(2,2)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
            strides=(2,2)))


        model.add(Convolution2D(192, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2) ))


        model.add(Convolution2D(128, 1, 1, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        
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

        model.add(Convolution2D(512, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        
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

        model.summary()
        return model

    def yolo_tiny(self):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H

        model = Sequential()

        model.add(Convolution2D(16, 3, 3, input_shape=(W,H,3), 
            border_mode='same' ))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
            strides=(2,2)))

        model.add(Convolution2D(32,3,3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
            strides=(2,2)))
        
        model.add(Convolution2D(64,3,3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
            strides=(2,2)))

        model.add(Convolution2D(128,3,3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
            strides=(2,2)))

        model.add(Convolution2D(256,3,3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
            strides=(2,2)))

        model.add(Convolution2D(512,3,3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
            strides=(2,2)))

        model.add(Convolution2D(1024,3,3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024,3,3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024,3,3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())
        model.add(Dense(256, activation='linear'))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))
        model.add(Dense(1470, activation='linear'))

        model.summary()
        return model

    def train(self, ):
        model = self.yolo_small()
        model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',metrics=['accuracy',f1])

    def predict(self, ):
        pass

    def save(self, ):
        pass



yolooo = YoloNetwork()
#model = yolooo.yolo_small()
model2 = yolooo.yolo_tiny()






