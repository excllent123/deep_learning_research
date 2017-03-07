
# Author : Kent Chiu & Grus

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.models import Sequential

class CNNFactory:
    """
    description : 
    this class provide an interface for building classic 
    convolutional neural network algorithm, powered by keras 

    example  :
    # build model 
    model = CNNFactory.build('shallownet', *args, **kargs)
    model.complie & train ... 
    """
    @staticmethod
    def build(name, *args, **kargs):
        memo = {
            "lenet":      CNNFactory.lenet,
            "shallownet": CNNFactory.shallownet,
            "karphynet":  CNNFactory.karphynet,
            "minivggnet": CNNFactory.minivggnet}

        builder = memo.get(name, None)

        if builder is None:
            return None

        return builder(*args, **kargs)

    @staticmethod
    def lenet(numChannels, imgRows, imgCols, numClasses, activation="tanh", **kwargs):
        # initialize the model
        model = Sequential()

        model.add(Convolution2D(20, 5, 5, border_mode="same",
            input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # define the second FC layer
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))
        return model

    @staticmethod
    def shallownet(numChannels, imgRows, imgCols, numClasses, **kwargs):
        model = Sequential()

        # define the first (and only) CONV => RELU layer
        model.add(Convolution2D(32, 3, 3, border_mode="same",
            input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))

        # FC layer
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))
        return model

    @staticmethod
    def minivggnet():
        # initialize the model
        model = Sequential()

        # define the first set of CONV => RELU => CONV => RELU => POOL layers
        model.add(Convolution2D(32, 3, 3, border_mode="same",
            input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if dropout:
            model.add(Dropout(0.25))

        # define the second set of CONV => RELU => CONV => RELU => POOL layers
        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))

        # define the set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))

        if dropout:
            model.add(Dropout(0.5))

        model.add(Dense(numClasses))
        model.add(Activation("softmax"))
        return model

    @staticmethod
    def karphynet():
        model = Sequential()

        # define the first set of CONV => RELU => POOL layers
        model.add(Convolution2D(16, 5, 5, border_mode="same",
            input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))

        # define the second set of CONV => RELU => POOL layers
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))

        # define the third set of CONV => RELU => POOL layers
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.5))

        # define the soft-max classifier
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        # return the network architecture
        return model
