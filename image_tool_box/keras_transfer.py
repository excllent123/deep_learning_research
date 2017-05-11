from keras.models import Model

from keras import backend as K 

from keras import optimizers

from keras.callbacks import ModelCheckpoint

from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception

import argparse 

parse = argparse.ArgumentParser()
# base_mode
parse.add_argument()
# model_structure 
parse.add_argument()
# learning rate 
parse.add_argument()
# 
parse.add_argument()
parse.add_argument()
parse.add_argument()




def get_base_model(name):
    memo_catch = {
    	'vgg16': VGG16
    	'vgg19': VGG19
    	'res50': ResNet50
    	'inception_v3': InceptionV3
    	'xception': Xception
    }
    assert (name in memo_catch.keys())==True
    return memo_catch[name]

