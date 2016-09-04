from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import os
import sys
import progressbar
import argparse
import commentjson as json

from imutils import paths
from skimage.io import imread
import imageio
import cv2

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import matplotlib.pyplot as plt


from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json


import os


#
