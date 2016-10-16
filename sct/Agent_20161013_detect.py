
# =============================================================================
# Author : Kent Chiu
# =============================================================================
# Des.
# - This CNN_Agent Series Py is to provide API with Keras Model
#
# Parameter
# - json_path
# - h5_path
# - data preprocessor parameter
# - test Data handler
# - output handler
# -

from keras.models import model_from_json
import os, math
import numpy as np
import cv2
from skimage.io import imread
import argparse
import logging
import imageio

# =======================================================
# Define cli variables

def get_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jsonFile', required=True)
    parser.add_argument('-5', '--h5File', required=True)
    parser.add_argument('-v','--vidFile', help='input vid file')
    parser.add_argument('-i', '--imgFolder', help='input folder')
    args = vars(parser.parse_args())
    return args

# Ex: -j hub/model/Agent_20161013.json -5 hub/model/Agent_201610139.h5
# -v D:\\2016-01-21\\10.167.10.158_01_20160121082638418_1.mp4

assert model_json_path.split('.')[-1]=='json'
assert model_weight_path.split('.')[-1]=='h5'


vid = imageio.get_reader(args['vidFile'])

with open(args['jsonFile'], 'r') as f:
    loaded_model = model_from_json(f.read())

print ('Successful loading Model')

loaded_model.load_weights(args['h5File'])
print ('Successful loading Model Weight')




def test_case():
    if os.name=='nt':
        vid = imageio.get_reader('D:\\2016-01-21\\10.167.10.158_01_20160121082638418_1.mp4')
    else:
        vid = imageio.get_reader('/Users/kentchiu/'
            'MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
