
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
#
# Command Example:
# - python testDNN.py -m t -j hub/model/Agent_20161013.json
#                          -5 hub/model/Agent_201610139.h5
# - python testDNN.py -m t -j hub/model/Agent_20161014.json
#                          -5 hub/model/Agent_20161014-1130-1.00.h5
# - python testDNN.py -m t -j hub/model/Agent_20161015.json
#                          -5 hub/model/Agent_20161015-70-0.99.h5

from keras.models import model_from_json
import os
import dlib
import imutils

from skimage.io import imread
import argparse
import logging
import imageio
from src.softmax_detect import detect

# =======================================================
# Define cli variables

def get_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jsonFile', required=True)
    parser.add_argument('-5', '--h5File', required=True)
    parser.add_argument('-v','--vidFile', help='input vid file')
    parser.add_argument('-i', '--imgFolder', help='input folder')
    parser.add_argument('-m', '--mode', help='t : as test mode')
    args = vars(parser.parse_args())
    return args



def main_usage():
    vid = imageio.get_reader(args['vidFile'])
    img = vid.get_data(1)
    img = detect(img, model=loaded_model,winDim=(30,30),pyScale=1.5,
                 winStep=35, minProb=0.999995, numLabel=4,negLabel=[2])


def test_case():
    if os.name=='nt':
        vid = imageio.get_reader('D:\\2016-01-21\\10.167.10.158_01_20160121082638418_1.mp4')
    else:
        vid = imageio.get_reader('/Users/kentchiu/'
            'MIT_Vedio/2016-01-21/10.167.10.158_01_20160121082638418_1.mp4')
    idd = 1
    win = dlib.image_window()
    while True:
        img = vid.get_data(idd)
        img = imutils.resize(img, width=400)
        print (img.shape)
        img = detect(img, model=loaded_model,winDim=(30,30),pyScale=1.2,
                     winStep=15, minProb=0.92,
                     numLabel=4, negLabel=[0])
        win.clear_overlay()
        win.set_image(img)
        dlib.hit_enter_to_continue()
        idd+=20




if __name__=='__main__':
    args = get_cli()

    with open(args['jsonFile'], 'r') as f:
        loaded_model = model_from_json(f.read())

    loaded_model.load_weights(args['h5File'])
    print ('Successful loading Model Weight')
    if args['mode'] == 't':
        test_case()
    else:
        main_usage()



