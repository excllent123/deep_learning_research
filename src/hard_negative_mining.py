
# =============================================================================
# Author : Kent Chiu   
# =============================================================================
# - Hard Mining from ImageSequence or Video 
# - Extrate background imgs to trainable format 
# - Store them into the Output folder 


import os
import argparse 
import dlib 
import cv2
import imageio 
from skimage.io import imread, imshow, imsave
from common_func import sliding_window, pyramid, gen_file_path

import matplotlib.pyplot as plt

# =============================================================================
# Define 

cliVariable = argparse.ArgumentParser()

cliVariable.add_argument('-o', '--output', type=str, required=True,
    help='assign the output folder'
        ',if not exist, would automate to create one') 

cliVariable.add_argument('-i', '--input', type=str, required=True,
    help='assign the input Folder or path') 

cliVariable.add_argument('-w', '--winSize', type=int, nargs='+',
    help='assign the winSize (height, wid)') 

cliVariable.add_argument('-r', '--resolution',type=int, nargs='+',
    help='assign the resolution (height,wid)') 

cliVariable.add_argument('-s', '--winStep', type=int,
    help='assign the window step size') 

cliVariable.add_argument('-p', '--pyraScale', type=int,
    help='assign the window step size') 


args = vars(cliVariable.parse_args())

# =============================================================================
# Define 

if args['resolution'] is not None:
    RESOLUTION = args['resolution']
else :
    RESOLUTION = (600, 1000)

if args['winSize'] is not None:
    winHeight , winWeight = args['winSize']
else:
    winHeight , winWeight = 30, 30

if args['winStep'] is not None:
    WinStep = args['winStep']
else :
    WinStep = int(winHeight/4.0)

if args['pyraScale'] is not None:
    PYRAMID_SCALE = args['pyraScale']
else:
    PYRAMID_SCALE =1.5

# =============================================================================
# Saving Path Handler 

class HardMinng(object):
    def __init__(self, img, descriptor, training_model):
        pass

class Node(object):
    def __init__(self, **kwargs):
        pass

    def add_Lable(self, lable):
        assert hasattr(nextNode, 'is_Lable') == True
        pass

    def add_Relation(self, nextNode, relation):
        assert hasattr(nextNode, 'is_Node') == True

    def is_Node(self):
        pass

class ObjMemo(Node):
    def __init__(self):
        pass

class Lable(object):
    def __init__(self, name):
        self.Name = name

    def is_Lable(self):
        pass


class VideoMemo(Node):
    def __init__(self):
        self.vidPath
        self.vidName
        self.frameID
        self.objectSet
        self.motionBase
        

# =============================================================================
if __name__ =='__main__':
    _=0
    if not os.path.isdir(args['output']):
        os.makedirs(args['output'])

    for imgPath in gen_file_path(args['input']):
        img = imread(imgPath)
        # Standard Size
        img = cv2.resize(img, RESOLUTION , interpolation=cv2.INTER_LINEAR)
        for img in pyramid(img,PYRAMID_SCALE,(winHeight , winWeight)):
            for x,y,win in sliding_window(img, WinStep,(winHeight , winWeight)):
                # the input dimension is critical
                if win.shape[0] != winHeight or win.shape[1] != winWeight:
                    continue
                imsave('{}//{}.png'.format(args['output'],_), win)
                _+=1

# =============================================================================
# Example 
# python hard_negative_mining.py 
# -i ~/MIT_Vedio/2D_DataSet/Bg_v3 -o ~/MIT_Vedio/2D_DataSet/Bg_v3_3030





