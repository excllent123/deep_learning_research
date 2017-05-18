'''
Test Multi-processing Pool Map function

Valid that O.method could be pickled if in the top-level of function 

Testing Result List Comprehension is faster (why ?)
'''


import unittest as ut 

import os, sys
from batch_generator import ImgOneTagGenerator

from ops_augmentator import ImgAugmentator

from datetime import datetime

from multiprocessing import cpu_count, Pool


import multiprocessing
import types
import random

from skimage.io import imread

def process(img):
    O = ImgAugmentator()
    img = O.resize(img, size=(10,10))
    img = O.normaliza(img)
    return img


def get_path(dir_path_list):
    res = []
    for dir_path in dir_path_list:
        res.append([os.path.join(dir_path, s) \
                                  for s in os.listdir(dir_path) \
                                  if s.split('.')[-1] in ('jpg', 'png')])
    return res


def get_sample(fpath_by_cls):
    select = random.sample(range(0,len(fpath_by_cls)), 30)
    return select 


if __name__ == '__main__':

    print ('CPU : ', cpu_count())

    O = ImgAugmentator()

    dir_list = ['/Users/kentchiu/MIT_Vedio/2D_DataSet/RHand','/Users/kentchiu/MIT_Vedio/2D_DataSet/Rhand_v2']

    image_all_path = get_path(dir_list)

    imgs = [ imread(image_all_path[0][j]) for  j in get_sample(image_all_path[0]) ]

    ###
    start = datetime.now()

    res = [ process(img) for img in imgs ]
    print (datetime.now()-start)


    start = datetime.now()
    po = Pool(cpu_count())

    res = po.map(process, imgs)
    print (datetime.now()-start)



