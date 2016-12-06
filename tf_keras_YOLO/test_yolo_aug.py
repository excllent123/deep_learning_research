from skimage.io import imread
from yolo_augment import imcv2_affine_trans, imcv2_recolor
import cv2
import numpy as np



img  = imread('../hub_data/vatic/vatic_id2_test/BG/561.png')
img.dtype
while True:
    cv2.imshow('Raw Image', img)
    cv2.imshow('Recolor', imcv2_recolor(img.copy()))
    img2, bb, cc = imcv2_affine_trans(img.copy())
    cv2.imshow('Affine' , img2)
    print bb, cc
    cv2.waitKey()





