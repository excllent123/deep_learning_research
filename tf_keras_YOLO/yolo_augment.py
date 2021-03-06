import cv2
import numpy as np


def recolor(im, a = .2, b = 20, c = 1.5):
    t = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t) * 2. - 1.

    # random amplify each channel
    im = im * (1 + t * a)
    # random brightness
    im += np.random.uniform() * 2 * b - b
    # random contrast
    mx = 255. * (1 + a) + b
    up = np.random.uniform() * c
    im = np.power(im/mx, 1 + up)
    return np.array(im * 255., np.uint8)

def imcv2_affine_trans(im):
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    im = cv2.resize(im, (0,0), fx = scale, fy = scale)
    im = im[offy : (offy + h), offx : (offx + w)]
    flip = np.random.binomial(1, .5)
    if flip: im = cv2.flip(im, 1)
    return im, [w, h, c], [scale, [offx, offy], flip]

def linear_op(pt, scale, off):
    return max(int(pt*scale-off),0)

def affine_trains(im, ann):
    '''description : 
    input : image and ann where 
            ann : [[classid, int(cX), int(cY), int(boxW), int(boxH)]]
            due to the fact that there are multi-objects in single frame
    output: as input but with augmentation 

    '''

    # image operation
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    im = cv2.resize(im, (0,0), fx = scale, fy = scale)
    im = im[offy : (offy + h), offx : (offx + w)]
    flip = np.random.binomial(1, .5)
    if flip: im = cv2.flip(im, 1)
    
    # annotation operation
    new_ann = []
    for classid , cx, cy, W, H in ann :
        x_min, x_max = int(cx-0.5*H), int(cx+0.5*H)
        y_min, y_max = int(cy-0.5*W), int(cy+0.5*W)

        x_min, x_max = linear_op(x_min,scale,offx), linear_op(x_max,scale,offx)
        y_min, y_max = linear_op(y_min,scale,offx), linear_op(y_max,scale,offx)
        if flip: 
            y_max_ = y_max
            y_max = w- y_min 
            y_min = w- y_max_

        cX   = (int(x_min)+int(x_max))/2
        cY   = (int(y_min)+int(y_max))/2
        boxH = int(x_max) - int(x_min)
        boxW = int(y_max) - int(y_min)
        new_ann.append([classid, int(cX), int(cY), int(boxW), int(boxH)])
    return im, new_ann



if __name__=='__main__':
    '''self-level testing'''
    from skimage.io import imread
    from yolo_preprocess import VaticPreprocess

    while True:
        img  = imread('../hub_data/vatic/vatic_id2_test/BG/561.png')
        # cv2.imshow('Raw Image', img)
        # cv2.imshow('Recolor', imcv2_recolor(img.copy()))
        x_min, x_max = 20, 200
        y_min, y_max = 20, 200
        print ('Before : ', y_min, y_max, x_min, x_max)

        img_ref = img.copy()[y_min:y_max, x_min:x_max,:]

        img2, bb, cc = imcv2_affine_trans(recolor(img))


        cv2.imshow('Affine' , img2)
        cv2.imshow('Ref.' , img_ref)

        # *scale - off
        scale = cc[0]
        offx, offy = cc[1]
        x_min, x_max = linear_op(x_min,scale,offx), linear_op(x_max,scale,offx)
        y_min, y_max = linear_op(y_min,scale,offx), linear_op(y_max,scale,offx)

        

        if cc[2]==1:
            x_max_ = x_max
            x_max = bb[0]- x_min 
            x_min = bb[0]- x_max_ 

        x_min= max(int(x_min),0)
        x_max= max(int(x_max),0)
        y_min= max(int(y_min),0)
        y_max= max(int(y_max),0)

        img_veri = img2[y_min:y_max, x_min:x_max,:]
        print ('AFTER : ', y_min, y_max, x_min, x_max)
        cv2.imshow('Veri.', img_veri)

        cv2.waitKey()
