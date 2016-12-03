from yolo_preprocess import VaticPreprocess
import os
import pandas as pd
from skimage.io import imread
from yolo_layer import YoloDetector
import cv2




if __name__ =='__main__':
    file_path = '../data_test/vatic_example.txt'
    maplist = ['Rhand', 'ScrewDriver']
    yoloProcessor = VaticPreprocess(file_path, maplist=maplist)
    import cv2 
    import numpy as np 

    #B = yoloProcessor.genYOLO_vid(vid)
    C = yoloProcessor.genYOLO_foler_batch('../data/vatic_id2', batch_size=1)

    for x, y in C:
        x = np.asarray(x)
        y = np.asarray(y)
        print (x.shape , y.shape)
        img = x[0]
        output_tensor = y[0]
        bbx   =  yoloProcessor.detector.decode(output_tensor)

        img_copy = img.copy()
        for item in bbx: 
            name, cX,cY,w,h , _= item
            def check_50(x):
                if x < 50 :
                    x = 50 
                return x
            #cX,cY,w,h = map(check_50,[cX,cY,w,h] )
            pt1= ( int(cX-0.5*w) ,int(cY-0.5*h) )
            pt2= ( int(cX+0.5*w) ,int(cY+0.5*h) )    
            cv2.rectangle(img_copy, pt1, pt2, (255,255,255), thickness=2)

        cv2.imshow("Before",img)
        cv2.imshow("After", img_copy)
        cv2.waitKey()

        raw_input()