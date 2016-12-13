
import os
import pandas as pd
from skimage.io import imread
from yolo_layer import YoloDetector
import cv2
import yolo_augment
#==============================================================================
# Author : Kent Chiu (kentchun33333@gmail.com)
# This is a script that manipulate the data from vatic-data to yolo setting
#==============================================================================

class VaticPreprocess(object):

    def __init__(self, fileName, maplist, detector=None):
        self.df = self.get_vatic_df(fileName)
        self.maplist = maplist
        if not detector:
            self.detector = YoloDetector(C=len(maplist), classMap=maplist)
        else :
            self.detector = detector

    def get_vatic_df(self, fileName):
        '''from vatic.txt, get df object'''
        with open(fileName,'r') as f :
            data = f.readlines()
        data2 = [i.split(' ') for i in data]
        col = ['track_id','xmin','ymin','xmax','ymax',
               'frameid','lost','occluded','generated','label_name']
        df = pd.DataFrame(data2,columns=col )
        return df

    def get_annotation(self, frameID, scale_factor=None):
        df = self.df ; maplist = self.maplist
        df_tmp = df[ df['frameid']==str(frameID) ]
        df_tmp = df_tmp[['xmin','ymin','xmax','ymax','label_name']]

        annotations = []
        for i_list in df_tmp.values:
            startX, startY, endX, endY , label_name = i_list
            label_name = label_name.replace('"','').split('\n')[0]

            # =================
            # It works

            classid = maplist.index(label_name)
            cY   = (int(startX)+int(endX))/2
            cX   = (int(startY)+int(endY))/2
            boxW = int(endX) - int(startX)
            boxH = int(endY) - int(startY)

            if scale_factor:
                assert len(scale_factor)==2
                scale_x, scale_y = scale_factor
                cX*=scale_y
                cY*=scale_x
                boxW*=scale_x
                boxH*=scale_y
            # It works
            # ======================

            annotations.append([classid, int(cX), int(cY), int(boxW), int(boxH)])
        return annotations

    def genYOLO_foler_batch(self, folder, batch_size=16):
        '''
        the img file in folder must only contain what we need  that is
        preprocess by vatic which is in the format like  : numbers.png
        return :
        X : 4D tensor (batch_zie, W, H, C)
        Y : 2D tensor (batch_zie, S*S*(5*B+C))
        '''

        df = self.df
        filelist = os.listdir(folder)

        batch_X = []
        batch_Y = []
        batch   = 1

        while filelist:
            filename = filelist.pop()
            if filename.split('.')[-1] not in ('png','jpg'):
                continue

            frameID = filename.split('.')[0]
            frame = imread(os.path.join(folder, filename))
            frame *= int(255.0/frame.max())

            # === This Section Shoud Be Refractoried ===

            h,w,c = frame.shape
            if w != 448 or h!=448:
                frame = cv2.resize(frame, (448, 448))
                scale_x = (448.0/w)
                scale_y = (448.0/h)
                annotations = self.get_annotation(frameID, scale_factor=(scale_x, scale_y))
            else:
                annotations = self.get_annotation(frameID)

            # Real Time Data Augmentation
            frame = yolo_augment.recolor(frame)
            frame, annotations = yolo_augment.affine_trains(frame, annotations)

            # === END ===

            y = self.detector.encode(annotations)

            batch_X.append(frame)
            batch_Y.append(y    )

            if batch % batch_size ==0 or len(filelist)==0 :
                result_X = batch_X
                result_Y = batch_Y

                batch_X  = []
                batch_Y  = []
                batch    = 1

                yield result_X, result_Y
            else :
                batch+=1



    def genYOLO_vid(self, vid):
        df = self.df
        for frameID in range(vid.get_length()):
            frame = vid.get_data(frameID)
            annotations = self.get_annotation(frameID)
            yield frame , annotations





if __name__ =='__main__':
    file_path = '../hub_data/vatic/vatic_id2/ann.txt'
    maplist = ['Rhand', 'ScrewDriver']
    yoloProcessor = VaticPreprocess(file_path, maplist=maplist)
    import cv2
    import numpy as np

    #B = yoloProcessor.genYOLO_vid(vid)
    C = yoloProcessor.genYOLO_foler_batch('../hub_data/vatic/vatic_id2')

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


#    import imageio
#    vid = imageio.get_reader('../data/vatic_id2/output.avi')
#    B = yoloProcessor.genYOLO_vid(vid)
#    for frame , annotations in B:
#        print annotations
#        cv2.imshow('ImageWindow',frame)
#        cv2.waitKey()
#        raw_input()

