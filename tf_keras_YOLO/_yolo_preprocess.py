# Author : Kent Chiu

import os, cv2
import pandas as pd
from skimage.io import imread
from _yolo_layer import YoloDetector
import yolo_augment
from random import shuffle

class DataPreprocess(object):
    def gen_from_foler_batch(self):
        pass

class VaticPreprocess(object):
    """description : 
    this module provide an interface for parsing label-data from vatic.txt 
    with coressponding video, and provide batch-pumping for training SGD-based model 
    especially for Deep Learning such as CNN/RNN.
    """
    def __init__(self, fileName, mapList, detector=None):
        '''
        input :
        fileName = vatic.txt file 
        mapList  = object mapping list in vatic setting
        detector = assigned the detector for 
        '''
        self.df = self.get_vatic_df(fileName)
        self.mapList = mapList
        if not detector:
            self.detector = YoloDetector(C=len(mapList), classMap=mapList)
        else :
            self.detector = detector

    def get_vatic_df(self, fileName):
        '''return : panda object of vatic.txt'''
        with open(fileName,'r') as f :
            data = f.readlines()
        data2 = [i.split(' ') for i in data]
        col = ['track_id','xmin','ymin','xmax','ymax',
               'frameid','lost','occluded','generated','label_name']
        df = pd.DataFrame(data2,columns=col )
        return df

    def get_annotation(self, frameID, scale_factor=None):
        '''return : annotation by frameID'''
        df = self.df ; mapList = self.mapList
        df_tmp = df[ df['frameid']==str(frameID) ]
        df_tmp = df_tmp[['xmin','ymin','xmax','ymax','label_name']]

        annotations = []
        for i_list in df_tmp.values:
            startX, startY, endX, endY , label_name = i_list
            label_name = label_name.replace('"','').split('\n')[0]

            classid = mapList.index(label_name)
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

            annotations.append([classid, int(cX), int(cY), int(boxW), int(boxH)])
        return annotations

    def genYOLO_foler_batch(self, folder, batch_size=16):
        '''
        description :
        the img file in folder must only contain what we need  that is
        preprocess by vatic which is in the format like  : numbers.png
        return : 
        X : 4D tensor (batch_zie, W, H, C)
        Y : 2D tensor (batch_zie, S*S*(5*B+C))
        '''
        df = self.df
        filelist = os.listdir(folder)
        shuffle(filelist)

        # init
        batch_X = []
        batch_Y = []
        batch   = 1
        while filelist:
            filename = filelist.pop()

            # check type
            if filename.split('.')[-1] not in ('png','jpg'):
                continue

            frameID = filename.split('.')[0]
            frame = imread(os.path.join(folder, filename))
            frame *= int(255.0/frame.max())

            # reshape img size
            h,w,c = frame.shape
            if w != 448 or h!=448:
                frame = cv2.resize(frame, (448, 448))
                scale_x = (448.0/w)
                scale_y = (448.0/h)
                annotations = self.get_annotation(frameID, scale_factor=(scale_x, scale_y))
            else:
                annotations = self.get_annotation(frameID)

            # in-Pipe data augmentation
            frame = yolo_augment.recolor(frame)
            frame, annotations = yolo_augment.affine_trains(frame, annotations)

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
        '''
        input : vid the video object from imageio 
        return : 
        X : 4D tensor (batch_zie, W, H, C)
        Y : 2D tensor (batch_zie, S*S*(5*B+C))
        '''
        df = self.df
        for frameID in range(vid.get_length()):
            frame = vid.get_data(frameID)
            annotations = self.get_annotation(frameID)
            yield frame , annotations