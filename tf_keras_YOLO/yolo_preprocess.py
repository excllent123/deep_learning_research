# Author : Kent Chiu
# Des : this module provide a tool for parsing label-data from vatic.txt
# with coressponding video, and provide batch-pumping for training model


import os, cv2
import pandas as pd
import numpy as np
from skimage.io import imread
from yolo_layer import YoloDetector
import yolo_augment
from random import shuffle


class VaticPreprocess(object):
    """
    # Description :  
      this module provide an interface for parsing label-data from vatic.txt 
      with coressponding video, and provide batch-pumping for training 
      SGD-based model especially for Deep Learning such as CNN/RNN.

      - this module is coupled by Yolo-detector and Vatic setting 
    """
    def __init__(self, fileName, mapList, detector=None, augmentation=True):
        '''
        Args : 
        fileName = vatic.txt file 
        mapList  = object mapping list in vatic setting
        detector = assigned the detector for 
        augmentation : Ture while training & False while evaluating 
        '''
        self.df = self._get_vatic_df(fileName)
        self.mapList = mapList
        if not detector:
            self.detector = YoloDetector(C=len(mapList), classMap=mapList)
        else :
            self.detector = detector
        self.batch_X=[]
        self.batch_Y=[]
        self.batch  = 1

    def _get_vatic_df(self, fileName):
        '''[return] panda object of vatic.txt'''
        with open(fileName,'r') as f :
            data = f.readlines()
        data2 = [i.split(' ') for i in data]
        col = ['track_id','xmin','ymin','xmax','ymax',
               'frameid','lost','occluded','generated','label_name']
        df = pd.DataFrame(data2,columns=col )
        return df

    def _get_annotation(self, frameid, scale_factor=None):
        '''
        Return : annotation by frameid in self.df
        Note : one frame may have multi-objs/annotations 
        '''
        df = self.df ; mapList = self.mapList
        df_tmp = df[ df['frameid']==str(frameid) ]
        df_tmp = df_tmp[['xmin','ymin','xmax','ymax','label_name']]

        annotations = []
        for i_list in df_tmp.values:
            startX, startY, endX, endY , label_name = i_list
            label_name = label_name.replace('"','').split('\n')[0]

            classid = mapList.index(label_name)
            
            cX   = (int(startX)+int(endX))/2
            cY   = (int(startY)+int(endY))/2
            boxW = int(endX) - int(startX)
            boxH = int(endY) - int(startY)

            if scale_factor:
                assert len(scale_factor)==2
                scale_x, scale_y = scale_factor
                cY*=scale_y
                cX*=scale_x
                boxW*=scale_x
                boxH*=scale_y

            annotations.append([classid, int(cX), int(cY), int(boxW), int(boxH)])
        return annotations

    def _get_training_pair(self, frame, framID):
        '''add single-training pair into batch'''
        frame *= int(255.0/frame.max())

        # reshape img size if labeled data is with raw-image-size != 448
        h,w,c = frame.shape
        if w != 448 or h!=448:
            frame = cv2.resize(frame, (448, 448))
            scale_x = (448.0/w) ; scale_y = (448.0/h)
            annotations = self._get_annotation(frameid, scale_factor=(scale_x, scale_y))
        else:
            annotations = self._get_annotation(frameid)

        if self.augmentation:
            frame = yolo_augment.recolor(frame)
            frame, annotations = yolo_augment.affine_trains(frame, annotations)

        y = self.detector.encode(annotations)
        self.batch_X.append(frame)
        self.batch_Y.append(y    )

    def genYOLO_foler_batch(self, folder, batch_size=16):
        '''
        # Description :
          - the img file in folder must only contain what we need
          - that is preprocess by vatic which is in the format like  : numbers.png

        # Return :
          - X : 4D tensor (batch_zie, W, H, C)
          - Y : 2D tensor (batch_zie, S*S*(5*B+C))
        '''
        filelist = os.listdir(folder)
        shuffle(filelist)

        while filelist:
            filename = filelist.pop()
            # check type
            if filename.split('.')[-1] not in ('png','jpg'):
                continue
            frameid = filename.split('.')[0]
            frame = imread(os.path.join(folder, filename))
 
            self._get_training_pair(frame, frameid)
            if self.batch % batch_size ==0 or len(filelist)==0 :
                result_X = self.batch_X
                result_Y = self.batch_Y
                self.batch_X  = []
                self.batch_Y  = []
                self.batch    = 1
                yield result_X, result_Y
            else :
                self.batch+=1

    def genYOLO_vid(self, vid, batch_size=16, max_num=10000):
        '''
        # Args : vid the video object from imageio 
        
        # Return : 
          - X : 4D tensor (batch_zie, W, H, C)
          - Y : 2D tensor (batch_zie, S*S*(5*B+C))
        '''
        for _ in range(max_num):
            frameid = np.random.randint(0, vid.get_length())
            frame = vid.get_data(frameid)
            self._get_training_pair(frame, frameid)
            if self.batch % batch_size ==0 :
                result_X = self.batch_X
                result_Y = self.batch_Y
                self.batch_X  = []
                self.batch_Y  = []
                self.batch    = 1
                yield result_X, result_Y
            else :
                self.batch+=1

# TEST
if __name__ =='__main__':
    file_path = '../hub_data/vatic/vatic_id2/ann.txt'
    mapList = ['Rhand', 'ScrewDriver']
    yoloProcessor = VaticPreprocess(file_path, mapList=mapList)
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