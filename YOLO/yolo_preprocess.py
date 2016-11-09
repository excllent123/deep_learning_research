


import os
import pandas as pd
from skimage.io import imread

class YoloPreprocess(object):
    
    def __init__(self, fileName, maplist):
        self.df = self.get_vatic_data(fileName)
        self.maplist = maplist

    def get_vatic_df(self, fileName):
        '''from vatic.txt, get df object'''
        with open(fileName,'r') as f :
            data = f.readlines()
        data2 = [i.split(' ') for i in data]
        col = ['track_id','xmin','ymin','xmax','ymax',
               'frameid','lost','occluded','generated','label_name']
        df = pd.DataFrame(data2,columns=col )
        return df

    def get_annotation(self, frameID):
        df = self.df ; maplist = self.maplist
        df_tmp = df[ df['frameid']==str(frameID) ]
        df_tmp = df_tmp[['xmin','ymin','xmax','ymax','label_name']]

        annotations = []
        for i_list in df_tmp.values:
            startX, startY, endX, endY , label_name = map(str,i_list)

            classid = maplist.index(label_name)
            cX   = (int(startX)+int(endX))/2
            cY   = (int(startY)+int(endY))/2
            boxW = int(endX) - int(startX)
            boxH = int(endY) - int(startY)    
            annotations.append([classid, cX, cY, boxW, boxH])  
        return annotations  

    def genYOLO_foler(self, folder):
        '''
        the img file in folder must only contain what we need  that is 
        preprocess by vatic which is in the format like  : numbers.png
        '''
        df = self.df
        filelist = os.listdir(folder)
        for filename in filelist:
            if filename.split('.')[-1] is not in ('png','jpg'):
                continue 
            frameID = filename.split('.')[0]
            annotations = self.get_annotation(frameID)
            frame = imread(os.path.join(folder, filename))
            yield frame , annotations


    def genYOLO_vid(self, vid):
        df = self.df
        for frameID in range(vid.get_length()):
            frame = vid.get_data(frameID)
            annotations = self.get_annotation(frameID)
            yield frame , annotations




