import cv2
import os
import sys

def TrainFilePath(folderList, constrain=None, **kargs ):
    if constrain is None:
        constrain = ('avi', 'mp4')
    for basePath in folderList :
        for (rootDir, dirNames, fileNames) in os.walk(basePath):
            for fileName in fileNames:
                if fileName.split('.')[-1] in constrain:
                    yield os.path.join(rootDir, fileName), rootDir
