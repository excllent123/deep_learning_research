
import os, sys, codecs
import numpy as np
import cv2
from skimage.io import imread

from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
# Data Augmentation
from keras.preprocessing.image import ImageDataGenerator

# model build
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.optimizers import Adam

from keras.models import model_from_json
'''
0 cat : 6
1 chipmunk : 7
2 dog : 4
3 fox : 3
4 giraffe : 8
5 guinea pigs 0
6 hyena : 10
7 reindeer : 9
8 sika deer : 2
9 squirrel : 1
10 weasel : 11
11 wolf : 5
'''

class ModelAverage():
    def __init__(self):
        self.model=[]
        self.proba=None

    def add_model(self, jsonPath, weightPath):
        # model have to be loaded weight
        with open(jsonPath, 'r') as f:
            loaded_model_json = f.read()


        # struc model
        model = model_from_json(loaded_model_json)
        # loading model weight
        model.load_weights(weightPath)
        # add model
        self.model.append(model)

    def predict_proba(self, iuPut):
        for model in self.model:
            predictProba = model.predict_proba(iuPut)[0]
            if self.proba is None:
                self.proba=predictProba
            else:
                self.proba+=predictProba
        result = self.proba/float(len(self.model))
        self.proba=None
        return [result]

def auto_resized(img,size):
    '''size = (width,height)'''
    size = tuple(size)
    resize_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return resize_img

def TrainFilePath(folderPath, constrain=None, **kargs):
    '''
    (1) Output filepath and calssName
    (2) folderPath
          --label_1
           -- xxx.jpg
    '''
    assert folderPath[-1]!='/'
    if constrain is None:
        constrain = ('avi', 'mp4','png','jpg','jpeg','jepg','gif')
    for (rootDir, dirNames, fileNames) in os.walk(folderPath):
        for fileName in fileNames:
            if fileName.split('.')[-1] in constrain:
                yield (os.path.join(rootDir, fileName))

#img_channels = 3
def genTrX(filePath, resolution, img_channels=3):
    assert type(resolution) == tuple
    img = auto_resized(imread(filePath),resolution)  #conf['sliding_size']
    if img_channels==1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_channels==3:
        try:
            img = img[:,:,:3]
        except:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

# Create mapping list
mapping_dict = {}
a = [6,7,4,3,8,0,10,9,2,1,11,5]
for i in range(12):
    mapping_dict[i]=a[i]

#----------------------------------------------------------------
# Setting Testing data source
Test_Dir = 'C:\\Users\\kentc\\Downloads\\Testset7'

# Setting output file name
output_file = '2016_bot_cv_test7_Nor_ensemble.txt'
testing_file = 'D:\\2016bot_cv\\Dog'
# The image preprocess should be the same
img_rows= 92
img_cols= 92
img_channels=3
samplewise_center=True
samplewise_std_normalization=True

# Setting model path and load
model = ModelAverage()
model.add_model(jsonPath = '..\\hub\\model\\2016_bot_005.json',
                weightPath= '..\\hub\\model\\2016_bot_0050.h5')
model.add_model(jsonPath = '..\\hub\\model\\2016_bot_006.json',
                weightPath= '..\\hub\\model\\2016_bot_0062.h5')

#-----------------------------------------------------------
# process Unicode text

with codecs.open(output_file,'w',encoding='utf-8') as f:
    #f.write('{} {} {} {} {}').format(imgPath.split('.')[0], )
    for imgPath in TrainFilePath(Test_Dir):
        print imgPath
        image = genTrX(imgPath, (img_rows,img_cols), img_channels)
        image = np.asarray(image, dtype = np.uint8)

        image = image.astype('float32')
        # ====================================================
        # Since the training process contrain the augmentation
        if samplewise_center:
            image -= np.mean(image, axis=0, keepdims=True) # for theano
        if samplewise_std_normalization:
            image /= (np.std(image, axis=0, keepdims=True) + 1e-7) # for theano

        image = image.reshape(1, img_channels, img_rows, img_cols)

        predictProba = model.predict_proba(image)[0]


        top2_ALL = predictProba.argsort()[-2:][::-1]
        top1 = top2_ALL[0]
        top1_p = round(predictProba[top1],6)
        top2 = top2_ALL[1]
        top2_p = round(predictProba[top2],6)
        top1_l = mapping_dict[top1]
        top2_l = mapping_dict[top2]
        imgName = imgPath.split('\\')[-1].split('.')[0]
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(imgName,
                                         top1_l,
                                         top1_p,
                                         top2_l,
                                         top2_p))
print ('finished ')
