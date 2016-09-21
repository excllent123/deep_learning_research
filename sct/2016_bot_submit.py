
import os, sys, glob
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

mapping_dict = {}

a = [6,7,4,3,8,0,10,9,2,1,11,5]
for i in range(12):
    mapping_dict[i]=a[i]





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
        constrain = ('avi', 'mp4','png','jpg','jpeg','jepg')
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
        img = img[:,:,:3]
    return img

def load_training(folderList, img_rows, img_cols, img_channels):
    TrY = []
    TrX = []
    TrY_template = np.eye(len(folderList))
    for eyeId, folderPath in enumerate(folderList):
        for imgPath in TrainFilePath(folderPath) :
            TrY.append(TrY_template[eyeId])
            TrX.append(genTrX(imgPath, (img_rows,img_cols), img_channels))
    print (len(TrX))
    return TrX, TrY

def create_folderList(rootDir):
    result=[]
    for a in os.listdir(rootDir):
        a = os.path.join(rootDir, a)
        if os.path.isdir(a):
            result.append(a)
    return result


def reshapeShuffle(TrX, TrY, img_rows, img_cols, img_channels):
    trainX = np.asarray(TrX, dtype = np.uint8)
    trainX = trainX.reshape(trainX.shape[0], img_channels, img_rows, img_cols)
    trainX = trainX.astype('float32')
    trainY = np.asarray(TrY, dtype = np.float32)
    # shuffle
    trainX , trainY = shuffle(trainX,trainY)
    print ('Train_X : ',trainX.shape,'Train_Y' ,trainY.shape)
    return trainX , trainY

def easyVGG(img_rows,img_cols,weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,img_rows,img_cols)))
    model.add(Convolution2D(32, 3, 3))
    model.add(MaxPooling2D((3,3), strides=(2,2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D((3,3), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1000, activation='tanh'))
    model.add(Dense(200, activation='softmax'))
    model.add(Dense(12, activation='softmax'))
    if weights_path:
        model.load_weights(weights_path)
        print ('loaded weight')

    return model
#


# Read Data and Set parameters

ROOT_Dir = 'D:\\2016bot_cv'
Test_Dir = 'C:\\Users\\kentc\\Downloads\\Testset 5'


img_rows= 64

img_cols= 64

img_channels=3

# folderList = create_folderList(ROOT_Dir)
# print (folderList)


########################################################################
model = easyVGG(img_rows,img_cols,'../hub/model/2016bot_0001.h5')
#https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',metrics=['accuracy'])
#########################################################################
with open('output5.txt','w') as f:
    #f.write('{} {} {} {} {}').format(imgPath.split('.')[0], )
    for imgPath in TrainFilePath(Test_Dir):
        print imgPath
        image = genTrX(imgPath, (img_rows,img_cols), img_channels)
        image = np.asarray(image, dtype = np.uint8)
        image = image.reshape(1, img_channels, img_rows, img_cols)
        image = image.astype('float32')

        predictProba = model.predict_proba(image)[0]


        top2_ALL = predictProba.argsort()[-2:][::-1]
        top1 = top2_ALL[0]
        top1_p = round(predictProba[top1],6)
        top2 = top2_ALL[1]
        top2_p = round(predictProba[top2],6)
        top1_l = mapping_dict[top1]
        top2_l = mapping_dict[top2]
        imgName = imgPath.split('\\')[-1].split('.')[0]
        f.write('{} {} {} {} {}\n'.format(imgName,
                                         top1_l,
                                         top1_p,
                                         top2_l,
                                         top2_p))
print ('finished ')


