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

class Conf:
    def __init__(self, confPath):
        # load and store the configuration and update the object's dictionary
        conf = json.loads(open(confPath).read())
        self.__dict__.update(conf)

    def __getitem__(self, k):
        # return the value associated with the supplied key
        return self.__dict__.get(k, None)


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
        constrain = ('avi', 'mp4','png','jpg','jpeg')
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

img_rows= 64

img_cols= 64

img_channels=3

folderList = create_folderList(ROOT_Dir)
print (folderList)


########################################################################
model = easyVGG(img_rows,img_cols,'../hub/model/2016bot_0001.h5'
)
#https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',metrics=['accuracy'])
#########################################################################

Train_X, Train_Y = load_training(folderList, img_rows, img_cols, img_channels)

model_name = '2016bot_0001'

train_X , train_Y = reshapeShuffle(Train_X, Train_Y, img_rows, img_cols, img_channels=img_channels)

# Data Aug.
gen_Img = ImageDataGenerator(featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=20.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

# Data Aug. Step2
gen_Img.fit(train_X)






# Start Training
kf = KFold(len(train_Y), n_folds=5)
for train, test in kf:
    #print (train, test)
    Tr_X = train_X[train]
    Te_X = train_X[test]
    Tr_Y = train_Y[train]
    Te_Y = train_Y[test]
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(gen_Img.flow(Tr_X, Tr_Y, batch_size=20),
                    samples_per_epoch=len(Tr_X), nb_epoch=50, validation_data=(Te_X, Te_Y))
    model.save_weights('../hub/model/{}.h5'.format(model_name),overwrite=True)
    print ('saving model weight as ' + '../hub/model/{}.h5'.format(model_name))
    # serialize model to JSON
    model_json = model.to_json()
    with open("../hub/model/{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)
    print ('saving model struct as ' + "../hub/model/{}.json".format(model_name))
