
# =============================================================================
# Author : Kent Chiu
# =============================================================================
# Des.
# - This CNN_Agent Series Py is build on the Keras
#
# Usage
# - Provide Image-Structure foder list
# - Define User Mapping
# - Define Model
# - Define Model_Name
# -
# - Extrate background imgs to trainable format
# - Store them into the Output folder

import os, sys, math
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

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,RemoteMonitor
from keras.optimizers import Adam, RMSprop
from keras.models import model_from_json
import h5py




#==============================================================================
# Define Functions

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

def gen_FilePath01(folderPath, constrain=None, **kargs):
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

def gen_FilePath(folderPath, constrain=None, size_limit=1000000, **kargs):
    '''
    (1) Output filepath and calssName
    (2) folderPath
          --label_1
           -- xxx.jpg
    '''
    assert folderPath[-1]!='/'
    if constrain is None:
        constrain = ('avi', 'mp4','png','jpg','jpeg','bmp')
    for (rootDir, dirNames, fileNames) in os.walk(folderPath):
        count = 0
        for fileName in fileNames:
            if size_limit < count:
                break
                    #continue
            if fileName.split('.')[-1] in constrain:
                yield (os.path.join(rootDir, fileName))
                count+=1
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
        for imgPath in gen_FilePath(folderPath) :
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


def VGG_K00001(img_rows,img_cols,weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,img_rows,img_cols)))
    model.add(Convolution2D(32, 1, 1, border_mode='same', activation='relu'))
    model.add(Convolution2D(32, 1, 1, border_mode='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu'))
    model.add(Convolution2D(32, 2, 2, border_mode='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(48, 2, 2, border_mode='same', activation='relu'))
    model.add(Convolution2D(48, 2, 2, border_mode='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(48, 2, 2, border_mode='same', activation='relu'))
    model.add(Convolution2D(48, 2, 2, border_mode='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    try :
        model.load_weights(weights_path)
        print ('loaded weight from h5',weights_path)
    except:
        pass

    return model

#==============================================================================
# Set parameters
img_rows= 30

img_cols= 30

img_channels=3#==============================================================================
# Define Functions

kfoldNums = 10

model_name = __file__.split('\\')[-1].split('.')[0]

name_data_h5 = 'test_factory_cnn'
#==============================================================================
# Read Data
try :
    with h5py.File('../hub/image_{}.h5'.format(name_data_h5),'r') as f:
        Train_X = np.array(f.get('x'))
        Train_Y = np.array(f.get('y'))
    print ('loded from hdf5 data')
except Exception as err:
    print (str(err))

    folderList = ['D:\\2D_DataSet\\Bg_v4_3030',
    'D:\\2D_DataSet\\RHwithScrewDriver',
    'D:\\2D_DataSet\\RhandBoost']
    Train_X, Train_Y = load_training(folderList, img_rows, img_cols, img_channels)

# =============================================================================
# Call back
filepath="..\\hub\\model\\"+model_name+"-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacksList = [checkpoint]

#==============================================================================
# train_test_split

Train_X, X_test, Train_Y, y_test = train_test_split(Train_X, Train_Y, test_size=0.1, random_state=0)

print ('train and split')
train_X , train_Y = reshapeShuffle(Train_X, Train_Y, img_rows, img_cols, img_channels=img_channels)

X_test, y_test = reshapeShuffle(X_test, y_test, img_rows, img_cols, img_channels=img_channels)

#==============================================================================
# Data Aug.

gen_Img = ImageDataGenerator(featurewise_center=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False)

# Data Aug. Step2
gen_Img.fit(train_X)

#==============================================================================
# Define model

model = VGG_K00001(img_rows,img_cols,
                   weights_path='..\hub\model\Agent_20161024-29-0.99.h5')
#https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
# if damping, use smaller lr

#RMSprop(lr=0.001)
model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

model_json = model.to_json()
with open("../hub/model/{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)
print ('saving model struct as ' + "../hub/model/{}.json".format(model_name))
#==============================================================================
# Start Training
remote = RemoteMonitor(root='http://localhost:9000')


def f1_score(tags, predicted):

    tags = set(tags)
    predicted = set(predicted)

    tp = len(tags & predicted)
    fp = len(predicted) - tp
    fn = len(tags) - tp

    if tp>0:
        precision=float(tp)/(tp+fp)
        recall=float(tp)/(tp+fn)
        return 2*((precision*recall)/(precision+recall))
    else:
        return 0
# ===============================================================
# fits the model on batches with real-time data augmentation:

batchSize = 45
model.fit_generator(gen_Img.flow(train_X, train_Y, batch_size=batchSize),
    samples_per_epoch=len(train_X), nb_epoch=3225,
    validation_data=gen_Img.flow(X_test, y_test,batch_size=batchSize),
    nb_val_samples=X_test.shape[0],callbacks=callbacksList)
# serialize model to JSON





# =========================================
# Load the pretrain method then ...
# pop last layer, insert my own
# =========================================
# model.layers.pop()
# model.add(Dropout(0.5))
# model.add(Dense(len(Y_train[1])))
# model.add(Activation('softmax'))

'''
# Inception Module

input_img = Input(shape=(3, 256, 256))

tower_1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
tower_1 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(tower_1)

tower_2 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input_img)
tower_2 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_img)
tower_3 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(tower_3)

output = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1)`
'''