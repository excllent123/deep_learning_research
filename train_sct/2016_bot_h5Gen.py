import os
import numpy as np
import cv2
from skimage.io import imread
from sklearn.utils import shuffle


ROOT_Dir = 'D:\\2016bot_cv'

img_rows= 92

img_cols= 92

img_channels=3

def auto_resized(img,size):
    '''size = (width,height)'''
    size = tuple(size)
    resize_img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return resize_img

def gen_FilePath(folderPath, constrain=None, size_limit=None, **kargs):
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
            if size_limit is not None :
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
        for imgPath in gen_FilePath(folderPath, size_limit=6000) :
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

#


# Read Data and Set parameters



folderList = create_folderList(ROOT_Dir)
print (folderList)

Train_X, Train_Y = load_training(folderList, img_rows, img_cols, img_channels)


print ('loaded all data ')
import h5py
with h5py.File('..\\hub\\image_{}.h5'.format(img_rows),'w') as f:
    f.create_dataset('x', data = Train_X)
    f.create_dataset('y', data = Train_Y)

print (' data HDF5')
'''
#with h5py.File('image_{}.h5'.format(img_rows),'r') as f:
#    train_X = np.array(f.get('x'))
#    train_Y = np.array(f.get('y'))
try :
    with h5py.File('image_{}.h5'.format(img_rows),'r') as f:
        train_X = np.array(f.get('x'))
        train_Y = np.array(f.get('y'))

except Exception:
except ZeroDivisionError as err:
    >>> try:
...     raise Exception('spam', 'eggs')
... except Exception as inst:
...     print(type(inst))    # the exception instance
...     print(inst.args)     # arguments stored in .args
...     print(inst)          # __str__ allows args to be printed directly,
...                          # but may be overridden in exception subclasses
...     x, y = inst.args     # unpack args
...     print('x =', x)
...     print('y =', y)
'''
