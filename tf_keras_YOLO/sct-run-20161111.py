
from yolo_layer import YoloDetector
from yolo_preprocess import VaticPreprocess
from yolo_cnn import YoloNetwork

import numpy as np 

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,RemoteMonitor
from keras.optimizers import Adam, RMSprop
from keras.models import model_from_json

from keras import backend as K

def squared_hinge(y_true, y_pred):
    print (K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1))
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)

def lossA(y_true, y_pred):
    S, B, C, W, H = 7, 2, 2, 448, 448
    COORD=5.
    NOOBJ=.5 

    loss_ALL= []


    truYs = np.asarray(y_true)
    preYs = np.asarray(y_pred)

    for truY, preY in truYs, preYs:
        loss_ = 0
        truCP ,truConf, truB = self.traDim(truY, mode=2)
        preCP ,preConf, preB = self.traDim(preY, mode=2)    

        # Select for responsible box which with max IOU
        iouT = self.iouTensor(truB,preB)           # iouT (7*7,2)
        iouT = np.argmax(iouT, axis=1).astype(int) # (7*7)    

        truB    = np.array([truB[i,j,:]  for i,j in enumerate(iouT)])
        preB    = np.array([preB[i,j,:]  for i,j in enumerate(iouT)])
        truConf = np.array([truConf[i,j] for i,j in enumerate(iouT)])
        preConf = np.array([preConf[i,j] for i,j in enumerate(iouT)])    

        # Obj or noobj is actually only depend on truth
        objMask  = np.array([ max(i) for i in truCP])
        nobjMask = 1 - np.array([ max(i) for i in truCP])    

        loss_ += sum(pow( (truB-preB), 2).sum(axis=1)    * objMask  ) * COORD 
        loss_ += sum(pow( (truConf- preConf) , 2)        * objMask  )
        loss_ += sum(pow( (truConf- preConf), 2)         * nobjMask ) * NOOBJ
        loss_ += sum(pow( (truCP- preCP), 2).sum(axis=1) * objMask  )
        loss_ALL.append([loss_])
    return np.asarray(loss_ALL)

model_name = __file__.split('\\')[-1].split('.')[0]
file_path = '../data_test/vatic_example.txt'
maplist = ['Rhand', 'ScrewDriver']


filepath= model_name+"-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacksList = [checkpoint]


gen_Img = ImageDataGenerator(featurewise_center=False,
        rotation_range=0,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        horizontal_flip=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False)


yoloProcessor = VaticPreprocess(file_path, maplist=maplist)


A = YoloNetwork(numCla=2)

C = yoloProcessor.genYOLO_foler_batch('../data/vatic_id2', batch_size=50)

model = A.yolo_tiny()

optimizer=RMSprop(lr=0.001)
updates = optimizer.get_updates(model.trainable_weights, model.constraints, lossA)
#model.compile(optimizer=RMSprop(lr=0.001),
#              loss=lossA,metrics=['acc'])


idd = 0
for x, y in C:

    x = np.asarray(x)
    y = np.asarray(y)
    if idd ==0 :
        gen_Img.fit(x)
    idd+=1
    model.fit_generator(gen_Img.flow(x, y, batch_size=8),
        samples_per_epoch=len(x), nb_epoch=3000,

        validation_data=gen_Img.flow(x, y,batch_size=8),
        nb_val_samples=x.shape[0],callbacks=callbacksList)

# =============================================================================
# The loss function is not compatible with keras current compile method 
# Therefore, I am going to seek a more fundermental approach by mixed-type 
# keras/tensorflow 
