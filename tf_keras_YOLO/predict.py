
from yolo_layer import YoloDetector
from yolo_preprocess import VaticPreprocess
from yolo_cnn import YoloNetwork

import numpy as np 
import imageio , cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,RemoteMonitor
from keras.optimizers import Adam, RMSprop
from keras.models import model_from_json

from keras import backend as K


model_name = __file__.split('\\')[-1].split('.')[0]
file_path = '../data_test/vatic_example.txt'
maplist = ['Rhand', 'ScrewDriver']




vid = imageio.get_reader('../data/vatic_id2/output.avi')
img = vid.get_data(2)
img = cv2.resize(img, (448,448))
img = img.reshape(1,448,448,3)


yoloProcessor = VaticPreprocess(file_path, maplist=maplist)


A = YoloNetwork(numCla=2)

C = yoloProcessor.genYOLO_foler_batch('../data/vatic_id2', batch_size=50)

model = A.yolo_tiny()

model.load_weights('sct-run-20161111-46-0.02.h5')

s = model.predict_proba(img)

print yoloProcessor.detector.decode(s[0])



   # print (loaded_model.predict_proba(TrX[x].reshape(1,img_channels,img_rows,img_rows)) )
