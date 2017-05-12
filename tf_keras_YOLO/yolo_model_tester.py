import tensorflow as tf
import keras.backend as K
from keras.layers import Input
from keras.models import model_from_json

from yolo_layer import YoloDetector
from yolo_preprocess import VaticPreprocess
import imageio, cv2, argparse
import numpy as np

import relat_import


class YoloModelTestor:
    '''
    This class provide the yolo-model-testor by yolo-config-file 

    Usage: 
    - EX : 
    ```python
    Testor = YoloDetector(config)
    Testor.run_show(*arg, *kwarg)
    ```
    '''

    def __init__(self,json_file):
        W=448; H=448; S=7 ; B=2; C=2
        self.model = self.get_model(jsonPath=json_file)

        self.detector = YoloDetector(C=C)
        self.detector.set_class_map(['Rhand', 'ScrewDriver'])

    def get_test_img(self, img, W=448, H=448):
        '''resize the img for detector
        '''
        h,w,c = img.shape
        if h!=H or w!=W:
            img = cv2.resize(img, (H, W))
        img = np.resize(img,[1,H,W,c])
        img *= int(255.0/img.max())
        return img

    # load trained model
    def get_model(self, jsonPath):
        with open(jsonPath, 'r') as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        return model

    @staticmethod
    def run_show(vid, start_frame, end_frame):
        pass

    @staticmethod
    def run_save(vid, start_frame, end_frame):
        pass

    def inference():
        pass



# =====================================
# Alternative way
# img = vid.get_data(frameid)
# img = cv2.resize(img, (H, W))
# test_img = get_test_img(img)
# print (TFmodel.predict(test_img)).shape

TFmodel = get_model(jsonPath=json_file)
input_tensor = Input(shape=(H, W, 3))
pred_y = TFmodel(input_tensor)
init = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init)

    TFmodel.load_weights(weight_file)
    while frameid<vid.get_length():
        img = vid.get_data(frameid)
        img = cv2.resize(img, (H, W))
        test_img = get_test_img(img)

        output_tensor = sess.run(pred_y, feed_dict =
                {input_tensor : test_img, K.learning_phase(): 0})

        bbx = yolo_detect.decode(output_tensor[0,:], threshold=threshold)
        print (bbx)

        img_copy = img.copy()
        for item in bbx:
            name, cX,cY,w,h , _= item
            # Shape filter
            if w > 0.4*W or h > 0.4*H or w < 35 or h < 35 or w > 2.5*h or h>2.5*w:
                continue
            #cX,cY,w,h = map(check_50,[cX,cY,w,h] )
            pt1= ( int(cX-0.5*w) ,int(cY-0.5*h) )
            pt2= ( int(cX+0.5*w) ,int(cY+0.5*h) )
            cv2.rectangle(img_copy, pt1, pt2, (255,255,255), thickness=2)
        # print img.shape
        #cv2.imshow("Before",img)
        cv2.imshow("After", img_copy)
        cv2.waitKey()
        frameid+=5

# initialize the parameters
if __name__=='__main__':
    # experiment parameters

    # get the cli parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--start_frame', type=int, default =1)
    parser.add_argument('-e', '--end_frame',   type = int, default = 2)
    parser.add_argument('-t', '--threshold',   type=float, default = .2)
    parser.add_argument('-w', '--weight_file', type=str)
    parser.add_argument('-j', '--json_file',   type=str)
    parser.add_argument('-v', '--vid_path',    type=str, required=True,
                         help=' the input file path with ')
    parser.add_argument('-o', '--outPut',      type=str)
    arg=parser.parse_args()

    # init cli parameters
    threshold   = arg.threshold
    weight_file = arg.weight_file if arg.weight_file else 'tf-keras-20161125-v7.h5'
    json_file   = arg.json_file if arg.json_file else '../hub/model/tf-keras-20161120.json'
    vid_path    = arg.vid_path if arg.vid_path else '../hub_data/vatic/vatic_id2/output.avi'
    vid  = imageio.get_reader(vid_path)
    start_frame = arg.start_frame if arg.start_frame< vid.get_length else 1
    end_frame = arg.end_frame if arg.end_frame < vid.get_length else vid..get_length




