import tensorflow as tf
import keras.backend as K
from keras.layers import Input
from keras.models import model_from_json

from yolo_layer import YoloDetector
from yolo_preprocess import VaticPreprocess
import imageio, cv2, argparse
from skimage.io import imread
import numpy as np

import relat_import


class YoloModelTestor:
    def __init__(self, conf):

        self.W=448; self.H=448; 
        S=7 ; B=2; C=2

        self.threshold = conf.threshold

        assert json_file.split('.')[-1]=='json'
        assert weight_file.split('.')[-1]=='h5'

        self.model = self._get_model(conf.json_file, conf.weight_file)
        self.detector = YoloDetector(C=C)

        self.detector.set_class_map(['Rhand', 'ScrewDriver'])

    def _trans_test_img(self, img):
        '''resize & normalize % reshape img for detector '''
        h,w,c = img.shape
        if h!=H or w!=W:
            img = cv2.resize(img, (self.H, self.W))
        img = np.resize(img,[1,H,W,c])
        img *= int(255.0/img.max())
        return img

    # load trained model
    def _get_model(self, json_file, weight_file):
        with open(jsonPath, 'r') as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weight_file)
        return model

    def _show_img(self, bbx, img):
        if h!=self.H or w!=self.W:
            img = cv2.resize(img, (H, W))

        for item in bbx:
            name, cX,cY,w,h , _= item
            # Shape filter 
            if w > 0.4*W or h > 0.4*H or w < 35 or h < 35 or w > 2.5*h or h>2.5*w:
                continue
            #cX,cY,w,h = map(check_50,[cX,cY,w,h] )
            pt1= ( int(cX-0.5*w) ,int(cY-0.5*h) )
            pt2= ( int(cX+0.5*w) ,int(cY+0.5*h) )
            cv2.rectangle(img, pt1, pt2, (255,255,255), thickness=2)
        return img

    @staticmethod
    def vid_inference(vid_path, start_frame, end_frame, save_path=None):
        '''tf-way'''
        vid  = imageio.get_reader(vid_path)
        input_tensor = Input(shape=(H, W, 3))
        pred_y = self.model(input_tensor)


        with tf.Session() as sess :
            # tf.global_variables_initializer()
            sess.run()
            self.model.load_weights()

            while start_frame<vid.get_length():
                img = vid.get_data(start_frame)
                
                test_img = self._trans_test_img(img)        

                output_tensor = sess.run(pred_y, feed_dict =
                        {input_tensor : test_img, K.learning_phase(): 0})        

                bbx = yolo_detect.decode(output_tensor[0,:], threshold=self.threshold)
                print (bbx)        

                img_show = self._show_img(img)
                cv2.imshow("Before",img)
                cv2.imshow("After", img_show)
                cv2.waitKey()
                start_frame+=5

            if save_path:
                raise NotImplementedError()

    @staticmethod
    def run_img(img, save_path=None):
        '''keras-way'''
        trans_img = self._trans_test_img(img)
        output_tensor = self.model.predict(trans_img)
        bbx = self.detector.decode(output_tensor[0,:], threshold=threshold)

        img_show = self._show_img(img)

        if save_path:
            raise NotImplementedError()


# initialize the parameters
if __name__=='__main__':
    # experiment parameters

    # get the cli parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--threshold',   type=float, default = .2)
    parser.add_argument('-w', '--weight_file', type=str)
    parser.add_argument('-j', '--json_file',   type=str)


    parser.add_argument('-v', '--vid_path',    type=str, 
                         help=' the input vid file path')
    parser.add_argument('-f', '--start_frame', type=int, default =1)
    parser.add_argument('-e', '--end_frame',   type = int, default = 2)

    parser.add_argument('-m', '--img_path', type=str, 
                         help=' the image img file path')
    parser.add_argument('-o', '--outPut',      type=str, 
                         help='if specify output address, would save img to it')

    arg=parser.parse_args()

    # init cli parameters
    threshold   = arg.threshold
    weight_file = arg.weight_file if arg.weight_file else 'tf-keras-20161125-v7.h5'
    json_file   = arg.json_file if arg.json_file else '../hub/model/tf-keras-20161120.json'

    vid_path    = arg.vid_path if arg.vid_path else '../hub_data/vatic/vatic_id2/output.avi'    
    
    start_frame = arg.start_frame if arg.start_frame< vid.get_length else 1
    end_frame = arg.end_frame if arg.end_frame < vid.get_length else vid..get_length




