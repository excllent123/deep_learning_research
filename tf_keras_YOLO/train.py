'''
Author : Kent Chiu

This is a train.py for yolo-detection 


'''


import relat_import

import tensorflow as tf
import keras.backend as K
import numpy as np

from keras.layers import Input

from tf_keras_YOLO.yolo_layer import YoloDetector
from tf_keras_YOLO.yolo_cnn import YoloNetwork
from tf_keras_YOLO.yolo_preprocess import VaticPreprocess
from tf_keras_YOLO.tf_keras_board import get_summary_op


flags = tf.flags
flags.DEFINE_float('learning_rate', 1e-7, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')

FLAGS = flags.FLAGS




class YoloTrainer():
    def __init__(self, config):
        pass

    def _get_config(self):
        pass

    def build_model(self):
        pass 

    def train(self, **callback):    
        '''
        train : 
        gradient-clip

        save_only_best
        
        '''
        with tf.Session() as sess :
            sess.run(init)
            writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
            # test mode
            epoch = 1
            while epoch < epoch_size:
                SUM_LOSS= 0
                if epoch == 1:
                    try:
                        model.load_weights('../hub_model/{}-v2.h5'.format(model_name))
                    except :
                        pass
                else :
                    pass
                step = 1
                DATAFLOW = processer.genYOLO_foler_batch('../hub_data/vatic/vatic_id2', batch_size=batch_size)
                for images_feed, labels_feed in DATAFLOW :
                    images_feed = batch_check(np.asarray(images_feed), batch_size)
                    labels_feed = batch_check(np.asarray(labels_feed), batch_size)        

                    _, lossN, summary_log = sess.run([train_step,loss,summary_op], feed_dict =
                        {input_tensor : images_feed, true_y :labels_feed, K.learning_phase(): 0})    
            

                    SUM_LOSS+=lossN
                    writer.add_summary(summary_log, epoch*step)
                    step+=1    
            

                SUM_LOSS = SUM_LOSS/(batch_size*step)    
            

                print ('EP [{}] Iter {} Loss {}'.format(epoch,step,SUM_LOSS ))        

                # loss clip / grdient clip
                MIN_LOSS = min(SUM_LOSS, MIN_LOSS)
                if SUM_LOSS<=MIN_LOSS:        

                    model.save_weights('../hub_model/{}-v3.h5'.format(model_name))
                    print ('SAVE WEIGHT')
                else:
                    print ('NOT SAVE')
                epoch +=1        
