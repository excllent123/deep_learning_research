
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


# =========================================
W = 448
H = 448
S = 7
B = 2
C = 2
nb_classes = C
batch_size = 2
epoch_size = 2500

model_name = __file__.split('\\')[-1].split('.')[0]
file_path = '../hub_data/vatic/vatic_id2/example_off_withSave4.txt'
maplist = ['Rhand', 'ScrewDriver']
log_dir = '../hub_logger/{}-v2'.format(model_name)

# =======================================

A =      YoloDetector(C = C)
processer = VaticPreprocess(file_path, maplist=maplist, detector=A)

yolooo = YoloNetwork(C=C)
model = yolooo.yolo_tiny()

# =======================================
model_json = model.to_json()
with open("../hub_model/{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)
    print ('saving model struct as ' + "../hub_model/{}.json".format(model_name))

# =======================================

true_y = tf.placeholder(tf.float32, shape = (None , S*S*(B*5+C)))
input_tensor = Input(shape=(H, W, 3))

pred_y = model(input_tensor)



def batch_check(x, batch_size):
    while len(x) != batch_size :
        x = list(x)
        x.append(x[0])
        x = np.asarray(x)
    return x



loss = A.loss(true_y, pred_y, batch_size=batch_size) # tf-stle slice must have same rank
#loss = tf.py_func(A.loss, true_y[0,:], pred_y[0,:])

#train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)
train_step = tf.train.RMSPropOptimizer(FLAGS.learning_rate, momentum=0.9).minimize(loss)
# Initializing the variables
summary_op = get_summary_op(model, loss)

init = tf.global_variables_initializer()

MIN_LOSS = 9999
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
        MIN_LOSS = min(SUM_LOSS, MIN_LOSS)
        if SUM_LOSS<=MIN_LOSS:

            model.save_weights('../hub_model/{}-v3.h5'.format(model_name))
            print ('SAVE WEIGHT')
        else:
            print ('NOT SAVE')
        epoch +=1

# v2 ~ loss = 1.8 with 1e-6 557EP


