import numpy as np 
import pandas as pd 
import tensorflow as tf
import data_model as model_hub

# Load Data 
df = pd.read_csv('train_1.csv')
df = df[ df.columns[ df.columns!='Page']].fillna(0)
tpc_name = np.load('tpc_name.npy')
tpc_feature = np.load('tpc_feature.npy', mmap_mode='r')
raw_leng = np.load('tpc_raw_leng.npy', mmap_mode='r')

# Cofig
hpara = {'his_window':10,
         'gap':10,
         'predict_window': 1,
         'df': df,  
         'tpc_name':tpc_name, 
         'tpc_feature':tpc_feature,
         'raw_leng': raw_leng, 
         'batch_size':2, 
         'verbose':1}

# Place Hoder 
PLD = tf.placeholder
p_X          = PLD(tf.float32, [None, hpara['his_window']])
p_dateTime   = PLD(tf.float32, [None, 5 * hpara['his_window']])
p_tpcName    = PLD(tf.int16  , [None, 35])
p_tpcFeature = PLD(tf.float32, [None, 4])
p_rawLeng    = PLD(tf.float32, [None, 1])
p_truY       = PLD(tf.float32, [None, hpara['his_window']])

# build graph
out = model_hub.model_002(p_X, p_dateTime, p_tpcName, p_tpcFeature, hpara)

loss_op = log_seq_rmse(p_truY,
	                   out, 
                       sequence_lengths = [hpara['predict_window']],
                       max_sequence_length = hpara['predict_window'])

train_op = tf.train.AdamOptimizer().minimize(loss_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #c = sess.run(lstm_layer(pld_b, [2], 3),  feed_dict={ pld_a:a, pld_b:b })
    #c = sess.run(out, feed_dict={ pld_a:a, pld_b:b, pld_c:c })


    a,b,c,d,e, f =  gen_data(**hpara)    

    feed_dict = {
        p_X          : a,
        p_dateTime   : b,
        p_tpcName    : c,
        p_tpcFeature : d,
        p_rawLeng    : e,
        p_truY       : f,
    }

    lo, _ = sess.run([loss_op, train_op], feed_dict=feed_dict)
    print (lo)