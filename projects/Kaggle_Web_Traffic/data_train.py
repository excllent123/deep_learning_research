import numpy as np 
import pandas as pd 
import tensorflow as tf

# Load Data 

df = pd.read_csv('train_1.csv')
df = df[ df.columns[ df.columns!='Page']].fillna(0)
tpc_name = np.load('tpc_name.npy')
tpc_feature = np.load('tpc_feature.npy', mmap_mode='r')
raw_leng = np.load('tpc_raw_leng.npy', mmap_mode='r')

# Cofig
hpara = {'his_window':120,
         'gap':10,
         'predict_window': 10,
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


def lstm_dense_on_datetime(p_dateTime, hidden_dim, hpara):
	p_dateTime = tf.reshape(p_dateTime, shape=[None, hpara['his_window'], 5])
	p_dateTime = tf.contrib.keras.layers.LSTM(hidden_dim, input_shape=(hpara['his_window'] ,5))(p_dateTime)
	res = tf.contrib.keras.Dense(int(hidden_dim/2))(p_dateTime)
	return res

def lstm_dense_on_X(p_X, hidden_dim, hpara):
	p_X = tf.expand_dims(p_X, axis=2)
	p_X = tf.contrib.keras.layers.LSTM(hidden_dim, input_shape=(hpara['his_window'] ,1))(p_X)
	res = tf.contrib.keras.Dense(int(hidden_dim/2))(p_X)
	return res

def build_model(p_X, p_dateTime, p_tpcName, p_tpcFeature, hpara):

	hidden_dim = 100

	a = lstm_dense_on_X(p_X, hidden_dim, hpara)

	b = lstm_dense_on_datetime(p_dateTime, hidden_dim, hpara)

	a_b = tf.add(a, b)
	

	emb_weight = tf.get_variable('tcp_name_weight', shape=(35 ,hidden_dim))
	c = tf.nn.embedding_lookup(emb_weight, p_tpcName)
	c = tf.contrib.keras.Dense(50)(c)

	d = tf.contrib.keras.Dense(3)(p_tpcFeature)
	d = tf.contrib.keras.Dense(1)(d)

	out = tf.add(tf.add(a_b, c),d)
	out = tf.contrib.keras.Dropout(0.4)
	out = tf.contrib.keras.Dense(hpara['predict_window'])
	return out



out = build_model(p_X, p_dateTime, p_tpcName, p_tpcFeature)

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