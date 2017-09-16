import numpy as np 
import pandas as pd 
import tensorflow as tf
import data_model as model_hub
from data_generator import gen_data
from tensorflow.contrib.keras import backend as K 
from tqdm import tqdm
import os


def train_loop(sess, 
    max_iter, 
    hpara, 
    train_op, 
    loss_op, 
    generator, 
    feed_dict, 
    restro_pth=None):
    '''
    generator return []

    '''
    saver = tf.train.Saver()
    save_loss = 999
    mod_step = 3
    if restro_pth:
        saver.restore(sess, restro_pth)

    for ep in range(hpara['epoch']):
        pbar = tqdm(range(max_iter))
        ep_loss = 0

        for i in pbar:
            res = generator(**hpara)
            _ = sess.run(train_op, feed_dict=feed_dict)
            if i % mod_step == 0:
                lo = sess.run(loss_op, feed_dict=feed_dict)
                ep_loss=+lo
                pbar.set_description('EP:{}, Loss:{}'.format( str(ep) , str(lo)))
        ep_loss /= int(max_iter/mod_step)
        pbar.set_description('EP:{}, Loss:{}'.format( str(ep) , str(ep_loss)))

        # save best 
        if save_loss > ep_loss:
            saver.save(sess, os.path.join(os.getcwd(), hpara['model_save_pth']))
            save_loss = ep_loss


if __name__ =='__main__':

    # Load Data 
    df = pd.read_csv('train_1.csv')
    df = df[ df.columns[ df.columns!='Page']].fillna(0)
    tpc_name = np.load('tpc_name.npy')
    tpc_feature = np.load('tpc_feature.npy', mmap_mode='r')
    raw_leng = np.load('tpc_raw_leng.npy', mmap_mode='r')    

    # Cofig
    hpara = {'his_window':20,
             'gap':10,
             'predict_window': 10,
             'df': df,  
             'tpc_name':tpc_name, 
             'tpc_feature':tpc_feature,
             'raw_leng': raw_leng, 
             'batch_size':10, 
             'epoch':10,
             'verbose':0,
             'model_save_pth':'model'}
    hpara = {'his_window':30,
             'gap':10,
             'predict_window': 5,
             'df': df,  
             'tpc_name':tpc_name, 
             'tpc_feature':tpc_feature,
             'raw_leng': raw_leng, 
             'batch_size':20, 
             'epoch':20,
             'verbose':0,
             'model_save_pth':'model2'}
    #test_hpara
    # Place Hoder 
    PLD = tf.placeholder
    p_X          = PLD(tf.float32, [None, hpara['his_window']])
    p_dateTime   = PLD(tf.float32, [None, 5 * hpara['his_window']])
    p_tpcName    = PLD(tf.int32  , [None, 35])
    p_tpcFeature = PLD(tf.float32, [None, 4])
    p_rawLeng    = PLD(tf.float32, [None, 1])
    p_truY       = PLD(tf.float32, [None, hpara['predict_window']])    

    # Build graph
    out = model_hub.model_001(p_X, p_dateTime, p_tpcName, p_tpcFeature, hpara)    

    loss_op = model_hub.log_seq_rmse(p_truY, out, 
                           sequence_lengths = [hpara['predict_window']],
                           max_sequence_length = hpara['predict_window'])    

    train_op = tf.train.AdamOptimizer().minimize(loss_op)    

    # Run Sess
    with tf.Session() as sess:
        # init keras sess
        K.set_session(sess)
        # init tf variable 
        sess.run(tf.global_variables_initializer())

        # init generator 
        res =  gen_data(**hpara)    
        # init feed_dict
        feed_dict = {
            p_X          : res[0],
            p_dateTime   : res[1],
            p_tpcName    : res[2],
            p_tpcFeature : res[3],
            p_rawLeng    : res[4],
            p_truY       : res[5],
            # keras 1 = train, 0 = test
            K.learning_phase(): 1, }    

        train_loop(sess, 1000, hpara, 
            train_op, loss_op, gen_data, feed_dict, 
            restro_pth=hpara['model_save_pth'])
