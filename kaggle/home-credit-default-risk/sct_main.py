

import pandas as pd 
import numpy as np 
import os 
import sys
import binary_model

import hub_model

import logging

from keras import backend as K
import tensorflow as tf 
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
import matplotlib.pyplot as plt
from pandas_preprocess import PandasMinMaxScaler, PandasStandardScaler


import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import sys
sys.path.append('../../')
from utility_preprocess import util as UT
import keras as ks


from random import sample

warnings.simplefilter(action='ignore', category=FutureWarning)

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val,self.y_val = validation_data
    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print('\n ROC_AUC - epoch:%d - score:%.6f \n' % (epoch+1, score))


def balanced_subsampling(df, target_col):
    '''
    Giving a df , with target_col 
    return an index with all_equal_target subsampling
    '''
    valid_size = df[target_col].value_counts().min()
    res = []
    for i in set(df[target_col]):
        temp_df = df[df[target_col]==i]
        res+=sample(list(temp_df.index), valid_size) 
    return res


def fit_predict(df,  feats, cat_feats, num_folds=2, 
    submit_pth = 'temp.csv', 
    stratified=True, model_para={}):

    for i in cat_feats:
        if i not in feats:
            raise Exception('{} must be in feats'.format(i))

    for col in  cat_feats:
        try:
            df[col] = df[col].apply(lambda x: x/(1.+abs(x)) if x < 150000 else 1)
        except:
            pass

    agent = PandasStandardScaler()
    agent.fit(df[feats])
    
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    
    del df 
    gc.collect()
    
    train_df = agent.transform(train_df)
    test_df = agent.transform(test_df)
    
    sub_preds = np.zeros(test_df.shape[0])
    oof_preds = np.zeros(train_df.shape[0])

    train_times = 0
    train_df = train_df.append(train_df, ignore_index=True)
    
    if stratified:
        folds = RepeatedStratifiedKFold(n_splits= num_folds,  n_repeats=2, random_state=32)
    else:
        folds = RepeatedKFold(n_splits= num_folds, n_repeats=1, random_state=205)

    res_vd_roc = []
    res_tr_roc = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]


        # =========================================
        # build model and training 
        with tf.Session(graph=tf.Graph()) as sess:
            ks.backend.set_session(sess)

            model = hub_model.Factoriza_model(train_x, 
                                              model_para['l1_neural'], 
                                              model_para['l2_neural'],
                                              model_para['l3_neural'])
            
            RocAuc = RocAucEvaluation(validation_data=(valid_x, valid_y), interval=1)
            
            for i in range(2):
                model.fit(x=train_x, y=train_y, batch_size=150+100*i, 
                              epochs=2, verbose=1, 
                              validation_data=(valid_x, valid_y), callbacks=[RocAuc], 
                             )
            pre_test = model.predict(test_df[feats])[:, 0]


            valid_roc = round(roc_auc_score(valid_y.values.flatten(), 
                            model.predict(valid_x[feats])[:, 0].flatten()), 4)

            train_roc = round( roc_auc_score(train_y.values.flatten(), 
                            model.predict(train_x[feats])[:, 0].flatten()), 4)

            res_vd_roc.append(train_roc)
            res_tr_roc.append(valid_roc)

        train_times +=1
        sub_preds += pre_test 
        # 
        # ======================================

    res_tr_roc = np.array(res_tr_roc).mean()
    res_vd_roc = np.array(res_vd_roc).mean()


    test_df['TARGET'] = sub_preds / train_times
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(submit_pth, index= False)

    print('saved ', submit_pth)
    return submit_pth, res_tr_roc, res_vd_roc

def get_bureau_count(tar_dir):
    bureau_balance = UT.import_data(tar_dir+'bureau_balance.csv')
    temp = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts()
    temp = temp.unstack().fillna(0)
    temp = temp.reset_index()
    bureau = UT.import_data(tar_dir+'bureau.csv')
    bureau = pd.merge(bureau, temp, on='SK_ID_BUREAU', how='left')
    bureau = bureau[['SK_ID_CURR', '0', '1', '2', '3', '4', '5', 'C', 'X']]
    bureau = bureau.fillna(0)
    del temp
    del bureau_balance        

    bureau = bureau.groupby('SK_ID_CURR').sum().reset_index()
    return bureau

def set_logger(file_pth):
    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_pth)
    fh.setLevel(logging.DEBUG)    

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s,  %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    return logger


if __name__ == '__main__':

    tar_dir = '../../.kaggle/competitions/home-credit-default-risk/'    
    logger = set_logger('hub_submit/result.log')

    preprocess_mode = 5
    if preprocess_mode ==2 :
        bureau = get_bureau_count(tar_dir)

        df = pd.read_csv('preprocessed_data_01.csv')

        print(len(df), 'BEFORE')
        df = pd.merge(df, bureau, on='SK_ID_CURR', how='left')
        print(len(df), 'AFTER')
        df_fill = df.fillna(0)    

        df_fill['TARGET'] = df['TARGET']
        #df_fill.to_csv('preprocessed_data_02.csv', index=False)        

        del df        

        df_fill.to_csv('preprocessed_data_02.csv')
        print('saved preprocessed_data_02.csv')    

    elif preprocess_mode ==4  :
        bureau = get_bureau_count(tar_dir)
        
        df = pd.read_csv('data_preprocess_3.csv')

        print(len(df), 'BEFORE')
        df = pd.merge(df, bureau, on='SK_ID_CURR', how='left')
        
        df.to_feather('data_preprocess_4.feather')
        print('saved data_preprocess_4')    
 
    else:
        print('Read data_preprocess_4 ')
        df = pd.read_feather('data_preprocess_4.feather')

    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR',
    'SK_ID_BUREAU','SK_ID_PREV', 'index', 'index_x', 'index_y', 'Unnamed: 0']]

    df_fill = df[feats].fillna(0.1)
    df_fill['SK_ID_CURR'] = df['SK_ID_CURR']
    df_fill['TARGET'] = df['TARGET']
    del df 
    gc.collect()
    

    # === Start Training 
    target_balance = 1

    if target_balance : 
        print('target balance')
        for i in range(2):
            df_fill = df_fill.append(df_fill[df_fill['TARGET']==1])
        
        a = len(df_fill[df_fill['TARGET']==1])
        b = len(df_fill[df_fill['TARGET']==0])
        print('1 [{}] samples, 0 [{}] samples, ratio : [{}]'.format(a, b, a/b))     

    cat_feats = ['PREV_APP_CREDIT_PERC_MAX','PREV_APP_CREDIT_PERC_MEAN',
                 'REFUSED_APP_CREDIT_PERC_MAX','REFUSED_APP_CREDIT_PERC_MEAN',
                 'INSTAL_PAYMENT_PERC_MAX','INSTAL_PAYMENT_PERC_MEAN','INSTAL_PAYMENT_PERC_SUM']

    model_para = {'l1_neural': 166,
                  'l2_neural': 66,'l3_neural': 60,}

    (submit_pth, 
     res_tr_roc, 
     res_vd_roc) = fit_predict(df_fill, 
                               feats=feats,
                               cat_feats = cat_feats, 
                               submit_pth = 'hub_submit/simple_dl_11.csv',
                               stratified=True, 
                               model_para=model_para)

    logger.debug('File : {} , Train_loss : {}, Valid_loss : {}, Metrics : {}, Model_para : {}'.format(
             submit_pth, 
             round(res_tr_roc,5), 
             round(res_vd_roc,5), 
             'roc_auc_score', str(model_para)))