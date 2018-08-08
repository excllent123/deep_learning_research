

from pandas_preprocess import PandasMinMaxScaler
import numpy as np 
import pandas as pd 
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import RFE

from catboost import CatBoostClassifier

def boostraping(df, models, y_col, X_col, iter_=200):
    '''
      Description : 
        - A bag of model for boostraping samples, voting 
      Args
        - df : main dataset 
        - models : a list of models scikit style
        - y_col : a str/int/float of a col-name
        - X_col : a list of col-name 
    '''
    df = df[df[y_col].notnull()]
    record = {}
    for modo_ in models:
        for iii in range(iter_):
            # bootstraping 
            train_all = df.sample(frac=0.7, replace=True)
            feature_n = len(X_col)
            X = train_all[X_col]
            y = train_all[y_col]

            
            rfe = RFE(estimator=modo_, n_features_to_select=5, step=100)
            rfe.fit(X, y)
            
            new_feature = np.array(X_col)[rfe.get_support()]
            for i in new_feature:
                if i not in record.keys():
                    record[i] = 1
                else:
                    record[i] +=1
        print(len(train_all))
    return record

def kfold_lightgbm(df, feats, num_folds=5, stratified = False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 100)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('================================') 
    valid_score = round(roc_auc_score(train_df['TARGET'], oof_preds), 3)

    print('Full AUC score %.6f' % valid_score)
    # Write submission file and plot feature importance
    test_df['TARGET'] = sub_preds

    submission_file_name = '{}_KF-{}_F-{}.csv'.format(
    	valid_score, num_folds, len(feats))

    test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

    print('saved ', submission_file_name)

def test(df, feats, num_folds=5, stratified=False, ):    
    for col in  ['INS_PAYMENT_PERC_MAX', 'INS_PAYMENT_PERC_MEAN', 'INS_PAYMENT_PERC_SUM', 
                 'PREV_APP_CREDIT_PERC_MAX', 'PREV_APP_CREDIT_PERC_MEAN', 
                 'REF_APP_CREDIT_PERC_MAX', 'REF_APP_CREDIT_PERC_MEAN']:
        #df[col] = df[col].apply(lambda x: np.log(x+1e-8))
        df[col] = df[col].apply(lambda x: x/(1.+abs(x)) if x < 100000 else 1.1)

    agent = PandasMinMaxScaler()
    agent.fit(df[feats])
    
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    
    train_df = agent.transform(train_df)
    test_df = agent.transform(test_df)
    
    sub_preds = np.zeros(test_df.shape[0])
    oof_preds = np.zeros(train_df.shape[0])

    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        model = GradientBoostingClassifier()
        model.fit(train_x, train_y)
        pre_valid = model.predict(valid_x).flatten()
        print('Fold {} : with {}'.format(n_fold, roc_auc_score(valid_y,pre_valid )))
        
        pre_test = model.predict_proba(test_df[feats])[:, 1]
 
        sub_preds += pre_test / folds.n_splits

    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

def kfold_trainer(df, feats, model, 
	model_fit_para={},
	model_predict_para={},
	num_folds=5, 
	stratified = 1, 
	flag_submit=1):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        model.fit(train_x, train_y, **model_fit_para)

        oof_preds[valid_idx] = model.predict_proba(valid_x, **model_predict_para)[:, 1]

        sub_preds += model.predict_proba(test_df[feats], **model_predict_para)[:, 1] / folds.n_splits

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    print('================================') 
    valid_score = round(roc_auc_score(train_df['TARGET'], oof_preds), 3)

    print('Full AUC score %.6f' % valid_score)

    if flag_submit:

        submission_file_name = '{}_KF-{}_F-{}.csv'.format(
    	    valid_score, num_folds, len(feats))
        
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

        print('saved ', submission_file_name)
