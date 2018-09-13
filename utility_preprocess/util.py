import pandas as pd 
import numpy as np 
from tqdm import tqdm


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


def probe_null(df)-> pd.DataFrame:
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def probe_column_content(df, col):
    try:
        display(df[col].value_counts())
    except:
        print(df[col].value_counts())

def auto_fillna(df, except_col=[]):
    for col in df.columns:
        if col in except_col:
            continue
        if df[col].dtype=='object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(0)
    return df 

def probe_cardinality(df, id_col=None):
    res = pd.DataFrame()

    if id_col:
        all_length = float(len(set(df[id_col])))
    else:
        all_length =  float(len(df))

    for col in df.columns:
        row = {
            'features': col,
            'cardinality' : len(set(df[col])),
            'null_percent' :   round(len(df[df[col].isnull()]) / all_length,4)*100,
            'cardinality_percent' : round(len(set(df[col])) / all_length, 4)*100,
        }
        res = res.append(row, ignore_index=True)
    return res
            
def show_lgbm_feature_imp(x_cols, lgbm):
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = x_cols
    fold_importance_df["importance"] = lgbm.feature_importance()
    fold_importance_df = fold_importance_df.sort_values('importance', ascending=False)
    display(fold_importance_df)
    return fold_importance_df


def ensemble_avg_pre(models, X):
    res = []
    for m in models:
        pre_y = m.predict(X)
        res.append(pre_y)
    return np.array(res).mean(0)

def ensemble_newX(models, X):
    res = []
    for m in models:
        pre_y = m.predict(X)
        res.append(pre_y)
    return np.array(res)


def interactive_encoder(train_df, valid_df, test_df,  agg_cols, end_cols):
    '''
    A data preprocess method that may be applied before train_df-valide split 
    if tar_cols did not contain tar_col, but should always take care of information-leakage
    '''
    res_encoder = {} # col
    for col in end_cols:
        train_df[col] = train_df[col].apply(float)
    for col in tqdm(agg_cols):
        for end_col in end_cols:
            gp = train_df.groupby(col)[end_col]
            mean = gp.mean()
            std  = gp.std()
            try: 
                # check valid_df could mapping first 
                valid_df[col + '_{}_avg'.format(end_col)] = valid_df[col].map(mean)
                valid_df[col + '_{}_std'.format(end_col)] = valid_df[col].map(std)

                test_df[col + '_{}_avg'.format(end_col)] = test_df[col].map(mean)
                test_df[col + '_{}_std'.format(end_col)] = test_df[col].map(std)

                train_df[col + '_{}_avg'.format(end_col)] = train_df[col].map(mean)
                train_df[col + '_{}_std'.format(end_col)] = train_df[col].map(std)
            except Exception as e :
                print('No value for valid_df/test_df map, imbalanced encoder', col)
                pass
    return train_df, valid_df, test_df

def preprocess_policy_gp(df, id_col='Policy_Number', tar_col='Next_Premium', agg_para=['mean', 'std']):

    temp = df[[id_col, tar_col]].copy()
    del df[tar_col]
    df = df.groupby(id_col).agg(agg_para)
    df.columns = [''.join(i) for i in df.columns.values ] 
    df[id_col] = df.index
    df = auto_fillna(df)
    df = pd.merge(df, temp, on=id_col, how='inner')
    df.index = range(len(df))
    return df



# temp = probe_cardinality(policy_df, 'Policy_Number')# 

# temp = temp[temp['cardinality'] > 1]
# agg_cols = list(temp[temp['cardinality'] < 2000]['features'])
# agg_cols = [i for i in agg_cols if i not in end_cols]
# temp

# end_cols=['Replacement_cost_of_insured_vehicle', 'Insured_Amount3', 'Premium', 'Next_Premium' ]# 

# train_df = pd.merge(train_df, policy_df, on='Policy_Number', how='left')# 

# test_df = pd.merge(test_df, policy_df, on='Policy_Number', how='left')# 

# tar_col='Next_Premium'# 

# id_col='Policy_Number'# 

# param_01={
#     'boosting_type': 'gbdt',
#     'class_weight': None,
#     'colsample_bytree': 0.733333,
#     'learning_rate': 0.008,
#     'max_depth': 6,    
#     'min_child_weight': 0.0005,
#     'min_split_gain': 0.0,
#     'n_estimators': 8000,
#     'n_jobs': 2,
#     'num_leaves': 72,
#     'objective': None,
#     'random_state': 42,
#     'reg_alpha': 0.877551,
#     'reg_lambda': 0.204082,
#     'silent': True,
#     'subsample': 0.989495,
#     'subsample_for_bin': 240000,
#     'subsample_freq': 1,
#     'metric': 'l1' # aliase for mae 
#     }    

def gen_kfold(train_df, test_df, agg_cols, end_cols,  n_fold=6, 
              tar_col='Next_Premium', id_col='Policy_Number'):
    '''
    test_df need have tar_col and id_col 
    '''
    fold = KFold(n_fold, shuffle=True)
    
    # kfold -- dataframe
    for epch, ind in enumerate(fold.split(train_df)):
        train_ind, valid_ind  = ind
        
        fold_train_df = train_df.iloc[train_ind]
        fold_valid_df = train_df.iloc[valid_ind]
        
        # ==========================
        fold_train_df, fold_valid_df, test_df = interactive_encoder(fold_train_df, 
                                                                    fold_valid_df, 
                                                                    test_df, 
                                                                    agg_cols, 
                                                                    end_cols)
        
        fold_train_df = preprocess_policy_gp(fold_train_df, id_col, tar_col)
        fold_valid_df = preprocess_policy_gp(fold_valid_df, id_col, tar_col)
        test_df       = preprocess_policy_gp(test_df, id_col, tar_col)
        # ==========================
        
        trainY = fold_train_df[tar_col].values.flatten()
        validY = fold_valid_df[tar_col].values.flatten()
        
        x_cols = [i for i in fold_train_df.columns if (i != tar_col and i!= id_col)]
        
        trainX = fold_train_df[x_cols].values
        validX = fold_valid_df[tar_col].values
        
        yield trainX_df, trainY, validX_df, validY, test_df, x_cols # need id -col


def gen_kfold_002(train_df, test_df, agg_cols, end_cols,  n_fold=6, 
              tar_col='Next_Premium', id_col='Policy_Number'):
    '''
    test_df need have tar_col and id_col 
    '''
    fold = KFold(n_fold, shuffle=True)
    
    
    
    # kfold -- dataframe
    for epch, ind in enumerate(fold.split(train_df)):
        train_ind, valid_ind  = ind
        
        fold_train_df = train_df.iloc[train_ind]
        fold_valid_df = train_df.iloc[valid_ind]
        
        # ==========================
        fold_train_df, fold_valid_df, test_df_atf = interactive_encoder(fold_train_df, 
                                                                    fold_valid_df, 
                                                                    test_df, 
                                                                    agg_cols, 
                                                                    end_cols)
        
        fold_train_df = preprocess_policy_gp(fold_train_df, id_col, tar_col)
        fold_valid_df = preprocess_policy_gp(fold_valid_df, id_col, tar_col)
        test_df_atf       = preprocess_policy_gp(test_df_atf, id_col, tar_col)
        # ==========================
        
        yield fold_train_df,fold_valid_df, test_df_atf 

