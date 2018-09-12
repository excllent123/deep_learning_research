import pandas as pd 
import numpy as np 


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
        all_length = float(len(set(df[col])))
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


def interactive_encoder(df, agg_cols, end_cols):
    '''
    A data preprocess method that may be applied before train-valide split 
    if tar_cols did not contain tar_col, but should always take care of information-leakage
    '''
    for col in end_cols:
        df[col] = df[col].apply(float)
    for col in tqdm(agg_cols):
        for end_col in end_cols:
            gp = df.groupby(col)[end_col]
            mean = gp.mean()
            std  = gp.std()
            df[col + '_{}_avg'.format(end_col)] = df[col].map(mean)
            df[col + '_{}_std'.format(end_col)] = df[col].map(std)
    return df 
