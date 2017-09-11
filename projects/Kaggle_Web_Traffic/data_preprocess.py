
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import datetime


def build_id(col, mode):
    '''
     Args 
       - mode :
         1 : for general feature 
         2 : for sequential feature like language in each cell 
       - col : pandas series
    '''
    if mode == 1:
        key = list(set(col.drop_duplicates()))
    elif mode == 2:
        key = list(set([val for sublist in list(col) for val in sublist]))
    value = [i for i in range(1,len(key)+1)]
    memo =  { k:v for k, v in list(zip(key, value))}
    if mode ==1:
        col = col.apply(lambda x: memo[x])
    elif mode ==2:
        col = col.apply(lambda x: [memo[i] for i in x] )
    return np.array(col), memo

def replace__(x):
    for i in ".'_:;()[!@#$%^&*-]":
        x=x.replace(i,' ')
    return x.split()

def pad_1d(array, max_len):
    array = array[:max_len]
    raw_length = len(array)
    padded = array + [0]*(max_len - len(array))
    return padded, raw_length


def get_ts(row):
    batch = np.array([])
    for i in range(len(row)):
        year, month, date = row[i].split('-')
        weekday = datetime.datetime.strptime( row[i], '%Y-%m-%d').weekday()
        isweekend = 1 if weekday > 5 else 0
        year_memo = {2015:0, 2016:1, 2017:2}
        year      = year_memo[int(year)]
        month     = int(month)-1
        date      = int(date)-1
        weekday   = int(weekday)
        temp_ = np.array([ int(x) for x in [year, month, date, weekday, isweekend]])
        batch = np.append(batch, temp_ )
    return batch.reshape(len(row), 5).astype(int)

time_series = None
name_feature = None

if time_series:
    df = pd.read_csv('train_1.csv')    

    df = df.columns[df.columns!='Page']    

    df = get_ts(df)    

    np.save('time-series.npy', df)


if name_feature:
    df = pd.read_csv('train_1.csv')     

    df = df.Page.str.extract(r'(?P<topic>.*)\_(?P<lang>.*).org\_(?P<access>.*)\_(?P<type>.*)')
    df['wiki']=df['lang'].apply(lambda x:x.split('.')[-1])
    df['lang']=df['lang'].apply(lambda x:x.split('.')[0])
    df['topic'] = df['topic'].apply(lambda x:replace__(x))
    # df.head()    

    df['wiki'], wiki_memo  = build_id(df['wiki'], 1) 
    df['lang'], lang_memo  = build_id(df['lang'], 1)
    df['access'], acc_memo = build_id(df['access'], 1)
    df['type'],   typ_memo = build_id(df['type'], 1)
    # 2 D 
    df['topic'], top_memo = build_id(df['topic'], 2)    
    

    df['topic'] = df['topic'].apply(lambda x: pad_1d(x, 35))
    df['raw_leng'] = df['topic'].apply(lambda x: x[1])
    df['topic'] = df['topic'].apply(lambda x:x[0])    
    

    np.save('tpc_name.npy', np.array(df['topic']))
    np.save('tpc_raw_leng.npy', np.array(df['raw_leng']))
    np.save('tpc_feature.npy', np.array(df[ ['wiki', 'lang', 'access', 'type']]))
