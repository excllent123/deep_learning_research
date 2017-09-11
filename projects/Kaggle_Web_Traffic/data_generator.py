import datetime
import numpy as np

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

def help_reshape(v, batch_size):
    # v = 1D
    return v.reshape(batch_size, int(len(v)/batch_size))

def gen_data(his_window, 
             gap, 
             predict_window, 
             df, 
             tpc_name,
             tpc_feature,
             raw_leng, 
             batch_size=50, 
             verbose=1):
    '''
     Args:
     - his_window : max_length of sequence data 
     - gap : the gap between 
     - df : pd.read train.csv
     Reture:
     - [bth_x, bth_datetime, bth_tpc_name, bth_tpc_feature, bth_raw_leng, bth_tru_y]        
    '''
    # select sample
    row_ids = np.random.randint(low=0, high=(len(df)-1), size=batch_size)
    
    # window
    start_id = np.random.randint(low=0, high=551-his_window-gap-predict_window)
    predict_id  = start_id + his_window + gap 
    
    # df columns must with only yyyy-mm-dd
    his_cols = df.columns[start_id: start_id + his_window]
    pre_cols = df.columns[predict_id: predict_id + predict_window]
    
    # init_variable
    bth_x           = np.array([])
    bth_datetime    = np.array([])
    bth_tpc_name    = np.array([])
    bth_tpc_feature = np.array([])
    bth_raw_leng    = np.array([])
    bth_tru_y       = np.array([])
    
    NAP = np.append
    for selected in row_ids:
        bth_x           = NAP(bth_x          , np.array(df[his_cols])[selected])
        bth_tru_y       = NAP(bth_tru_y      , np.array(df[pre_cols])[selected])
        bth_datetime    = NAP(bth_datetime   , get_ts(his_cols))
        bth_tpc_name    = NAP(bth_tpc_name   , tpc_name[selected] )
        bth_tpc_feature = NAP(bth_tpc_feature, tpc_feature[selected])
        bth_raw_leng    = NAP(bth_raw_leng   , raw_leng[selected] )
        
    res = [bth_x, bth_datetime, bth_tpc_name, bth_tpc_feature, bth_raw_leng, bth_tru_y] 
    res = [help_reshape(i, batch_size) for i in res ]
    if verbose:
        print([i.shape for i in res])

    return res

