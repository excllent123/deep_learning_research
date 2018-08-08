import tensorflow as tf
import numpy as np
import pandas as pd
import random
import datetime

def lstm_layer(inputs, lengths, state_size, keep_prob=1.0, scope='lstm-layer', reuse=False, return_final_state=False):
    """
    LSTM layer.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, max sequence length, state_size] containing the lstm
        outputs at each timestep.

    """
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(
                state_size,
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, output_state = tf.nn.dynamic_rnn(
            inputs=inputs,
            cell=cell_fw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        if return_final_state:
            return outputs, output_state
        else:
            return outputs


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
        temp_ = np.array([ int(x) for x in [year, month, date, weekday, isweekend]])
        batch = np.append(batch, temp_ )
    return batch.reshape(len(row), 5).astype(int)
    
def gen_data(his_window, gap, predict_window, df, name_df=None, batch_size=50, verbose=0):
    '''
     Args:
     - his_window : max_length of sequence data 
     - gap : the gap between 
     - df : time-series-cols df 
     - name_df : the categorical feature, indepent from time-series, 
                 this df should aligned-row with df as the same sample.
    '''
    if name_df is not None :
        assert len(df)==len(name_df)
        
    start_id = np.random.randint(low=0, high=551-his_window-gap-predict_window)
    predict_id  = start_id + his_window + gap 
    select_sample_id = np.random.randint(low=0, high=(len(df)-1), size=batch_size)
    his_cols = df.columns[start_id: start_id + his_window]
    pre_cols = df.columns[predict_id: predict_id + predict_window]
    
    batch_t2d = np.array([])
    batch_t1d = np.array([])
    batch_x  = np.array([])
    batch_y  = np.array([])
    batch_ts = np.array([])
    for selected in select_sample_id:
        if name_df is not None:
            batch_t2d = np.append(batch_t2d, name_df['topic'][selected])
            batch_t1d = np.append(batch_t1d, 
                        np.array(name_df[name_df.columns[name_df.columns!='topic']])[selected])
        batch_ts = np.append(batch_ts, get_ts(his_cols))
        batch_x  = np.append(batch_x, np.array(df[his_cols])[selected])
        batch_y  = np.append(batch_y, np.array(df[pre_cols])[selected])
        
    if name_df is not None:
        batch_t2d = batch_t2d.reshape(batch_size, 35)
        batch_t1d = batch_t1d.reshape(batch_size, len(name_df.columns[name_df.columns!='topic']))
            
    batch_x  = batch_x.reshape(batch_size, his_window)
    batch_ts = batch_ts.reshape(batch_size, his_window*5).astype(int)
    batch_y  = batch_y.reshape(batch_size, predict_window)
    
    if name_df is not None:
        res = [batch_x, batch_t2d, batch_t1d, batch_ts, batch_y]
    else:
        res = [batch_x, batch_ts, batch_y]
    if verbose:
        print([i.shape for i in res])
    return res

train = pd.read_json('test_sample_1.json')

# Topic Feature 
#train['Page']
page_details = train.Page.str.extract(r'(?P<topic>.*)\_(?P<lang>.*).org\_(?P<access>.*)\_(?P<type>.*)')
page_details['wiki']=page_details['lang'].apply(lambda x:x.split('.')[-1])
page_details['lang']=page_details['lang'].apply(lambda x:x.split('.')[0])
page_details['topic'] = page_details['topic'].apply(lambda x:replace__(x))
# page_details.head()

page_details['wiki'], wiki_memo  = build_id(page_details['wiki'], 1) 
page_details['lang'], lang_memo  = build_id(page_details['lang'], 1)
page_details['access'], acc_memo = build_id(page_details['access'], 1)
page_details['type'],   typ_memo = build_id(page_details['type'], 1)
# 2 D 
page_details['topic'], top_memo = build_id(page_details['topic'], 2)


page_details['topic'] = page_details['topic'].apply(lambda x: pad_1d(x, 35))
page_details['raw_leng'] = page_details['topic'].apply(lambda x: x[1])
page_details['topic'] = page_details['topic'].apply(lambda x:x[0])

# gen_data(4,3,3, train, name_df=page_details, batch_size=2, verbose=1)

###
# For LSTM it eat (batch, time-step, feature_dim)
### 
def build_model(a):
    #c = tf.concat([a,b], axis=1)

    a = tf.expand_dims(a,2)
    #c = tf.reshape(c, shape=(2,1,39))
    c = tf.contrib.keras.layers.LSTM(32, input_shape=(4,1))(a)
    return c

print (tf.__version__)
a,b,c,d,e =  gen_data(4,3,3, train, name_df=page_details, batch_size=2, verbose=1)
pld_a = tf.placeholder(tf.float32, [None, 4])
pld_c = tf.placeholder(tf.float32, [None,])
out_put = build_model(pld_a)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c = sess.run(out_put, feed_dict={pld_a:a})
    print (c.shape)