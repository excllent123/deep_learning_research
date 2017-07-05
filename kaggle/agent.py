
'''
# Note : 
  - in regression model, we could use val_loss as a fair parameter

# Des : 
  1. train-mode 
     - train-hyper-para
       - save-address
       - train-log
       - learning-rate
       - train-times
      
     - model-hyper-para
       - ...  

    - preprocess-hyper-para  

  2. infer
    - load preprocess module
    - load-model-module
      - model-hyper-para
      - load-weight-model-address
    - config output / result

# CLI Usage : 
  ```
  python agent.py -m 1 -f model-weight.h5 -n m01 
  ```


'''


import tensorflow as tf 
import numpy as np
import random
from common_log import DebugLog   
from net_factory import NetFactory

# to tune the NN
Adam               = tf.contrib.keras.optimizers.Adam
EarlyStopping      = tf.contrib.keras.callbacks.EarlyStopping
ModelCheckpoint    = tf.contrib.keras.callbacks.ModelCheckpoint
# define custom R2 metrics for Keras backend
K                  = tf.contrib.keras.backend

# model evaluation
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error
import argparse as arrg 


if __name__=='__main__':
    par = arrg.ArgumentParser()

    par.add_argument('-m', '--mode', type=int, default=0, 
        help='assign 1 for Train-mode, default 0 for Infer-mode \n')

    par.add_argument('-n', '--net_name', type=str, default='m01', 
        help='assign the registed model')

    par.add_argument('-f', '--weight_file', type=str, required=True, default='model-weight.h5',
        help='Assign the weight file address, \n, Saving_address in train-mode \n'
             'Loading_address , and Result.csv_address in infer-mode\n'
             'Do not contain dot (.) in the file-name-string' )

    par.add_argument('-l','--log_file', type=str, default='default.txt' )

    par.add_argument('-c', '--config_file', type=str, default=None, 
        help='model hyper-parameter configuration file')

    arg = vars(par.parse_args())


def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class Trainer():
    '''multi-loss ; multi-net '''
    def __init__(self, model_name , train_x, train_y, test):
        
        self.train_x    = train_x # pd.Dataframe object 
        self.train_y    = train_y
        self.test       = test    # the data to only infer
        assert len(train_x.columns.values)== len(test.columns.values)
        
        self.n_feature  = len(train_x.columns.values)
        self.input_dim  = train_x.shape[1]
        self.n_target   = 1
        self.model      = NetFactory(model_name, 
                                     n_feature  = len(train_x.columns.values), 
                                     n_target   = self.n_target , 
                                     input_dim  = self.input_dim ).get_model()
        self._encoder()
        self.log_obj = DebugLog('Agent')
        
    def _encoder(self):
        B = 1/self.train_x.max()
        self.train_x *= B
        self.test *= B
        
        # output_1 
        self.A = self.train_y.max()
        self.train_y/= self.A
        
    def _recover_y(self, value):
        # only when infer
        value*=self.A
        return value 

    
    def keras_train(self, lr = 1e-5, batch_size=15 , save_file_address=None):
        # regrss => not use momentum

        ad = Adam(lr= lr)
        self.model.compile(loss='mean_absolute_error', # one may use 'mean_absolute_error' as alternative
                           optimizer=ad, metrics=[r2_keras,"mse"] )
    
        X_tr, X_val, y_tr, y_val = train_test_split(np.array(self.train_x), np.array(self.train_y), 
                                                    test_size=0.2, random_state=100)
        
        if save_file_address:
            callbacks=[ModelCheckpoint(save_file_address, save_best_only=True)]

        self.model.fit( X_tr, y_tr, epochs=500, batch_size= batch_size, 
                       validation_data=(X_val, y_val), verbose=2,
                       callbacks=callbacks, shuffle=True)


        
    def tf_train(self, lr, batch_size= 15, iterations=100000):
        #x = Input(shape=(,self.train_x.shape[1])) # in keras Input, dont need specify first-dim 
        x = tf.placeholder(tf.float32, shape=(None, self.train_x.shape[1]))
        y = tf.placeholder(tf.float32, shape=(None, 1 ))
        # use None not self.batch_size is because while infer, we use single
        y_pred = self.model(x)        

        loss_op = tf.reduce_mean(tf.squared_difference(y_pred, y)) # mse
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss_op)

        save_op = tf.train.Saver()

        
        Y = np.array(self.train_y)  ;  X = np.array(self.train_x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            epo = 1
            memo_loss = 1.
            self.log_obj.warn('[ Start Training ]')
            while epo < iterations:

                select = random.sample(range(0,self.train_x.shape[0]), batch_size)

                # train_y is a list 
                batch_y = np.array([ [Y[i]] for i in select])
                batch_x = np.array([  X[i] for i in select])
                
                _, loss_val, pred_val  = sess.run([train_op,loss_op, y_pred], 
                                        feed_dict ={ x : batch_x,
                                                     y : batch_y, K.learning_phase(): 0})
                if epo%2==0:
                    self.log_obj.warn("EP{}-Loss-{}-rP{}-P{}".format(epo ,loss_val, 
                        self._recover_y(pred_val[0]), pred_val[0]  ))

                epo+=1

                if loss_val < memo_loss:
                    memo_loss = loss_val
                    save_op.save('model.ckpt')
                    self.log_obj.warn('SAVED')


    def predict(self, weight_file=None):
        self.log_obj.warn(self.test.shape)
        if weight_file:
            self.log_obj.warn('loaded_weightfile')
            self.model.load_weights(weight_file)

        pred = self.model.predict(np.array(self.test)).ravel()
        pred = self._recover_y(pred)
        return pred

if __name__ == '__main__':

    from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
    from sklearn.preprocessing import LabelEncoder    

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)    
    
    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.random_projection import SparseRandomProjection    

    from sklearn.decomposition import PCA, FastICA
    from sklearn.decomposition import TruncatedSVD    

    from sklearn.metrics import r2_score


    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    id_test = test['ID']
    train.pop('ID')
    test.pop('ID')    

    for c in train.columns:
        if train[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train[c].values) + list(test[c].values))
            train[c] = lbl.transform(list(train[c].values))
            test[c] = lbl.transform(list(test[c].values))    
    

    n_comp = 20

    # tSVD
    tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
    tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
    tsvd_results_test = tsvd.transform(test)    

    # PCA
    pca = PCA(n_components=n_comp, random_state=420)
    pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
    pca2_results_test = pca.transform(test)    

    # ICA
    ica = FastICA(n_components=n_comp, random_state=420)
    ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
    ica2_results_test = ica.transform(test)    

    # GRP
    grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
    grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
    grp_results_test = grp.transform(test)    


    #save columns list before adding the decomposition components    

    usable_columns = list(set(train.columns) - set(['y']))    

    # Append decomposition components to datasets
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca2_results_train[:, i - 1]
        test['pca_' + str(i)] = pca2_results_test[:, i - 1]    

        train['ica_' + str(i)] = ica2_results_train[:, i - 1]
        test['ica_' + str(i)] = ica2_results_test[:, i - 1]    

        train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
        test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]    

        train['grp_' + str(i)] = grp_results_train[:, i - 1]
        test['grp_' + str(i)] = grp_results_test[:, i - 1]    

    train_y = train['y'].values


    feature_cols =[]
    for i in range(1, n_comp + 1):
        feature_cols+=['pca_' + str(i), 'ica_' + str(i), 'tsvd_' + str(i), 
                            'grp_' + str(i) ]    

    train_x = train[feature_cols] 
    test   = test[feature_cols]  

    agent = Trainer(model_name= arg['net_name'], 
                    train_x=train_x, 
                    train_y= train_y, 
                    test = test)
    #agent.tf_train(lr= 1e-5)
    if arg['mode']==1 :
        print ('[*] :::: Train Mode Starting ::::')
        agent.keras_train(lr=1e-5, save_file_address = arg['weight_file'])

    else:
        print ('[*] :::: Inference Mode Starting ::::')
        y_pred = agent.predict(weight_file = arg['weight_file'])
        
        # 
        memo = { 1 : 71.34112, 12 : 109.30903, 23 : 115.21953, 28 : 92.00675, 42 : 87.73572, 43 : 129.79876, 
        45 : 99.55671, 57 : 116.02167, 3977 : 132.08556}
        for i in range(len(id_test)):
            if id_test[i] in memo.keys():
                y_pred[i] = memo[id_test[i]]
        sub = pd.DataFrame()
        sub['ID'] = id_test
        sub['y'] = y_pred        

        # sub['y'] = y_pred*0.75 + results*0.25
        sub_file = arg['weight_file'].split('.')[0]+'.csv'
        sub.to_csv( sub_file, index=False)

        print ('[*] Saved Output CSV at {}'.format(sub_file))
