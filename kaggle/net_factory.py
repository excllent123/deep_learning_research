import tensorflow as tf
Callback           = tf.contrib.keras.callbacks.Callback
Dense              = tf.contrib.keras.layers.Dense
Input              = tf.contrib.keras.layers.Input
Dropout            = tf.contrib.keras.layers.Dropout
Activation         = tf.contrib.keras.layers.Activation
BatchNormalization = tf.contrib.keras.layers.BatchNormalization
Sequential         = tf.contrib.keras.models.Sequential
maxnorm            = tf.contrib.keras.constraints.max_norm
# define custom R2 metrics for Keras backend
K                  = tf.contrib.keras.backend

#from keras.initializersitializers import TruncatedNormal
TruncatedNormal    = tf.contrib.keras.initializers.TruncatedNormal



class NetFactory(object):
    '''
    simple net for regression, the input is 1D sequence 
    '''
    def __init__(self, model_name, **kwargs):
        '''
        check by self.registed
        '''

        # set attribute by parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.registed = {'m01': self.model_1, 
                         'm02': self.model_2,
                         'm03': self.model_3}

        self.model_name = model_name


        if model_name not in self.registed.keys():
            raise ValueError('the model name [{}] is not in the registed model \n'
                             'go to model NetFactory to regiested it')

    def get_model(self):
        '''
        return keras-model
        '''
        return self.registed[self.model_name]()

    def model_1(self):
        # equial to build-model
        model = Sequential()
        model.add(Dense(int(self.n_feature*0.8), kernel_constraint=maxnorm(3),
                        kernel_initializer= TruncatedNormal(mean=0.1, stddev=0.05, seed=None),
                        bias_initializer='zeros', activation='elu',
                        input_dim=self.input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(int(self.n_feature*0.5),kernel_constraint=maxnorm(3),
                        kernel_initializer=TruncatedNormal(mean=0.1, stddev=0.05),
                        bias_initializer='zeros',
                        activation='elu' ))
        
        model.add(Dropout(0.2))
        model.add(Dense(int(self.n_feature*0.2), kernel_constraint=maxnorm(3),
                        bias_initializer='zeros',
                        kernel_initializer=TruncatedNormal(mean=0.1, stddev=0.05)))
        model.add(Activation("linear"))
        model.add(Dense(1))
        model.summary()
        return model

    def model_2(self):
        # equial to build-model
        model = Sequential()
        model.add(Dense(int(self.n_feature*0.8), kernel_constraint=maxnorm(3),
                        kernel_initializer= TruncatedNormal(mean=0.1, stddev=0.05, seed=None),
                        bias_initializer='zeros',
                        input_dim=self.input_dim))
        model.add(BatchNormalization())
        model.add(Dense(int(self.n_feature*0.8),kernel_constraint=maxnorm(3),
                        kernel_initializer=TruncatedNormal(mean=0.1, stddev=0.05),
                        bias_initializer='zeros',
                        activation='elu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(int(self.n_feature*0.5),kernel_constraint=maxnorm(3),
                        kernel_initializer=TruncatedNormal(mean=0.1, stddev=0.05),
                        bias_initializer='zeros',
                        activation='elu'))
        
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(int(self.n_feature*0.5),kernel_constraint=maxnorm(3),
                        kernel_initializer=TruncatedNormal(mean=0.1, stddev=0.05),
                        bias_initializer='zeros',
                        activation='elu' ))
        
        model.add(Dropout(0.1))
        model.add(Dense(int(self.n_feature*0.3), kernel_constraint=maxnorm(3),
                        bias_initializer='zeros',
                        kernel_initializer=TruncatedNormal(mean=0.1, stddev=0.05)))
        model.add(Activation("linear"))
        model.add(Dense(self.n_target))
        model.summary()
        return model

    def model_3(self):
        # equial to build-model
        model = Sequential()
        model.add(Dense(int(self.n_feature*0.7),
                        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01 ),
                        input_dim  = self.input_dim))

        model.add(Dropout(0.5))
        model.add(Dense(int(self.n_feature*0.5), 
                        activation ='elu'))

        model.add(Dropout(0.3))
        model.add(Dense(int(self.n_feature*0.3), 
                        activation ='elu'))

        model.add(Dropout(0.1))
        model.add(Dense(int(self.n_feature*0.3), 
                        activation ='elu'))

        model.add(Dense(1))
        model.summary()
        return model



