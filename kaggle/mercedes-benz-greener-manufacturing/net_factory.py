import tensorflow as tf
Callback           = tf.contrib.keras.callbacks.Callback
Dense              = tf.contrib.keras.layers.Dense
Input              = tf.contrib.keras.layers.Input
Dropout            = tf.contrib.keras.layers.Dropout
Activation         = tf.contrib.keras.layers.Activation
BatchNormalization = tf.contrib.keras.layers.BatchNormalization
Sequential         = tf.contrib.keras.models.Sequential
Model              = tf.contrib.keras.models.Model
maxnorm            = tf.contrib.keras.constraints.max_norm
Multiply           = tf.contrib.keras.layers.Multiply
# define custom R2 metrics for Keras backend
K                  = tf.contrib.keras.backend

#from keras.initializersitializers import TruncatedNormal
TruncatedNormal    = tf.contrib.keras.initializers.TruncatedNormal



class NetFactory(object):
    '''
    simple net for regression, the input is 1D sequence 
    '''
    def __init__(self, model_name, **kwargs):

        # set attribute by parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.registed = {'n01': self.model_1, 
                         'n02': self.model_2,
                         'n03': self.model_3, 
                         'n04': self.model_4}

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

    def model_4(self):
        
        x = Input(shape= (self.input_dim,))

        block_1 = self.residual(x, self.input_dim)
        pro_1   = Dense(int(self.n_feature*0.5), activation ='elu')(block_1)

        block_2 = self.residual(pro_1,  int(self.n_feature*0.5))

        pro_2   = Dense(int(self.n_feature*0.3), activation ='elu')(block_2)

        block_3 = self.residual(pro_2, int(self.n_feature*0.3))

        pro_3   = Dense(int(self.n_feature*0.3), activation ='elu')(block_3)

        y      = Dense(1)(pro_3)

        model = Model(x, y)
        model.summary()
        return model

    def residual(self, x, n_feature):
        x1 = Dense( int(n_feature), 
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01 ), 
                   activation='elu')(x)

        x1 = Dropout(0.5)(x1)

        x1 = Dense( int(n_feature), 
                   kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01 ), 
                   activation='elu')(x1)

        x2 = Multiply()([x1, x])

        return x2

    def spare_gate_net(self, input_vector, experts):
        from keras.layers import Reshape, Lambada, merge
        from keras import backend as K

        def slice(x, expert_num):
            return x[:,:, :expert_num]

        def reduce(x, axis=2):
            return K.sum(x, axis=2)

        expert_num = len(experts)

        gate = Dense (nb_class*(expert_num+1), activation='sigmoid')(input_vector)

        gate = Reshape((nb_class, expert_num+1))(gate)

        gate = Lambada(slice, output_shape = (nb_class, expert_num), 
                       arguments = {'expert_num':expert_num})

        expert = Dense(nb_class*expert_num, activation='softmax')(input_vector)
        expert = Reshape((nb_class, expert_num))(expert)

        output = merge([gate, expert], mode = 'mul')

        output = Lambada(reduce, output_shape=(nb_class,), 
                         arguments = {'axis':2 })(output)
        model = Model(input_vector, output)

        return model






