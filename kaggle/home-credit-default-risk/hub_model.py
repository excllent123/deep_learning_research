




from keras import backend as K
import tensorflow as tf 

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from deep_learning_research.preprocess_toolkit import util as UT
import keras as ks
from keras.backend.tensorflow_backend import clip
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from random import sample


class FactorizationMachinesLayer(Layer):
    '''Factorization Machines layer.

    # Arguments
        output_dim: int > 0.
        k: k of Factorization Machines
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.

    # Input shape
        nD tensor with shape: `(nb_samples, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(nb_samples, input_dim)`.

    # Output shape
        nD tensor with shape: `(nb_samples, ..., output_dim)`.
        For instance, for a 2D input with shape `(nb_samples, input_dim)`,
        the output would have shape `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None, weights=None, k=2,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.k = k

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2')]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(FactorizationMachinesLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim=2)]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.V = self.add_weight((self.output_dim, input_dim, self.k),
                                 initializer=self.init,
                                 name='{}_V'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        output = K.sum(K.square(K.dot(x, self.V)) - K.dot(K.square(x), K.square(self.V)), 2)/2
        output += K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(FactorizationMachinesLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def Factoriza_model(train_x, l1_neural, l2_neural, l3_neural):

    model_in = ks.Input(shape=(train_x.shape[1],), dtype='float32')
    
    out = FactorizationMachinesLayer(l1_neural, )(model_in)
    out = Lambda(lambda x: clip(x, min_value=0.01, max_value=1.1))(out)
    out = ks.layers.Dropout(0.2)(out)
    
    out = ks.layers.Dense(l2_neural, activation='relu',  )(out)
    out = Lambda(lambda x: clip(x, min_value=0.01, max_value=1.1, ))(out)

    out = ks.layers.Dense (l3_neural, activation='relu', )(out)
    out = Lambda(lambda x: clip(x, min_value=0.01, max_value=1.1, ))(out)
    out = ks.layers.Dense(1)(out)
    out = Lambda(lambda x: clip(x, min_value=0, max_value=1))(out)
    model = ks.Model(model_in, out)
    model.compile(loss='binary_crossentropy', 
                  optimizer=ks.optimizers.Adam(lr=1e-3), 
                  metrics=['accuracy'])
    return model