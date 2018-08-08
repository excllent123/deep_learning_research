

import tensorflow as tf 
import numpy as np
from util import OperationTag

class EmbeddingTesting():
    def __init__(self, col, X):
        self.data = col
        # notice size ..
        self.col_set_dim = 2*int(max( [max(list(col)), len(col) ]))
        self.sess = tf.Session
        self.graph = tf.Graph()

    def embdding_ly(self,name, hidded_dims, input_id):
        '''embdding 2D matrix, 1D input '''
        assert hidded_dims > self.col_set_dim
        emb = tf.get_variable(name, shape=(self.col_set_dim, hidded_dims))
        return tf.nn.embedding_lookup(emb, input_id)

    def concate_emb(self, input_id):
        emb1 = self.embdding_ly('A', 40, input_id)
        emb2 = self.embdding_ly('B', 45, input_id)
        return tf.concat([emb1, emb2], axis=1)
    
    def add_emb(self, input_id):
        emb1 = self.embdding_ly('1A', 40, input_id)
        emb2 = self.embdding_ly('1B', 40, input_id)
        return emb1 + emb2

    def prod_emb(self, input_id):
        emb1 = self.embdding_ly('2A', 45, input_id)
        emb2 = self.embdding_ly('2B', 45, input_id)
        return emb1 * emb2   

    def run(self):
        '''
        while any operator contrain variable initializer
        it must have variable under same code piece for 
        tf.global_variables_initializer to initializer... 
        '''
        # [ batch_size, windows, features ]
        input_pld = tf.placeholder(dtype=tf.int32, shape=[None])

        # 

        # [ batch_size, windows, features(date, weekend, weekday, ... etc)]
        pld_datetime = tf.placeholder(dtype=tf.int32)

        # [ batch_size, ]
        pld_name = tf.placeholder(tf.int32)

        out = self.embdding_ly('T', 100, input_pld)

        out2 = self.concate_emb(input_pld)
        out3 = self.add_emb(input_pld)
        out4 = self.prod_emb(input_pld)

        
        def test(obj):
            print (sess.run(obj , 
                feed_dict={input_pld : self.data})[0].shape )

        with self.sess() as sess:
            sess.run(tf.global_variables_initializer())
            print (sess.run(input_pld, feed_dict={input_pld:self.data}) )
            
            with OperationTag('ADD'): test(out3)
            with OperationTag('Product'): test(out4)





class TimeSeriesEmbedding():
    def __init__(self):
        pass

    def data_prepare():
        # Batch-Free-Feature
        self.date     = np.load(feature) # depth 31 [0~30]
        self.month    = np.load(feature) # depth 12 [0~11]
        self.year     = np.load(feature) # depth 4 [2015, 2016, 2017, 2018]

        # Order_embedding by window_size
        self.order    = np.load(feature)
        self.nlp_emb  = np.load()

        # Sample-Feature

    def embdding_ly(self, name, hidded_dims, input_id):
        '''embdding 2D matrix, 1D input '''
        assert hidded_dims > self.col_set_dim
        emb = tf.get_variable(name, shape=(self.col_set_dim, hidded_dims))
        return tf.nn.embedding_lookup(emb, input_id)


    def get_input_data(self):
        date  = self.embdding_ly('date', 40, date_pl_holder)
        month = self.embdding_ly('month', )

        
if __name__ == '__main__':
    col = [1,3,4,1,2,3,4,1,13,2,5,6,7]
    print (int(max( [max(list(col)), len(col) ])))
    Agent = EmbeddingTesting(col, [])
    Agent.run()


def lstm_layer(inputs, lengths, state_size, keep_prob=1.0, 
    scope='lstm-layer', reuse=False, return_final_state=False):
    """
    LSTM layer.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        lengths: Tensor of shape [batch size].
        state_size: LSTM state size.
        keep_prob: 1 - p, where p is the dropout probability.

    Returns:
        Tensor of shape [batch size, 
                         max sequence length, 
                         state_size] containing the lstm
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


def temporal_convolution_layer(inputs, output_units, 
    convolution_width, causal=False, dilation_rate=[1], bias=True,
    activation=None, dropout=None, scope='causal-conv-layer', reuse=False):
    """
    Convolution over the temporal axis of sequence data.

    Args:
        inputs: Tensor of shape [batch size, max sequence length, input_units].
        output_units: Output channels for convolution.
        convolution_width: Number of timesteps to use in convolution.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        if causal:
            shift = (convolution_width / 2) + (int(dilation_rate[0] - 1) / 2)
            pad = tf.zeros([tf.shape(inputs)[0], shift, inputs.shape.as_list()[2]])
            inputs = tf.concat([pad, inputs], axis=1)

        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[convolution_width, shape(inputs, 2), output_units]
        )

        z = tf.nn.convolution(inputs, W, padding='SAME', dilation_rate=dilation_rate)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        z = z[:, :-shift, :] if causal else z
        return z


def time_distributed_dense_layer(inputs, output_units, bias=True, 
    activation=None, batch_norm=None,
    dropout=None, scope='time-distributed-dense-layer', reuse=False):
    """
    Applies a shared dense layer to each timestep of a tensor of shape 
    [batch_size, max_seq_len, input_units]
    to produce a tensor of shape [batch_size, max_seq_len, output_units].

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.einsum('ijk,kl->ijl', inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z


def dense_layer(inputs, output_units, bias=True, 
    activation=None, batch_norm=None, dropout=None,
    scope='dense-layer', reuse=False):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].

    Args:
        inputs: Tensor of shape [batch size, input_units].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, output_units].

    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            shape=[shape(inputs, -1), output_units]
        )
        z = tf.matmul(inputs, W)
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b

        if batch_norm is not None:
            z = tf.layers.batch_normalization(z, training=batch_norm, reuse=reuse)

        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout is not None else z
        return z

### unfix_sequence_loss
def sequence_log_loss(y, y_hat, sequence_lengths, max_sequence_length, eps=1e-15):
    y = tf.cast(y, tf.float32)
    y_hat = tf.minimum(tf.maximum(y_hat, eps), 1.0 - eps)
    log_losses = y*tf.log(y_hat) + (1.0 - y)*tf.log(1.0 - y_hat)
    # because output is a sequence but input sequence is padded 
    sequence_mask = tf.cast(tf.sequence_mask(sequence_lengths, maxlen=max_sequence_length), tf.float32)
    avg_log_loss = -tf.reduce_sum(log_losses*sequence_mask) / tf.cast(tf.reduce_sum(sequence_lengths), tf.float32)
    return avg_log_loss


def sequence_rmse(y, y_hat, sequence_lengths, max_sequence_length):
    y = tf.cast(y, tf.float32)
    squared_error = tf.square(y - y_hat)
    # because output is a sequence but input sequence is padded 
    sequence_mask = tf.cast(
        tf.sequence_mask(sequence_lengths, maxlen=max_sequence_length),
                    tf.float32)

    avg_squared_error = tf.reduce_sum(squared_error*sequence_mask) / tf.cast(
        tf.reduce_sum(sequence_lengths), tf.float32)
    return tf.sqrt(avg_squared_error)

def log_loss(y, y_hat, eps=1e-15):
    y = tf.cast(y, tf.float32)
    y_hat = tf.minimum(tf.maximum(y_hat, eps), 1.0 - eps)
    log_loss = -tf.reduce_mean(y*tf.log(y_hat) + (1.0 - y)*tf.log(1.0 - y_hat))
    return log_loss


def rank(tensor):
    """Get tensor rank as python list"""
    return len(tensor.shape.as_list())

def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]



PLD = tf.placeholder

# where windows = 
# [ batch_size, windows, features ] 
pld_visit    = PLD(dtype=tf.int32, shape=[None, windows, dim_feature])

# [ batch_size, windows, features(date, weekend, weekday, ... etc)]
pld_datetime = PLD(dtype=tf.int32, shape=[None, windows, dim_feature])

# [ batch_size, windows]
pld_order    = PLD(dtype=tf.int32, shape=[None, windows])

# [ batch_size, ]
pld_name     = PLD(dtype=tf.int32, shape=[None, max_nlp_leng])

# [ batch_size, 1] 
pld_target   = PLD(dtype=tf.float32, shape=[None])


tf.concat(pld_visit, emb(pld_datetime))


def zero_1d_pad(x, max_sequence_length):
    return x


def gen_data(max_iteration, batch_size):
    i                  = 0 
    max_history_window = 200 # need zero_pad
    predict_window     = 30
    day_gap            = 10  # due to the 2 stages 
    all_timeseries_len = 551

    # dim_sample = population 
    select = random.sample(range(0,dim_sample), batch_size)

    start_window_id = np.random.randint(
        max_history_window, high= (
            all_timeseries_len - predict_window - day_gap))

    while max_iteration > i :
        batch_x      = []
        batch_x_time = []
        batch_x_name = []
        batch_y = []
        i += 1
        batch = 0
        while batch < batch_size:
            batch+=1

def test_data_generator(max_iteration, batch_size):
    pass

def valid_data_generator(max_iteration, batch_size):
    pass

