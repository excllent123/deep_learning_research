import tensorflow as tf 

def lstm_dense_on_datetime(p_dateTime, hidden_dim, hpara):
	p_dateTime = tf.reshape(p_dateTime, shape=[hpara['batch_size'], hpara['his_window'], 5])
	p_dateTime = tf.contrib.keras.layers.LSTM(hidden_dim, input_shape=(hpara['his_window'] ,5))(p_dateTime)
	p_dateTime = tf.contrib.keras.layers.Dropout(0.2)(p_dateTime)
	res = tf.contrib.keras.layers.Dense(int(hidden_dim/2))(p_dateTime)
	return res

def lstm_dense_on_X(p_X, hidden_dim, hpara):
	p_X = tf.expand_dims(p_X, axis=2)
	p_X = tf.contrib.keras.layers.LSTM(hidden_dim, input_shape=(hpara['his_window'] ,1))(p_X)
	p_X = tf.contrib.keras.layers.Dropout(0.2)(p_X)
	res = tf.contrib.keras.layers.Dense(int(hidden_dim/2))(p_X)
	return res

def model_001(p_X, p_dateTime, p_tpcName, p_tpcFeature, hpara):

	hidden_dim = 100

	a = lstm_dense_on_X(p_X, hidden_dim, hpara)

	b = lstm_dense_on_datetime(p_dateTime, hidden_dim, hpara)

	a_b = tf.add(a, b)
	

	emb_weight = tf.get_variable('tcp_name_weight', shape=[35, hidden_dim])
	c = tf.nn.embedding_lookup(emb_weight, [p_tpcName])
	c = tf.reshape(c, shape=[hpara['batch_size'], 35* hidden_dim])
	c = tf.contrib.keras.layers.Dense(50)(c)

	out = tf.concat([a_b, c],1)
	out = tf.contrib.keras.layers.Dropout(0.33)(out)
	out = tf.contrib.keras.layers.Dense(hpara['predict_window'])(out)
	out = tf.contrib.keras.layers.Dropout(0.4)(out)
	out = tf.contrib.keras.layers.Dense(hpara['predict_window'])(out)
	return out

def model_002(p_X, p_dateTime, p_tpcName, p_tpcFeature, hpara):
	a = lstm_dense_on_X(p_X, int(1.5*hpara['his_window']), hpara)
	b = tf.contrib.keras.layers.Dense( int(1.5*hpara['his_window']/2) )(p_dateTime)
	a_b = tf.add(a, b)

	emb_weight = tf.get_variable('tcp_name_weight', shape=[35, hidden_dim])
	c = tf.nn.embedding_lookup(emb_weight, [p_tpcName])
	c = tf.reshape(c, shape=[hpara['batch_size'], 35* hidden_dim])
	c = tf.contrib.keras.layers.Dense(50)(c)

	out = tf.concat([a_b, c],1)

	out = tf.contrib.keras.layers.Conv1D(8, kernel_size=3, 
		padding='causal', activation='elu', dilation_rate=2)(out)
	
	out = tf.contrib.keras.layers.Conv1D(8, kernel_size=3, 
		padding='causal', activation='elu', dilation_rate=2)(out)

	out = tf.contrib.keras.layers.MaxPooling1D()(out)

	out = tf.contrib.keras.layers.Conv1D(8, kernel_size=3, 
		padding='causal', activation='elu', dilation_rate=2)(out)
	
	out = tf.contrib.keras.layers.Conv1D(8, kernel_size=3, 
		padding='causal', activation='elu', dilation_rate=2)(out)

	out = tf.contrib.keras.layers.MaxPooling1D()(out)

	out = tf.contrib.keras.layers.Dropout(0.33)(out)
	out = tf.contrib.keras.layers.Dense(hpara['predict_window'])(out)
	out = tf.contrib.keras.layers.Dropout(0.4)(out)
	out = tf.contrib.keras.layers.Dense(hpara['predict_window'])(out)
	return out
	
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

def log_seq_rmse(y, y_hat, sequence_lengths, max_sequence_length):
	return tf.log(sequence_rmse(y, y_hat, sequence_lengths, max_sequence_length))