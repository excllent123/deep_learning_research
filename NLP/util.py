import tensorflow as tf 
import time

class OperationTag():
    def __init__(self, op_name, verbose=1):
        '''Des
         - use to registrating operation and evaluation time comsuming
         - if verbose == 0, it would not do evaluation
        '''
        self.op_name = op_name
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print('========================================')
            print('[*] start [{}] job'.format(self.op_name))
            self.s_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose:
            print('[X] Time Comsuming {} ms'.format( 
            	str( round(1000*(time.time() - self.s_time) ,4))))

class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'user_id',
            'aisle_id',
            'department_id',
            'eval_set',
            'is_ordered_history',
            'index_in_order_history',
            'order_dow_history',
            'order_hour_history',
            'days_since_prior_order_history',
            'order_size_history',
            'order_number_history',
            'num_products_from_aisle_history',
            'history_length',
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]
        self.test_df = DataFrame(columns=data_cols, data=data)

        print ( self.test_df.shapes() )
        print ( 'loaded data' )

        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.9)

        print ( 'train size', len(self.train_df) )
        print ( 'val size', len(self.val_df) )
        print ( 'test size', len(self.test_df) )

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=False,
            num_epochs=1,
            is_test=True
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        batch_gen = df.batch_generator(batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)
        for batch in batch_gen:
            batch['order_dow_history']              = np.roll(batch['order_dow_history'], -1, axis=1)
            batch['order_hour_history']             = np.roll(batch['order_hour_history'], -1, axis=1)
            batch['days_since_prior_order_history'] = np.roll(batch['days_since_prior_order_history'], -1, axis=1)
            batch['order_number_history']           = np.roll(batch['order_number_history'], -1, axis=1)
            batch['next_is_ordered']                = np.roll(batch['is_ordered_history'], -1, axis=1)
            if not is_test:
                batch['history_length'] = batch['history_length'] - 1
            yield batch

class Model_1():
	def __init__():
		pass

	def get_input_tensor():
		'''
		Data 

		X : { name_emb }
		'''