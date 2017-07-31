'''
embedding_op

'''

import tensorflow as tf 


class SkipGram:
    pass

class CBOW:
    pass


class JustEmbedding:
	pass


class TSNE:
	'''
	# T-distribution Stochastic Neighbor Embedding 

	# similar conditional probability => q_j|i
	  only interested in modeling pairwise similarities, we set q_j|i = 0.

	# similarity of high dimensional space => p_j|i
	  giving perplexity 
	  log_2( perplexity ) = -[ sum_{j}[ p_j|i * log_{2}(p+j|i) ]] 

	# Kullback-Leibler divergences
	  KL(prob_a, prob_b) = Sum(prob_a * log(prob_a/prob_b))
	  The cross entropy H, on the other hand, is defined as:

	  H(prob_a, prob_b) = -Sum(prob_a * log(prob_b))

	  So, if you create a variable y = prob_a/prob_b, 
	  you could obtain the KL divergence by calling negative H(proba_a, y). 

	  In Tensorflow notation, something like:
	  KL = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(prob_a, y))

	[sklearn implementation](https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/manifold/t_sne.py#L497)
	
	X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    n = pdist(X_embedded, "sqeuclidean")
    n += 1.
    n /= degrees_of_freedom
    n **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(n / (2.0 * np.sum(n)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if len(P.shape) == 2:
        P = squareform(P)
    kl_divergence = 2.0 * np.dot(P, np.log(P / Q))

    return kl_divergence

    # perplexity : float, optional (default: 30)
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. The choice is not extremely critical since t-SNE
        is quite insensitive to this parameter.
	'''
	def __init__(self, data, perplexity=30):
		'''
		input data, as the feeding data, must be numpy-matrix
		'''
		if len(data.shape)!=2:
			raise 'the input data must be 2D tensor with (sample, features)'
		self.sample_size, self.feature_dim = data.shape
		self.init_op = tf.global_variables_initializer()
		self.perplexity = perplexity


	def train(self, lr, iteration, momentum, batch_size=None):
		'''
		the t-sne process is default at looking entire data set
		therefore, the batch_size is deault as the sample_size 

		however if the dataset is too large, we could set a batch_size
		this would impact a little bit on loss computation, the smaller the batch_size
		the larger effect/unsertaninty of the output
		'''
		X = tf.placeholder('float',(self.sample_size ,self.feature_dim))
		init_value = tf.random_normal([150,2]) * 0.0001#映射到二维空间
		Y = tf.Variable(init_value)

		q_ij = self.compute_q_ij(Y)
		
		p_ij = self.compute_p_ij(X)

		loss = self.kl_divergence(p_ij, q_ij)



	def compute_pair_distance(self, u, v):
		assert u.shape == v.shape
		#tf.pow( tf.add(u, tf.negative(v)), 2)
		tf.square(tf.sub(x1, x2))
		D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
		tfD = tf.reduce_sum(tf.sub(X, tf.transpose(X), axis=1))
		raise NotImplementedError()

	def test_pair_distance(X):
		sum_X = np.sum(np.square(X), 1);
		D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
		# ||a-b||^{2} = a^2+b^2-2ab
		X = tf.Variable(X)
		# tfD = tf.reduce_sum(tf.sub(X, tf.transpose(X), axis=1))
		#with tf.Session() as sess:
		#	sess.run(tf.global_variables_initializer())
			a = sess.run(tfD)
		print (a, D)

	def compute_q_ij(self, Y):
		assert Y.rank ==2 # sample , dim
		Y_ = tf.transpose(Y, [1,0])
		self_pair_distance = tf.add(tf.reduce_sum(tf.sub(Y, Y_), axis=1), 1)
		sum_PD = tf.reduce_sum(self_pair_distance)
		return tf.div(self_pair_distance, sum_PD)

	def compute_p_ij(self, X):
		assert X.rank == 2 # sample, dim
		X_ = tf.transpose(X, [1,0])

		self_pair_distance = tf.exp(
			                   tf.div(
			                     tf.reduce_sum(tf.sub(X, X_), axis=1), 2*sigma))
		raise NotImplementedError()


	def kl_divergence(self, p, q):
		'''
		This assumes that p and q are both 1-D tensors of floats, 
		of the same shape and for each their values sum to 1.
		'''
		return tf.reduce_sum(p * tf.log(p/q))

	def normalize(self, x):
		# 1-D normalize
		pass
		

def tsne(lr, data):
	'''
	'''
	# def distance 
	q_ij = 
	loss = tf.reduce_sum(  p_ij*tf.log( p_ij / q_ij)   )
	retun pass


def pair_loss(x1, x2, y):
    # Euclidean distance between x1,x2
    l2diff = tf.sqrt( tf.reduce_sum(tf.square(tf.sub(x1, x2)),
                                    reduction_indices=1))

    # you can try margin parameters
    margin = tf.constant(1.)     

    labels = tf.to_float(y)

    match_loss = tf.square(l2diff, 'match_term')
    mismatch_loss = tf.maximum(0., tf.sub(margin, tf.square(l2diff)), 'mismatch_term')

    # if label is 1, only match_loss will count, otherwise mismatch_loss
    loss = tf.add(tf.mul(labels, match_loss), \
                  tf.mul((1 - labels), mismatch_loss), 'loss_add')

    loss_mean = tf.reduce_mean(loss)
    return loss_mean

def Hbeta(D = np.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row 
	for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta);
	sumP = sum(P);
	H = np.log(sumP) + beta * np.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a 
	way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print "Computing pairwise distances..."
	(n, d) = X.shape;
	sum_X = np.sum(np.square(X), 1);
	D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
	P = np.zeros((n, n));
	beta = np.ones((n, 1));
	logU = np.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print "Computing P-values for point ", i, " of ", n, "..."

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf;
		betamax =  np.inf;
		Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	print "Mean value of sigma: ", np.mean(np.sqrt(1 / beta));
	return P;


# reference 
#     X = tf.placeholder('float',(150,150))
#     initial = tf.random_normal([150,2]) * 0.0001#映射到二维空间
#     Y = tf.Variable(initial)
#     A = tf.reduce_sum(Y*Y, axis=1)
#     A = tf.reshape(r, [-1, 1])
#     #pair wise distance
#     pairD = A - 2*tf.matmul(Y, tf.transpose(Y)) + tf.transpose(A) + 1.
#     qij = 1./pairD
#     sumq = tf.reduce_sum(qij,axis=1)
#     qij /= sumq
#     loss = tf.reduce_sum( X*tf.log(X / qij) )
#     global_step = tf.Variable(0, name = 'global_step',trainable=False)
#     starter_learning_rate = 0.1
#     learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,20, 0.95, staircase=True)
#     train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(los