'''
embedding_op

auto-encoder 

autoencoders use supervised learning where the target is the same as the input. 
Hence autoencoder, as an encoding itself. 
word2vec seems to be using target words that are not the same as the input words. 
So although similar, it is not an autoencoder.

If linear activations are used, or only a single sigmoid hidden layer, 
then the optimal solution to an autoencoder is strongly related to 
principal component analysis (PCA).[10]
'''

import tensorflow as tf 
import numpy as np 


class Embedding_OP(object):
    pass


class SkipGram(Embedding_OP):
    pass

class CBOW(Embedding_OP):
    pass

class JustEmbedding(Embedding_OP):
    pass



class TSNE:
    '''# T-distribution Stochastic Neighbor Embedding 
      # The perplexity of a discrete probability distribution p is defined as 
      {\displaystyle 2^{H(p)}=2^{-\sum _{x}p(x)\log _{2}p(x)}} 2^{{H(p)}}=2^{{-\sum _{x}p(x)\log _{2}p(x)}}

      where H(p) is the entropy (in bits) of the distribution and x ranges over events.

      Perplexity of a random variable X may be defined as the perplexity of the distribution over its possible values x.

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

      # The disadvantages to using t-SNE are roughly:
      
      1. t-SNE is computationally expensive, and can take several hours 
         on million-sample datasets where PCA will finish in seconds or minutes

      2. The Barnes-Hut t-SNE method is limited to two or three dimensional embeddings.

      3. The algorithm is stochastic and multiple restarts with different 
         seeds can yield different embeddings. However, it is 
         perfectly legitimate to pick the embedding with the least error.

      4. Global structure is not explicitly preserved. 
         This is problem is mitigated by initializing points with PCA (using init=’pca’).

      “Visualizing High-Dimensional Data Using t-SNE” 
      van der Maaten, L.J.P.; Hinton, G. Journal of Machine Learning Research (2008)

      “t-Distributed Stochastic Neighbor Embedding”
      van der Maaten, L.J.P.

      “Accelerating t-SNE using Tree-Based Algorithms.”
      L.J.P. van der Maaten. Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
      '''

    def __init__(self, data, perplexity=30.):
        '''
        input data, as the feeding data, must be numpy-matrix
        '''
        if len(data.shape)!=2:
            raise 'the input data must be 2D tensor with (sample, features)'
        self.sample_size, self.feature_dim = data.shape
        self.init_op = tf.global_variables_initializer()
        self.perplexity = perplexity
        self.tolerance = 1e-5
        self.data = data

    def train(self, lr=1e-3, max_step=5000, batch_size=None):
        '''
        the t-sne process is default at looking entire data set
        therefore, the batch_size is deault as the sample_size 

        however if the dataset is too large, we could set a batch_size
        this would impact a little bit on loss computation, the smaller the batch_size
        the larger effect/unsertaninty of the output
        '''
        X = tf.placeholder('float32',(self.sample_size ,self.feature_dim))
        
        #p_ij = self.compute_p_ij(X)
        p_ij , loss1_op = self.tf_compute_p_ij(X)

        optimize1 = tf.train.AdamOptimizer(lr)
        train1_op = optimize1.minimize(loss1_op)


        init_value = tf.random_normal([self.sample_size,2])*0.001+1.
        Y = tf.Variable(init_value)

        q_ij = self.compute_q_ij(Y)

        optimize2 = tf.train.AdamOptimizer(1e-5)
        loss2_op = self.kl_divergence(p_ij, q_ij)
        #loss2_op = tf.reduce_sum( p_ij*tf.log(p_ij / q_ij) )
        train2_op = optimize2.minimize(loss2_op)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # rather than use binary search, here we use updating way to 
            # find beta 
            step=1
            while step < 100:
                step+=1
                loss, _ = sess.run([loss1_op, train1_op], feed_dict={X:self.data})
                if step%100==0:
                    print ('p_ij selection : ', loss)

            print ('YYYY',sess.run(Y))
            step=0
            while step < max_step:
                step+=1
                y, loss, _ = sess.run([Y, loss2_op, train2_op], feed_dict = {X:self.data})
                if step %100==0:

                    print ('@@@@Y', y[1])
                    print ('LOSS:', loss)

    def compute_q_ij(self, Y):
        Y_ = tf.transpose(Y,[1,0])
        sumY =  tf.reduce_sum(tf.square(Y),1)
        D = sumY -2.* tf.tensordot( Y, Y_, axes=[[1],[0]]) + sumY+1
        return 1./D/sumY

    def tf_compute_p_ij(self, X):
        '''
        [*] X could be numpy-array also a placeholder
        '''
        assert len(X.shape) == 2 # sample, dim

        n, dim = X.shape
        X_ = tf.transpose(X, [1,0])

        # i with j_for j in range(n) & thus reduce_sum on j-axis, which is 1
        sumX = tf.reduce_sum(tf.square(X), 1)

        # tensordot: axes order matters [ [1], [0] ] is not equal to [ [0],[1] ]
        # D shape = (n, n) represent p_ij
        pariDistance = tf.add( tf.add( sumX, tf.multiply(-2., 
                             tf.tensordot( X, X_, axes=[[1],[0]])
                             )), sumX)

        # ======================================================
        # Rather than use binary-search, we use gradient-decent 
        # for variance-calculate
        # ======================================================
        
        # log(perplexity) ~ Entropy
        targetEntropy = tf.log(self.perplexity)

        beta = tf.Variable(tf.ones(shape=(n,1))) # a tuning parameter for variance_i

        Entropy = self.compute_entropy(pariDistance, beta)

        loss_op = tf.reduce_sum(tf.abs(tf.subtract(Entropy, targetEntropy)))
        return pariDistance, loss_op

    def compute_entropy(self, D, beta):
        '''
        where D is Di(row) from p_ij
        '''
        P    = tf.exp(-D * beta)
        sumP = tf.reduce_sum(P)
        H = tf.add(tf.log(sumP), tf.multiply(beta, tf.div(tf.reduce_sum(D * P), sumP)))
        return H

    def compute_p_ij(self, X = np.array([]), tol = 1e-5, perplexity = 30.0):
        """Performs a binary search to get P-values in such a 
        way that each conditional Gaussian has the same perplexity.
        Ref from source-tsne-python"""    

        # Initialize some variable
        (n, d) = X.shape
        print ("Computing pairwise distances with {}".format( str(n)+'-'+str(d) ))
        sum_X = np.sum(np.square(X))
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);    

        P = np.zeros((n, n));
        beta = np.ones((n, 1));
        logU = np.log(perplexity);    

        # Loop over all datapoints
        for i in range(n):    

            # Print progress
            if i % 500 == 0:
                print ("Computing P-values for point ", i, " of ", n, "..."  )  

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf;
            betamax =  np.inf;
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]; # avoid Dii where is zero
            (H, thisP) = self.Hbeta(Di, beta[i]);    

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

            # Set the final row of P # Use np.r_ to set P_ii = zeros
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;    

        # Return final P-matrix
        print ("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
        return P;

    def kl_divergence(self, p, q):
        '''
        [!] This assumes that p and q are both 1-D tensors of floats, 
        of the same shape and for each their values sum to 1.
        '''
        # avoid explotion by setting p>0
        p = tf.maximum(0.000001, p)/tf.reduce_sum(p)
        q /= tf.reduce_sum(q)
        return tf.reduce_sum(p * tf.log(p/q))


    def Hbeta(self, D = np.array([]), beta = 1.0):
        """Compute the perplexity and the P-row 
           for a specific value of the precision of a Gaussian distribution.
           ref : from source
           """    

        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta);
        sumP = sum(P);
        H = np.log(sumP) + beta * np.sum(D * P) / sumP;
        P = P / sumP;
        return H, P

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

if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    tsne = TSNE(X)

    tsne.train()



    # init TSNE



# code sample
#with tf.device('/cpu:0'):
#    X = tf.placeholder('float',(150,150))
#    initial = tf.random_normal([150,2]) * 0.0001# Mapped to 2D space 
#    Y = tf.Variable(initial)
#    A = tf.reduce_sum(Y*Y, axis=1)
#    A = tf.reshape(r, [-1, 1])
#    #pair wise distance
#    pairD = A - 2*tf.matmul(Y, tf.transpose(Y)) + tf.transpose(A) + 1.
#    qij = 1./pairD
#    sumq = tf.reduce_sum(qij,axis=1)
#    qij /= sumq
#    loss = tf.reduce_sum( X*tf.log(X / qij) )
#    global_step = tf.Variable(0, name = 'global_step',trainable=False)
#    starter_learning_rate = 0.1
#    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,20, 0.95, staircase=True)
#    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss,global_step = global_step) 