'''

@ latent semantic analysis (LSA)
  - A technique in natural language processing, in particular distributional semantics, 
    of analyzing relationships between a set of documents and 
    the terms they contain by producing a set of concepts related to the documents and terms. 

  - LSA assumes that words that are close in meaning will occur in similar pieces of text 
    (the distributional hypothesis). 
    A matrix containing word counts per paragraph 
    (rows represent unique words and columns represent each paragraph) 
    is constructed from a large piece of text and
    singular value decomposition (SVD) is used to reduce 
    the number of rows while preserving the similarity structure among columns. 

  - Words are then compared by taking the cosine of the angle between the two vectors 
    (or the dot product between the normalizations of the two vectors)
    formed by any two rows. 
  - Values close to 1 represent very similar words 
    while values close to 0 represent very dissimilar words.[1]
    An information retrieval technique using latent semantic structure 
    was patented in 1988 (US Patent 4,839,853, now expired) 

@ Singular Value Decomposition (SVD)
  -  In linear algebra, an orthogonal matrix or real orthogonal matrix 
     is a square matrix with real entries 
    whose columns and rows are orthogonal unit vectors 
    (i.e., orthonormal vectors), i.e.

@ a complex square matrix U is Unitary 
  - if its conjugate transpose U∗ is also its inverse 

@ The real analogue of a unitary matrix is an orthogonal matrix. 
  Unitary matrices have significant importance in quantum mechanics 
  because they preserve norms, and thus, probability amplitudes.

@ orthogonal matrix 
  - if M is a orthogonal matrix 
  - then transpose(M) == Invert(M)

@ In mathematics, a unitary transformation is a transformation 
  that preserves the inner product: 
  the inner product of two vectors before the transformation 
  is equal to their inner product after the transformation.

@ 如何求反死朕（反矩陣）
  - Cayley-Hamilton定理，這定理敍述一矩陣A滿足其特徵方程式，也就是說如果A的特徵方程式為 
    a0+a1*x+a2*x^2+....+an*x^n=0 
    那麼如果A的反矩陣存在，則a0不等於0，將A,I矩陣代入上面的式子 
    a0*I+a1*A+a2*A^2+....+an*A^n=0 
    a0*I=-a1*A-a2*A^2-....-an*A^n 
    A的反矩陣=(-a1*I-a2*A^1-....-an*A^(n-1))/a0 
  
  - 公式法求反矩陣：
    想法 A A-1 = I ，其中的第一條 [A-1]n1 的部分，有 [A] [A-1]n1 = [I]n1，
    此時 [A-1]n1 是未知數，A 及 I 是已知。
    即可套用公式解 (Cramer's rule) 求整組 n 個 [A-1]n1 值
    [A-1]jk = cofactor( [A]kj ) / det([A])
    其中 cofactor 翻譯作餘因子， cofactor( [A]kj ) 定義為  [A] 去掉了 k 列、j 行後的行列式值。
    從這個解法可看出，若方陣 A 之行列式值不為零，A 有反矩陣。

@ 行列式的幾何意義
  - 則是上述 n 個向量，在空間中所張開來之 n 維體積值。
    若這 n 個向量形成維度簡併（如三維時之共面），的情形，則 n 維體積值為零。
    此一（共面向量組體積為零之）觀念常用在線性方程組求解問題之唯一解、無解或無限多解之判定。

@ Determinants are mathematical objects that are very useful 
  in the analysis and solution of systems of linear equations. 
  As shown by Cramer's rule, a nonhomogeneous system of linear equations
  has a unique solution iff the determinant of the system's matrix is nonzero 
  (i.e., the matrix is nonsingular).

@ 幾個很有用的公式
  (AB)T = BTAT
  (AB)-1 = B-1A-1
  det(AB) = det(A) det(B)
  tr(B-1AB) = tr(A)
  det(B-1AB) = det(A)
  det(AT) = det(A)


@ 方陣的行列值(determinant)為零時稱為奇異(singular)；
  反之，行列值不為零時稱為非奇異(nonsingular)。

@ 非奇異方陣亦稱為可逆方陣(invertible matrix)，因為恆有唯一的逆陣(inverse)存在。
  非奇異方陣中，各行與各列均為線性獨立，是一個滿秩(rank)的方陣。

@ invert a matrix that may be singular or ill-conditioned. 
I understand it is pythonic to simply do this:

@ A matrix is ill-conditioned if the condition number is too large
@ A matrix is singular        if the condition number is too infinite
@ The system on the left has solution x = 2, y = 0 
  while the one on the right has solution x = 1, y = 1. 
  The coefficient matrix is called ill-conditioned because a small change in the 
  constant coefficients results in a large change in the solution.

@ which is called the determinant for this system of equation. 
  Determinants are defined only for square matrices.

@ 一般線性代數的書在談論到矩陣時，都會特別說明，若是一個矩陣沒有反矩陣，
  它就被稱做singular matrix。
  一般來說，只要是該矩陣的determinant等於或非常非常接近0，它就不存在或很
  難用一般的方法求出反矩陣。

  就實際應用面而言，一般會想求反矩陣，並不是真的對那個矩陣的反矩陣有興趣
  ，通常都是為了解之後的問題，必須在過程中先求得反矩陣。
  舉例來說，最常遇到的就是解聯立方程式。

@ Determinants is commonly denoted det(A), |A|, 

@ A square matrix that does not have a matrix inverse. 
@ A matrix is singular iff its determinant is 0

求值　　evaluation
求解　　resolution　（除法division）
加法　　addition
減法　　subtraction
倍率　　scaling
複合  　composition　（乘法multiplication）
分解  　decomposition
疊代  　iteration　　（次方exponentiation）
反函數　inverse
轉置　　transpose
秩　　　rank
行列式　determinant
跡　　　trace
積和式　permanent
微分　　differentiation
範數　　norm
梯度　　gradient

@ tf-idf (term frequency–inverse document frequency)
    - the weight of an element of the matrix is proportional 
      to the number of times the terms appear in each document,
      where rare terms are upweighted to reflect their relative importance.


@ Kullback–Leibler divergence or KL-Distance
    - 相對熵（relative entropy）又稱為KL散度（Kullback–Leibler divergence，簡稱KLD）
      [1]，信息散度（information divergence），信息增益（information gain）。
    - scipy.stats.entropy(pk, qk=None, base=None)
    - 當且僅當P = Q時DKL(P||Q)為零。



@ DAG with NLP-machine, the machine 

@ KL散度和其它量的關係[編輯]
  - 自信息（en:self-information）和KL散度
    {\displaystyle I(m)=D_{\mathrm {KL} }(\delta _{im}\|\{p_{i}\}),} I(m)=D_{{{\mathrm  {KL}}}}(\delta _{{im}}\|\{p_{i}\}),

  - 互信息（en:Mutual information）和KL散度 
    {\displaystyle {\begin{aligned}I(X;Y)&=D_{\mathrm {KL} }(P(X,Y)\|P(X)P(Y))\\&=\mathbb {E} _{X}\{D_{\mathrm {KL} }(P(Y|X)\|P(Y))\}\\&=\mathbb {E} _{Y}\{D_{\mathrm {KL} }(P(X|Y)\|P(X))\}\end{aligned}}} {\begin{aligned}I(X;Y)&=D_{{{\mathrm  {KL}}}}(P(X,Y)\|P(X)P(Y))\\&={\mathbb  {E}}_{X}\{D_{{{\mathrm  {KL}}}}(P(Y|X)\|P(Y))\}\\&={\mathbb  {E}}_{Y}\{D_{{{\mathrm  {KL}}}}(P(X|Y)\|P(X))\}\end{aligned}}

  - 信息熵（en: Shannon entropy）和KL散度
    {\displaystyle {\begin{aligned}H(X)&=\mathrm {(i)} \,\mathbb {E} _{x}\{I(x)\}\\&=\mathrm {(ii)} \log N-D_{\mathrm {KL} }(P(X)\|P_{U}(X))\end{aligned}}} {\begin{aligned}H(X)&={\mathrm  {(i)}}\,{\mathbb  {E}}_{x}\{I(x)\}\\&={\mathrm  {(ii)}}\log N-D_{{{\mathrm  {KL}}}}(P(X)\|P_{U}(X))\end{aligned}}

  - 條件熵（en:conditional entropy）和KL散度
    {\displaystyle {\begin{aligned}H(X|Y)&=\log N-D_{\mathrm {KL} }(P(X,Y)\|P_{U}(X)P(Y))\\&=\mathrm {(i)} \,\,\log N-D_{\mathrm {KL} }(P(X,Y)\|P(X)P(Y))-D_{\mathrm {KL} }(P(X)\|P_{U}(X))\\&=H(X)-I(X;Y)\\&=\mathrm {(ii)} \,\log N-\mathbb {E} _{Y}\{D_{\mathrm {KL} }(P(X|Y)\|P_{U}(X))\}\end{aligned}}} {\begin{aligned}H(X|Y)&=\log N-D_{{{\mathrm  {KL}}}}(P(X,Y)\|P_{U}(X)P(Y))\\&={\mathrm  {(i)}}\,\,\log N-D_{{{\mathrm  {KL}}}}(P(X,Y)\|P(X)P(Y))-D_{{{\mathrm  {KL}}}}(P(X)\|P_{U}(X))\\&=H(X)-I(X;Y)\\&={\mathrm  {(ii)}}\,\log N-{\mathbb  {E}}_{Y}\{D_{{{\mathrm  {KL}}}}(P(X|Y)\|P_{U}(X))\}\end{aligned}}

  - 交叉熵（en:cross entropy）和KL散度
    {\displaystyle \mathrm {H} (p,q)=\mathrm {E} _{p}[-\log q]=\mathrm {H} (p)+D_{\mathrm {KL} }(p\|q).\!} {\mathrm  {H}}(p,q)={\mathrm  {E}}_{p}[-\log q]={\mathrm  {H}}(p)+D_{{{\mathrm  {KL}}}}(p\|q).\!




'''

# discrited - ids => 

class Node(object):
  '''
  # Description:
    - basic node for constructing The Machine 
    - the constructed graph should support asynchronous learning   
  '''
  def __init__(self, *args, **kwargs):
    self.operation_set = []
    self.input_set     = []
    self.output_set    = []

class DataType(object):
  '''
  TypeTree
  - Natural Language 
  - Image 
  - Audio 
  - Continuous Tensor
  - Discrite Tensor 
  '''
  def __init__(self, *args , **kwargs):
    pass


class DataPreprocess(object):
    pass

