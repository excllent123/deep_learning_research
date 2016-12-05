from __future__ import absolut_import
from __future__ import division
from __future__ import print_function
import collection
import math
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def negative_log_likehood(x):
    SUM=0
    for individual in x:
        SUM+=-np.log(x)
    return SUM

"""
embedding_lookup function retrieves rows of the params tensor.
The behavior is similar to using indexing with arrays in numpy. E.g.

matrix = np.random.random([1024, 64])  # 64-dimensional embeddings
ids = np.array([0, 5, 17, 33])
print matrix[ids]  # prints a matrix of shape [4, 64]

params argument can be also a list of tensors
in which case the ids will be distributed among the tensors.

E.g. given a list of 3

[2, 64] tensors the default behavior is that
they will represent ids: [0, 3], [1, 4], [2, 5].

partition_strategy controls the way how the ids are distributed among the list.
The partitioning is useful for larger scale problems
when the matrix might be too large to keep in one piece."""


