from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from six.moves import xrange

from .. import common_attention
from .. import common_hparams
from .. import common_layers

from .. import registry
from .. import t2t_model

import tensorflow as tf 


def ffn_layer(x, hparams):
    '''Feed-forward layer in transformer 

    Args: 
      x: a Tensor of shape [batch_size, length, hparams.hidden_size]

      hparams : hyperparameters for model

    Returns:
      a Tensor of shape [ batch_size, length, hparams.hidden_size ]

    Note: 
      there are 3 kinds of ffn_layer
      - conv_hidden_relu
      - parameter_attention
      - conv_hidden_relu_with_speconv
      - none
    '''
    if hparams.ffn_layer == 'conv_hidden_relu':
        return common_layers.conv_hidden_relu(
                    x,
                    hparams.filter_size,
                    hparams.hidden_size,
                    dropout=hparams.relu_dropout)



def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, 
  averaging over the last dimension."""
  if filters is None:
    filters = x.get_shape()[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    scale = tf.get_variable(
        "layer_norm_scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
    if allow_defun:
      result = layer_norm_compute(x, tf.constant(epsilon), scale, bias)
      result.set_shape(x.get_shape())
    else:
      result = layer_norm_compute_python(x, epsilon, scale, bias)
    return result

def layer_norm_compute_python(x, epsilon, scale, bias):
  """Layer norm raw computation."""
  mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return norm_x * scale + bias

# in the model the residual connections are all implemented as this way  
def residual_fn(x, y):
  return common_layers.s(x + tf.nn.dropout(
      y, 1.0 - hparams.residual_dropout))




def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     kernel_size=(1, 1),
                     second_kernel_size=(1, 1),
                     summaries=True,
                     dropout=0.0,
                     **kwargs):
  """Hidden layer with RELU activation followed by linear projection."""
  name = kwargs.pop("name") if "name" in kwargs else None
  with tf.variable_scope(name, "conv_hidden_relu", [inputs]):
    if inputs.get_shape().ndims == 3:
      is_3d = True
      inputs = tf.expand_dims(inputs, 2)
    else:
      is_3d = False
    conv_f1 = conv if kernel_size == (1, 1) else separable_conv
    h = conv_f1(
        inputs,
        hidden_size,
        kernel_size,
        activation=tf.nn.relu,
        name="conv1",
        **kwargs)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    if summaries and not tf.get_variable_scope().reuse:
      tf.summary.histogram("hidden_density_logit",
                           relu_density_logit(
                               h, list(range(inputs.shape.ndims - 1))))
    conv_f2 = conv if second_kernel_size == (1, 1) else separable_conv
    ret = conv_f2(h, output_size, second_kernel_size, name="conv2", **kwargs)
    if is_3d:
      ret = tf.squeeze(ret, 2)
    return ret


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4.")
  inputs.set_shape([static_shape[0], None, None, static_shape[3]])
  # Add support for left padding.
  if "padding" in kwargs and kwargs["padding"] == "LEFT":
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    kwargs["padding"] = "VALID"
  # Special argument we use to force 2d kernels (see below).
  force2d = kwargs.get("force2d", True)

  def conv2d_kernel(kernel_size_arg, name_suffix):
    """Call conv2d but add suffix to name."""
    if "name" in kwargs:
      original_name = kwargs["name"]
      name = kwargs.pop("name") + "_" + name_suffix
    else:
      original_name = None
      name = "conv_" + name_suffix
    original_force2d = None
    if "force2d" in kwargs:
      original_force2d = kwargs.pop("force2d")
    result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
    if original_name is not None:
      kwargs["name"] = original_name  # Restore for other calls.
    if original_force2d is not None:
      kwargs["force2d"] = original_force2d
    return result

  # Manually setting the shape to be unknown in the middle two dimensions so
  # that the `tf.cond` below won't throw an error based on the convolution
  # kernels being too large for the data.
  inputs._shape = tf.TensorShape([static_shape[0], None, None, static_shape[3]])  # pylint: disable=protected-access
  if kernel_size[1] == 1 or force2d:
    # Avoiding the cond below can speed up graph and gradient construction.
    return conv2d_kernel(kernel_size, "single")
  return tf.cond(
      tf.equal(tf.shape(inputs)[2],
               1), lambda: conv2d_kernel((kernel_size[0], 1), "small"),
      lambda: conv2d_kernel(kernel_size, "std"))




##################################
# sinusoids position embbedding 


def get_timing_signal(length,
                      min_timescale=1,
                      max_timescale=1e4,
                      num_timescales=16):
  """Create Tensor of sinusoids of different frequencies.

  Args:
    length: Length of the Tensor to create, i.e. Number of steps.
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int

  Returns:
    Tensor of shape (length, 2*num_timescales)
  """
  positions = tf.to_float(tf.range(length))
  log_timescale_increment = (math.log(max_timescale / min_timescale) /
                             (num_timescales - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(inv_timescales, 0)
  return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

def add_timing_signal(x, min_timescale=1, max_timescale=1e4, num_timescales=16):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  This allows attention to learn to use absolute and relative positions.
  The timing signal should be added to some precursor of both the source
  and the target of the attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the depth dimension, padded with zeros to be the same depth as the input,
  and added into input.

  Args:
    x: a Tensor with shape [?, length, ?, depth]
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int <= depth/2

  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[1]
  depth = tf.shape(x)[3]
  signal = get_timing_signal(length, min_timescale, max_timescale,
                             num_timescales)
  padded_signal = tf.pad(signal, [[0, 0], [0, depth - 2 * num_timescales]])
  return x + tf.reshape(padded_signal, [1, length, 1, depth])
'''


tf.layer.

