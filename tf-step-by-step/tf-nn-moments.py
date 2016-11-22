def moments(x, axes, shift=None, name=None, keep_dims=False):
  """Calculate the mean and variance of `x`.
  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.
  When using these moments for batch normalization (see
  `tf.nn.batch_normalization`):
   * for so-called "global normalization", used with convolutional filters with
     shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
   * for simple batch normalization pass `axes=[0]` (batch only).
  Args:
    x: A `Tensor`.
    axes: Array of ints.  Axes along which to compute mean and
      variance.
    shift: A `Tensor` containing the value by which to shift the data for
      numerical stability, or `None` if no shift is to be performed. A shift
      close to the true mean provides the most numerically stable results.
    name: Name used to scope the operations that compute the moments.
    keep_dims: produce moments with the same dimensionality as the input.
  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  with ops.name_scope(name, "moments", [x, axes, shift]):
    # The dynamic range of fp16 is too limited to support the collection of
    # sufficient statistics. As a workaround we simply perform the operations
    # on 32-bit floats before converting the mean and variance back to fp16
    y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x
    shift = math_ops.cast(shift, dtypes.float32) if (
        shift is not None and x.dtype == dtypes.float16) else shift
    counts, m_ss, v_ss, shift = sufficient_statistics(
        y, axes, shift=shift, keep_dims=keep_dims, name=name)
    with ops.control_dependencies([counts, m_ss, v_ss]):
      mean, variance = normalize_moments(counts, m_ss, v_ss, shift, name=name)
      if x.dtype == dtypes.float16:
        return (math_ops.cast(mean, dtypes.float16),
                math_ops.cast(variance, dtypes.float16))
      else:
        return (mean, variance)


def sufficient_statistics(x, axes, shift=None, keep_dims=False, name=None):
  """Calculate the sufficient statistics for the mean and variance of `x`.
  These sufficient statistics are computed using the one pass algorithm on
  an input that's optionally shifted. See:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data
  Args:
    x: A `Tensor`.
    axes: Array of ints. Axes along which to compute mean and variance.
    shift: A `Tensor` containing the value by which to shift the data for
      numerical stability, or `None` if no shift is to be performed. A shift
      close to the true mean provides the most numerically stable results.
    keep_dims: produce statistics with the same dimensionality as the input.
    name: Name used to scope the operations that compute the sufficient stats.
  Returns:
    Four `Tensor` objects of the same type as `x`:
    * the count (number of elements to average over).
    * the (possibly shifted) sum of the elements in the array.
    * the (possibly shifted) sum of squares of the elements in the array.
    * the shift by which the mean must be corrected or None if `shift` is None.
  """
  axes = list(set(axes))
  with ops.name_scope(name, "sufficient_statistics", [x, shift]):
    x = ops.convert_to_tensor(x, name="x")
    x_shape = x.get_shape()
    if x_shape.is_fully_defined():
      counts = 1
      for d in axes:
        counts *= x_shape[d].value
      counts = constant_op.constant(counts, dtype=x.dtype)
    else:  # shape needs to be inferred at runtime.
      x_dims = array_ops.gather(array_ops.shape(x), axes)
      counts = math_ops.cast(
          math_ops.reduce_prod(x_dims), x.dtype, name="count")
    if shift is not None:
      shift = ops.convert_to_tensor(shift, name="shift")
      m_ss = math_ops.sub(x, shift)
      v_ss = math_ops.squared_difference(x, shift)
    else:  # no shift.
      m_ss = x
      v_ss = math_ops.square(x)
    m_ss = math_ops.reduce_sum(m_ss, axes, keep_dims=keep_dims, name="mean_ss")
    v_ss = math_ops.reduce_sum(v_ss, axes, keep_dims=keep_dims, name="var_ss")
  return counts, m_ss, v_ss, shift

