# Includes some special purpose functions for "explaining" how various forms of matrix multiplication
# produce the results that they do.
# For example, you can obtain zero or near-zero results in a matmul operation for a number of reasons:
#  - everything on both inputs could be near-zero
#  - everything in just one input could be near-zero while the other had only large positive values
#  - both inputs could be a mixture of near-zeros and positive values that overlap in such a way that
#    you never get a combination of positive * positive.
#  - both inputs could contain large values, but some are negative, and they cancel out to produce a near-zero value.
#
# The functions here group the computations in such a way that makes it possibly to easily classify the results
# of a given computation to explain its output.
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def summarise(counts, sums=None):
    """
    Generates a concise summary text from the result of calling matmul_classify() or any of its variants.

    By default, lists the components ordered by largest count first.

    Example usages:
    > counts, sums = matmul_classify(a, b)
    > summarise(counts, sums)
    >
    > summarise(matmul_classify(a, b))

    Example output:
    > PN: 13.0 = -5.6, PZ: 7.0 = 0.0, ZN: 1.0 = 0.0
    Meaning:
    - 13 instances of positive x negative, summing to -5.6
    - 7 instances of positive x near-zero, summing to 0.0
    - 1 instance of near-zero x negative, summing to 0.0

    Args:
        counts: the counts returned by matmul_classify() or one of its variants.
            Alternatively, the first argument may be a tuple containing both counts and sums.
        sums: the sums returned by matmul_classify() or one of its variants.
    Returns:
        string description
    """
    if isinstance(counts, tuple):
        counts, sums = counts
    terms = classify_terms()
    counts_by_class = np.sum(counts, axis=tuple(range(counts.ndim - 1)))
    sums_by_class = np.sum(sums, axis=tuple(range(counts.ndim - 1)))

    # sort by largest counts
    sort_order = np.argsort(counts_by_class)[::-1]
    terms = np.array(terms)[sort_order]
    counts_by_class = counts_by_class[sort_order]
    sums_by_class = sums_by_class[sort_order]

    # drop anything with zero counts
    mask = counts_by_class > 0
    terms = terms[mask]
    counts_by_class = counts_by_class[mask]
    sums_by_class = sums_by_class[mask]

    # turn into a summary text
    summary = ''
    for this_term, this_count, this_sum in zip(terms, counts_by_class, sums_by_class):
        if len(summary) > 0:
            summary += ', '
        summary += f"{this_term}: {this_count} = {this_sum}"
    if len(summary) == 0:
        # pretty weird, but maybe this will happen
        summary = '<empty>'

    return summary


def classify_terms(example=None):
    """
    Identifies the appropriate terms list based on the example given, or otherwise
    assumes the terms for a full mat-mul like operation.

    Args:
        example: a count or sum result from a call to matmul_classify() or similar,
            or the tensor containing both the count and sum.
    Returns:
        list of strings, containing the names of the terms in the default order
    """
    if example is None:
        tensor_count = 2   # default
    else:
        if isinstance(example, tuple):
            example, _ = example
        shape = tf.shape(example)
        channels = shape[-1]
        if channels == 3:
            tensor_count = 1
        elif channels == 9:
            tensor_count = 2
        else:
            raise ValueError("Unrecognised example with {channels} in last dim: {shape}")

    if tensor_count == 1:
        return ['P', 'Z', 'N']
    else:
        return ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']


def tensor_classify(x, confidence: float = 0.95, threshold: float = None):
    """
    Calculates the usual counts and sums of positive, near-zero, and negative values in a single tensor.

    This is an extension from the matmul-like operations to a single tensor.
    It seems somewhat strange on its own, but it proves useful so that you can
    use the other functions to get a nice summary()

    Args:
      x: np-array or tensor with shape (n, k)
      confidence: statistical confidence (0.0 to 1.0) that you wish to meet
        that a value is accurately placed within the P, Z, or N categories.
        Higher values lead to more strict requirements for "near zero".
        1.0 only considers exactly 0.0 as "near zero".
      threshold: abs(x) values less than this are considered near-zero,
        otherwise inferred from confidence

    Returns:
      (counts, sums) where each is an np-array with shape (n, m, 3)
    """
    x = np.array(x)
    x_p, x_z, x_n = classification_mask(x, confidence, threshold)

    # compute counts and sums
    counts = np.zeros_like(x, dtype=int)
    sums = np.zeros_like(x, dtype=x.dtype)
    counts[:, :, 0] = x_p.astype(int)
    counts[:, :, 1] = x_z.astype(int)
    counts[:, :, 2] = x_n.astype(int)
    sums[:, :, 0] = x * x_p
    sums[:, :, 1] = x * x_z
    sums[:, :, 2] = x * x_n

    return counts, sums


# change threshold args to 'thresholds', taking a single value or a list, eg: 0.35; [None, 0.75]; [0.45, 0.75]
def matmul_classify(x1, x2, confidence: float = 0.95, threshold1: float = None, threshold2: float = None):
    """
    Calculates how the dot-product of x1 . x2 comes to have the range of values that it does.

    For each computation of an output cell, both sets of inputs are independently grouped
    into those that are near-zero (Z), positive at or above threshold (P), and negative at or below threshold (N).
    The output cell value is then defined as the sum of the combinations of those groups, with 9 combinations
    in total, in the following fixed order: PP, PZ, PN, ZP, ZZ, ZN, NP, NZ, NN.

    The counts and sums of each component are returned as separate tensors.
    The original matmul result for any given output cell is given by simply
    summing the sums of each category:
      sum(PP) + sum(PZ) + sum(PN) + ... sum(NN)
    For the entire result this can be computed via:
      output = np.sum(sums, axis=-1)

    "Near zero" is defined as any value having an absolute value less than a threshold.
    By default, the threshold is determined by taking the (1 - zero_confidence) percentile level across each
    of the separate inputs. Alternatively, an absolute magnitude threshold may be specified for
    any of the inputs.

    Example usage:
    >>> a = np.tile(np.arange(0.0, 1.0, 0.1), (10,1))
    >>> counts, sums = matmul_classify(a, a, confidence=0.75)
    >>> for i, name in enumerate(classify_terms()):
    >>>   if np.sum(counts[:,:,i]) > 0:
    >>>     print(f"Counts({name}): {counts[:,:,i]}")
    >>> for i, name in enumerate(classify_terms()):
    >>>   if np.sum(counts[:,:,i]) > 0:
    >>>     print(f"Sums({name}): {sums[:,:,i]}")
    >>> print(f"True matmul: {np.matmul(a, a)}")
    >>> print(f"Derived matmul: {np.sum(sums, axis=-1)}")

    Args:
      x1: np-array or tensor with shape (n, k)
      x2: np-array or tensor with shape (k, m)
      confidence: statistical confidence (0.0 to 1.0) that you wish to meet
        that a value is accurately placed within the P, Z, or N categories.
        Higher values lead to more strict requirements for "near zero".
        1.0 only considers exactly 0.0 as "near zero".
      threshold1: abs(X1) values less than this are considered near-zero,
        otherwise inferred from confidence
      threshold2: abs(X2) values less than this are considered near-zero,
        otherwise inferred from confidence

    Returns:
      (counts, sums) where each is an np-array with shape (n, m, 9)
    """

    # standardise on data format
    x1 = np.array(x1)
    x2 = np.array(x2)

    # apply thresholds and create masks
    x1_p, x1_z, x1_n = classification_mask(x1, confidence, threshold1)
    x2_p, x2_z, x2_n = classification_mask(x2, confidence, threshold2)

    # compute counts
    counts = np.zeros((x1.shape[0], x2.shape[1], 9), dtype=int)
    sums = np.zeros((x1.shape[0], x2.shape[1], 9), dtype=np.result_type(x1.dtype, x2.dtype))

    counts[:, :, 0] = np.matmul(x1_p.astype(int), x2_p.astype(int))
    counts[:, :, 1] = np.matmul(x1_p.astype(int), x2_z.astype(int))
    counts[:, :, 2] = np.matmul(x1_p.astype(int), x2_n.astype(int))
    counts[:, :, 3] = np.matmul(x1_z.astype(int), x2_p.astype(int))
    counts[:, :, 4] = np.matmul(x1_z.astype(int), x2_z.astype(int))
    counts[:, :, 5] = np.matmul(x1_z.astype(int), x2_n.astype(int))
    counts[:, :, 6] = np.matmul(x1_n.astype(int), x2_p.astype(int))
    counts[:, :, 7] = np.matmul(x1_n.astype(int), x2_z.astype(int))
    counts[:, :, 8] = np.matmul(x1_n.astype(int), x2_n.astype(int))

    sums[:, :, 0] = np.matmul(x1 * x1_p, x2 * x2_p)
    sums[:, :, 1] = np.matmul(x1 * x1_p, x2 * x2_z)
    sums[:, :, 2] = np.matmul(x1 * x1_p, x2 * x2_n)
    sums[:, :, 3] = np.matmul(x1 * x1_z, x2 * x2_p)
    sums[:, :, 4] = np.matmul(x1 * x1_z, x2 * x2_z)
    sums[:, :, 5] = np.matmul(x1 * x1_z, x2 * x2_n)
    sums[:, :, 6] = np.matmul(x1 * x1_n, x2 * x2_p)
    sums[:, :, 7] = np.matmul(x1 * x1_n, x2 * x2_z)
    sums[:, :, 8] = np.matmul(x1 * x1_n, x2 * x2_n)

    return counts, sums


# change threshold args to 'thresholds', taking a single value or a list, eg: 0.35; [None, 0.75]; [0.45, 0.75]
def multiply_classify(x, y, confidence: float = 0.95, x_threshold: float = None, y_threshold: float = None):
    """
    Like matmul_classify but for elementwise multiplication.

    Args:
        x: np-array or tensor
        y: np-array or tensor, must have the same type and shape as x

        confidence: statistical confidence (0.0 to 1.0) that you wish to meet
          that a value is accurately placed within the P, Z, or N categories.
          Higher values lead to more strict requirements for "near zero".
          1.0 only considers exactly 0.0 as "near zero".
        x_threshold: abs(X1) values less than this are considered near-zero,
          otherwise inferred from confidence
        y_threshold: abs(X2) values less than this are considered near-zero,
          otherwise inferred from confidence

    Returns:
        (counts, sums) containing the counts and sums of each component, respectively.
        Each a tensor with shape `x_shape + (9,)`.
    """
    # standardise on data format
    x = tf.constant(x)
    y = tf.constant(y)

    # apply thresholds and create masks
    x_p, x_z, x_n = classification_mask(x, confidence, x_threshold)
    y_p, y_z, y_n = classification_mask(y, confidence, y_threshold)

    # compute counts and sums for each classification
    counts = []
    x_pc = tf.cast(x_p, tf.float32)
    x_zc = tf.cast(x_z, tf.float32)
    x_nc = tf.cast(x_n, tf.float32)
    y_pc = tf.cast(y_p, tf.float32)
    y_zc = tf.cast(y_z, tf.float32)
    y_nc = tf.cast(y_n, tf.float32)
    counts.append(tf.math.multiply(x_pc, y_pc))
    counts.append(tf.math.multiply(x_pc, y_zc))
    counts.append(tf.math.multiply(x_pc, y_nc))
    counts.append(tf.math.multiply(x_zc, y_pc))
    counts.append(tf.math.multiply(x_zc, y_zc))
    counts.append(tf.math.multiply(x_zc, y_nc))
    counts.append(tf.math.multiply(x_nc, y_pc))
    counts.append(tf.math.multiply(x_nc, y_zc))
    counts.append(tf.math.multiply(x_nc, y_nc))

    sums = []
    x_pv = tf.where(x_p, x, tf.zeros_like(x))
    x_zv = tf.where(x_z, x, tf.zeros_like(x))
    x_nv = tf.where(x_n, x, tf.zeros_like(x))
    y_pv = tf.where(y_p, y, tf.zeros_like(y))
    y_zv = tf.where(y_z, y, tf.zeros_like(y))
    y_nv = tf.where(y_n, y, tf.zeros_like(y))
    sums.append(tf.math.multiply(x_pv, y_pv))
    sums.append(tf.math.multiply(x_pv, y_zv))
    sums.append(tf.math.multiply(x_pv, y_nv))
    sums.append(tf.math.multiply(x_zv, y_pv))
    sums.append(tf.math.multiply(x_zv, y_zv))
    sums.append(tf.math.multiply(x_zv, y_nv))
    sums.append(tf.math.multiply(x_nv, y_pv))
    sums.append(tf.math.multiply(x_nv, y_zv))
    sums.append(tf.math.multiply(x_nv, y_nv))

    # format into final output
    return tf.stack(counts, axis=-1), tf.stack(sums, axis=-1)


# change threshold args to 'thresholds', taking a single value or a list, eg: 0.35; [None, 0.75]; [0.45, 0.75]
def conv_classify(inputs, kernel, strides=1, padding="VALID", confidence: float = 0.95,
                  inputs_threshold: float = None, kernel_threshold: float = None):
    """
    Like matmul_classify but for convolutions.
    Supports 1D, 2D and 3D convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)`
        kernel: Tensor of rank N+2. `kernel` has shape
            `(kernel_spatial_shape, num_input_channels, num_output_channels)`.
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        confidence: statistical confidence (0.0 to 1.0) that you wish to meet
          that a value is accurately placed within the P, Z, or N categories.
          Higher values lead to more strict requirements for "near zero".
          1.0 only considers exactly 0.0 as "near zero".
        inputs_threshold: abs(X1) values less than this are considered near-zero,
          otherwise inferred from confidence
        kernel_threshold: abs(X2) values less than this are considered near-zero,
          otherwise inferred from confidence

    Returns:
        (counts, sums) containing the counts and sums of each component, respectively.
        Each a tensor with shape `(batch_size,) + inputs_spatial_shape + (num_channels,9)`.
    """
    # standardise on data format
    inputs = tf.constant(inputs)
    kernel = tf.constant(kernel)

    # apply thresholds and create masks
    inputs_p, inputs_z, inputs_n = classification_mask(inputs, confidence, inputs_threshold)
    kernel_p, kernel_z, kernel_n = classification_mask(kernel, confidence, kernel_threshold)

    # compute counts and sums for each classification
    counts = []
    inputs_pc = tf.cast(inputs_p, tf.float32)
    inputs_zc = tf.cast(inputs_z, tf.float32)
    inputs_nc = tf.cast(inputs_n, tf.float32)
    kernel_pc = tf.cast(kernel_p, tf.float32)
    kernel_zc = tf.cast(kernel_z, tf.float32)
    kernel_nc = tf.cast(kernel_n, tf.float32)
    counts.append(tf.nn.convolution(input=inputs_pc, filters=kernel_pc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_pc, filters=kernel_zc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_pc, filters=kernel_nc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_zc, filters=kernel_pc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_zc, filters=kernel_zc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_zc, filters=kernel_nc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_nc, filters=kernel_pc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_nc, filters=kernel_zc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_nc, filters=kernel_nc, strides=strides, padding=padding))

    sums = []
    inputs_pv = tf.where(inputs_p, inputs, tf.zeros_like(inputs))
    inputs_zv = tf.where(inputs_z, inputs, tf.zeros_like(inputs))
    inputs_nv = tf.where(inputs_n, inputs, tf.zeros_like(inputs))
    kernel_pv = tf.where(kernel_p, kernel, tf.zeros_like(kernel))
    kernel_zv = tf.where(kernel_z, kernel, tf.zeros_like(kernel))
    kernel_nv = tf.where(kernel_n, kernel, tf.zeros_like(kernel))
    sums.append(tf.nn.convolution(input=inputs_pv, filters=kernel_pv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_pv, filters=kernel_zv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_pv, filters=kernel_nv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_zv, filters=kernel_pv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_zv, filters=kernel_zv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_zv, filters=kernel_nv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_nv, filters=kernel_pv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_nv, filters=kernel_zv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_nv, filters=kernel_nv, strides=strides, padding=padding))

    # format into final output
    return tf.stack(counts, axis=-1), tf.stack(sums, axis=-1)


def classification_mask(x, confidence: float = 0.95, threshold: float = None):
    """
    Classifies the values of x as positive (P), near-zero (Z), or negative (N) according to a threshold.
    Args:
        x: np-array or tensor
        confidence: statistical confidence (0.0 to 1.0) that you wish to meet
            that a value is accurately placed within the P, Z, or N categories.
            Higher values lead to more strict requirements for "near zero".
            1.0 only considers exactly 0.0 as "near zero".
        threshold: abs(x) values less than this are considered near-zero,
            otherwise inferred by using confidence to draw an appropriate percentile from the
            values of x.
    Returns:
        (pos_mask, zero_mask, neg_mask) - np-array bool tensors

    """
    # determine threshold
    if threshold is None:
        threshold = tfp.stats.percentile(tf.abs(x), 100 * (1 - confidence), interpolation='midpoint')

    # apply threshold
    # Note: on small matrices with few discrete numbers, percentile() will find a value on either side
    #  of the percentage threshold, thus we should apply the threshold rule:
    #     - zero if value < threshold
    # However, the threshold may be zero, which requires the extra rule:
    #     - zero if zero
    zero_mask = tf.logical_or(x == 0, x < threshold)
    pos_mask = tf.logical_and(x > 0, tf.logical_not(zero_mask))
    neg_mask = tf.logical_and(x < 0, tf.logical_not(zero_mask))

    return pos_mask.numpy(), zero_mask.numpy(), neg_mask.numpy()
