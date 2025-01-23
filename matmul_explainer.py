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


def summarise(counts, sums=None, terms=None, *, mask=None, show_percentage=False):
    """
    Generates a concise summary text from the result of calling matmul_classify() or any of its variants.

    By default, lists the components ordered by largest count first.

    May be used in either form:
    > summarise(counts, sums)
    > summarise(counts, sums, terms)
    > summarise(xxx_classify(a,b))

    Example output:
    > PN: 13.0 = -5.6, PZ: 7.0 = 0.0, ZN: 1.0 = 0.0
    Meaning:
    - 13 instances of positive x negative, summing to -5.6
    - 7 instances of positive x near-zero, summing to 0.0
    - 1 instance of near-zero x negative, summing to 0.0

    Args:
        counts: the counts returned by matmul_classify() or one of its variants.
          Shape: value_shape + (terms,)
          Alternatively pass a tuple containing the counts and sums, and optionally
          the terms.
        sums: the sums returned by matmul_classify() or one of its variants.
          Shape: value_shape + (terms,)
        terms: terms returned by filter_classifications(), same shape as counts and sums.
          Shape: value_shape + (terms,)
          Must be included if filter_classifications() has been called.
    Keyword args:
        mask: bool with shape: value_shape.
        show_percentage: False.
          Whether to show count values as a percentage of all counts being summarised
          (after masking), or as absolute values.
    Returns:
        string description
    """
    # parse and validate args
    if isinstance(counts, tuple):
        if len(counts) == 2:
            counts, sums = counts
        else:
            counts, sums, terms = counts
    if mask is not None:
        required_shape = np.shape(counts)[:-1]
        if np.shape(mask) != required_shape:
            raise ValueError(f"mask should have shape {required_shape}, but got: {np.shape(mask)}")

    # cleanup order for consistent order by terms
    if terms is not None:
        counts, sums, _ = _standardize_order(counts, sums, terms)
    terms_list = classify_terms(counts)

    # apply mask
    if mask is not None:
        counts = counts * mask[..., np.newaxis]
        sums = sums * mask[..., np.newaxis]

    # calculate summaries across each class
    counts_by_class = np.sum(counts, axis=tuple(range(counts.ndim - 1)))
    sums_by_class = np.sum(sums, axis=tuple(range(counts.ndim - 1)))

    # sort summary values by largest counts
    sort_order = np.argsort(counts_by_class)[::-1]
    terms_list = np.array(terms_list)[sort_order]
    counts_by_class = counts_by_class[sort_order]
    sums_by_class = sums_by_class[sort_order]

    # optional: convert counts to fractions
    if show_percentage:
        factor = np.sum(counts_by_class)
        if factor == 0:
            factor = 1.0  # avoid div-by-zero
        counts_by_class = np.divide(counts_by_class, factor)

    # drop anything with zero counts
    summary_mask = counts_by_class > 0
    terms_list = terms_list[summary_mask]
    counts_by_class = counts_by_class[summary_mask]
    sums_by_class = sums_by_class[summary_mask]

    # format as one-line text description
    summary = ''
    for this_term, this_count, this_sum in zip(terms_list, counts_by_class, sums_by_class):
        if len(summary) > 0:
            summary += ', '
        this_count = f"{this_count*100:.1f}%" if show_percentage else this_count
        summary += f"{this_term}: {this_count} = {this_sum}"
    if len(summary) == 0:
        # pretty weird, but maybe this will happen
        summary = '<empty>'

    return summary


def classify_terms(example=None, retain_shape=False):
    """
    Identifies the appropriate terms list based on the example given, or otherwise
    assumes the terms for a full mat-mul like operation.

    Args:
        example: a count or sum result from a call to matmul_classify() or similar,
            or the tensor containing both the count and sum.
        retain_shape: whether to tile the terms out and to return an array
            in the same shape as the provided example. Only allowed if an example is included.

    Returns:
        list of strings, containing the names of the terms in the default order
        OR
        np-array in same shape as example with identified term for each value
    """
    shape = None
    if example is None:
        tensor_count = 2   # default
    else:
        if isinstance(example, tuple):
            example, _ = example
        shape = np.shape(example)
        channels = shape[-1]
        if channels == 3:
            tensor_count = 1
        elif channels == 9:
            tensor_count = 2
        else:
            raise ValueError("Unrecognised example with {channels} in last dim: {shape}")

    if tensor_count == 1:
        term_list = ['P', 'Z', 'N']
    else:
        term_list = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']

    if retain_shape:
        if shape is None:
            raise ValueError("Cannot retain shape without example")
        value_shape = shape[:-1]
        term_count = len(term_list)
        term_array = np.array(term_list)
        terms = np.reshape(term_array, (1,) * len(value_shape) + (term_count,))
        terms = np.tile(terms, reps=value_shape + (1,))
        return terms
    else:
        return term_list


def tensor_classify(x, confidence: float = 0.95, threshold: float = None,
                    return_threshold=False):
    """
    Calculates the usual counts and sums of positive, near-zero, and negative values in a single tensor.

    This is an extension from the matmul-like operations to a single tensor.
    It seems somewhat strange on its own, but it proves useful so that you can
    use the other functions to get a nice summary().

    Args:
        x: np-array or tensor with any shape
        confidence: statistical confidence (0.0 to 1.0) that you wish to meet
            that a value is accurately placed within the P, Z, or N categories.
            Higher values lead to more strict requirements for "near zero".
            1.0 only considers exactly 0.0 as "near zero".
        threshold: abs(x) values less than this are considered near-zero,
            otherwise inferred from confidence
        return_threshold: whether to additionally return the derived threshold

    Returns:
        (counts, sums) containing the counts and sums of each component, respectively.
        Each a tensor with shape `x_shape + (3,)`.
        OR
        (counts, sums, thresholds) with list of thresholds also returned
    """
    # apply thresholds and create masks
    x = tf.constant(x)
    x_p, x_z, x_n, threshold = classification_mask(x, confidence, threshold)

    # compute counts and sums for each classification
    counts = []
    counts.append(tf.cast(x_p, tf.float32))
    counts.append(tf.cast(x_z, tf.float32))
    counts.append(tf.cast(x_n, tf.float32))

    sums = []
    sums.append(tf.where(x_p, x, tf.zeros_like(x)))
    sums.append(tf.where(x_z, x, tf.zeros_like(x)))
    sums.append(tf.where(x_n, x, tf.zeros_like(x)))

    # format into final output
    counts = tf.stack(counts, axis=-1)
    sums = tf.stack(sums, axis=-1)
    if return_threshold:
        return counts, sums, threshold
    else:
        return counts, sums


# change threshold args to 'thresholds', taking a single value or a list, eg: 0.35; [None, 0.75]; [0.45, 0.75]
def matmul_classify(x1, x2, confidence: float = 0.95, threshold1: float = None, threshold2: float = None,
                    return_thresholds=False):
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
      return_thresholds: whether to additionally return the derived thresholds

    Returns:
      (counts, sums) where each is an np-array with shape (n, m, 9)
      OR
      (counts, sums, thresholds) with list of thresholds also returned
    """

    # standardise on data format
    x1 = np.array(x1)
    x2 = np.array(x2)

    # apply thresholds and create masks
    x1_p, x1_z, x1_n, threshold1 = classification_mask(x1, confidence, threshold1)
    x2_p, x2_z, x2_n, threshold2 = classification_mask(x2, confidence, threshold2)

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

    if return_thresholds:
        return counts, sums, [threshold1, threshold2]
    else:
        return counts, sums


# change threshold args to 'thresholds', taking a single value or a list, eg: 0.35; [None, 0.75]; [0.45, 0.75]
def multiply_classify(x, y, confidence: float = 0.95, x_threshold: float = None, y_threshold: float = None,
                      return_thresholds=False):
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
        return_thresholds: whether to additionally return the derived thresholds

    Returns:
        (counts, sums) containing the counts and sums of each component, respectively.
        Each a tensor with shape `x_shape + (9,)`.
        OR
        (counts, sums, thresholds) with list of thresholds also returned
    """
    # standardise on data format
    x = tf.constant(x)
    y = tf.constant(y)

    # apply thresholds and create masks
    x_p, x_z, x_n, x_threshold = classification_mask(x, confidence, x_threshold)
    y_p, y_z, y_n, y_threshold = classification_mask(y, confidence, y_threshold)

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
    counts = tf.stack(counts, axis=-1)
    sums = tf.stack(sums, axis=-1)
    if return_thresholds:
        return counts, sums, [x_threshold, y_threshold]
    else:
        return counts, sums


# change threshold args to 'thresholds', taking a single value or a list, eg: 0.35; [None, 0.75]; [0.45, 0.75]
def conv_classify(inputs, kernel, strides=1, padding="VALID", confidence: float = 0.95,
                  inputs_threshold: float = None, kernel_threshold: float = None,
                  return_thresholds=False):
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
        return_thresholds: whether to additionally return the derived thresholds

    Returns:
        (counts, sums) containing the counts and sums of each component, respectively.
        Each a tensor with shape `(batch_size,) + inputs_spatial_shape + (num_channels,9)`.
        OR
        (counts, sums, thresholds) with list of thresholds also returned
    """
    # standardise on data format
    inputs = tf.constant(inputs)
    kernel = tf.constant(kernel)

    # apply thresholds and create masks
    inputs_p, inputs_z, inputs_n, inputs_threshold = classification_mask(inputs, confidence, inputs_threshold)
    kernel_p, kernel_z, kernel_n, kernel_threshold = classification_mask(kernel, confidence, kernel_threshold)

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
    counts = tf.stack(counts, axis=-1)
    sums = tf.stack(sums, axis=-1)
    if return_thresholds:
        return counts, sums, [inputs_threshold, kernel_threshold]
    else:
        return counts, sums


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
        (pos_mask, zero_mask, neg_mask, threshold) - np-array bool tensors, plus the derived threshold

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
    zero_mask = tf.logical_or(x == 0, tf.abs(x) < threshold)
    pos_mask = tf.logical_and(x > 0, tf.logical_not(zero_mask))
    neg_mask = tf.logical_and(x < 0, tf.logical_not(zero_mask))

    return pos_mask.numpy(), zero_mask.numpy(), neg_mask.numpy(), threshold.numpy()


def filter_classifications(counts, sums, completeness=0.75):
    """
    Filters classification data in order to retain a certain
    "completeness of explanatory coverage". This eliminates less important noise,
    making it easier to understand any summaries produced from it.
    It is also useful for grouping similar classification results that have
    the same major structure, while ignoring less important noise.

    The final returned counts, sums, and terms are sorted for descending
    counts.

    Everything is determined independently for each position
    within the original value tensor, including for final sort. Thus the result
    also includes a terms tensor, with the same shape as counts and sums,
    in order to identify the final terms order for each position.

    Args:
        counts: the counts returned by matmul_classify() or one of its variants.
        sums: the sums returned by matmul_classify() or one of its variants.
        completeness: float in range 0.0 to 1.0.
            Minimum required "completeness of explanatory coverage".
            After filtering, the retained counts and sums will explain at least this much
            of the final result, measured as a combined fraction of the total number of counts
            and the maximum positive or negative extent of the sums.
    Returns:
        (counts, sums, terms) - sorted and filtered (less important counts and sums zerod-out)
    """
    # steps:
    # - initialise a full tuple of (counts, sums, terms, masks) that will always be sorted
    #   together simultaneously
    # - do a sort and filter by sums first
    #   - while the final output is primarily by count, sometimes there are a small number
    #     of unusually large values and we want to see those too
    # - then do a sort and filter by counts
    # - combine final result masks
    # - set all rejected values to zeros
    # - final result is sorted according to counts, largest first
    terms = classify_terms(counts, retain_shape=True)
    masks = np.zeros_like(counts, dtype=bool)
    counts, sums, terms, masks = _partial_filter_by_sum(counts, sums, terms, masks, completeness)
    counts, sums, terms, masks = _partial_filter_by_count(counts, sums, terms, masks, completeness)

    # apply masks - zero-out discarded counts and sums
    # (Must retain terms. I had initially replaced masked terms with '--',
    #  but that causes problems when later needing to standardize the order for
    #  summarisation.)
    counts = counts * masks
    sums = sums * masks

    # apply final sorting - by descending count after masking
    # - although _partial_filter_by_count() will have returned things in the right sort order,
    #   now that we've applied the mask, some of the counts have changed to zero and the order
    #   isn't now quite accurate.
    # - for example, if there's a low-count high-sum, it'll be at the end somewhere
    #   and now needs to come further forward.
    sort_order = np.argsort(-counts, axis=-1)  # negate for descending order
    counts = np.take_along_axis(counts, sort_order, axis=-1)
    sums = np.take_along_axis(sums, sort_order, axis=-1)
    terms = np.take_along_axis(terms, sort_order, axis=-1)
    return counts, sums, terms


def _partial_filter_by_sum(counts, sums, terms, masks, completeness):
    """
    Internal method for use by filter_classifications().

    Applies a sort and filter over the provided counts, sums, terms and masks.
    All are sorted according to descending sum order.
    Masks are updated to indicate values that must be retained.
    Args:
        counts: counts tensor from a classification function, with shape (value_shape + (terms,))
        sums: sums tensor from a classification function, with shape (value_shape + (terms,))
        terms: terms tensor corresponding to the counts and sums, with shape (value_shape + (terms,))
        masks: boolean mask indicating values to retain, with shape (value_shape + (terms,))
            Input value will either be filled with False, or indicate values that must be retained
            as determined by prior filter logic.
        completeness: float in range 0.0 to 1.0.
            Determines minimum fraction of total range that must be retained.
    Returns:
        (counts, sums, terms, masks) with all values sorted along terms dimension, and masks updated
            with additional entries that must be retained
    """
    # determine thresholds against sums
    pos_range = np.sum((sums * (sums > 0)), axis=-1)
    neg_range = np.abs(np.sum((sums * (sums < 0)), axis=-1))
    abs_range = np.maximum(pos_range, neg_range)
    threshold = abs_range * (1 - completeness)  # shape: value_shape

    # sort everything according to sum magnitudes, smallest first for cumsum
    sort_order = np.argsort(np.abs(sums), axis=-1)
    counts = np.take_along_axis(counts, sort_order, axis=-1)
    sums = np.take_along_axis(sums, sort_order, axis=-1)
    terms = np.take_along_axis(terms, sort_order, axis=-1)
    masks = np.take_along_axis(masks, sort_order, axis=-1)  # shape: value_shape + (terms,)

    # update masks
    # - starting from front, sums up positive and negative "marginal" sums separately
    # - the threshold point is the point where the max of the absolute margins >= threshold
    pos_margin_sums = np.cumsum(sums * (sums > 0), axis=-1)
    neg_margin_sums = np.cumsum(sums * (sums < 0), axis=-1)
    mag_margin_sums = np.maximum(pos_margin_sums, np.abs(neg_margin_sums))  # shape: value_shape + (terms,)
    threshold_mask = mag_margin_sums >= threshold[..., np.newaxis]  # shape: value_shape + (terms,)
    masks = np.logical_or(threshold_mask, masks)

    # flip final results for largest-to-smallest order
    counts = np.flip(counts, axis=-1)
    sums = np.flip(sums, axis=-1)
    terms = np.flip(terms, axis=-1)
    masks = np.flip(masks, axis=-1)
    return counts, sums, terms, masks


def _partial_filter_by_count(counts, sums, terms, masks, completeness):
    """
    Internal method for use by filter_classifications().

    Applies a sort and filter over the provided counts, sums, terms and masks.
    All are sorted according to descending count order.
    Masks are updated to indicate values that must be retained.
    Args:
        counts: counts tensor from a classification function, with shape (value_shape + (terms,))
        sums: sums tensor from a classification function, with shape (value_shape + (terms,))
        terms: terms tensor corresponding to the counts and sums, with shape (value_shape + (terms,))
        masks: boolean mask indicating values to retain, with shape (value_shape + (terms,))
            Input value will either be filled with False, or indicate values that must be retained
            as determined by prior filter logic.
        completeness: float in range 0.0 to 1.0.
            Determines minimum fraction of total range that must be retained.
    Returns:
        (counts, sums, terms, masks) with all values sorted along terms dimension, and masks updated
            with additional entries that must be retained
    """
    # determine thresholds against counts
    threshold = np.sum(counts, axis=-1) * (1 - completeness)

    # sort everything according to counts, smallest first for cumsum
    sort_order = np.argsort(counts, axis=-1)
    counts = np.take_along_axis(counts, sort_order, axis=-1)
    sums = np.take_along_axis(sums, sort_order, axis=-1)
    terms = np.take_along_axis(terms, sort_order, axis=-1)
    masks = np.take_along_axis(masks, sort_order, axis=-1)  # shape: value_shape + (terms,)

    # update masks
    # - starting from front, sums up "marginal" counts
    # - the threshold point is the point where the margins >= threshold
    margin_counts = np.cumsum(counts, axis=-1)  # shape: value_shape + (terms,)
    threshold_mask = margin_counts >= threshold[..., np.newaxis]  # shape: value_shape + (terms,)
    masks = np.logical_or(threshold_mask, masks)

    # flip final results for largest-to-smallest order
    counts = np.flip(counts, axis=-1)
    sums = np.flip(sums, axis=-1)
    terms = np.flip(terms, axis=-1)
    masks = np.flip(masks, axis=-1)
    return counts, sums, terms, masks


def _fixargsort(a, reference, axis=-1):
    """
    Like np.argsort() but that it returns the indices needed to "fix" the sort order of the given
    list or array so that it has the same order as reference.
    Assumes that both lists or arrays are of the same shape and have the same values, just in different orders.
    Args:
      a: a list or array needing to have its order "fixed"
      reference: the 1D list or array with the reference order
      axis: int or None, optional. Axis along which to sort. The default is -1 (the last axis). If None, the flattened array is used.
    Returns:
      indices for sorting 'a'
    """
    # For very simple lists, this function would look like the following:
    #  a, reference = np.array(a), np.array(reference)
    #  ref_meta_order = np.argsort(np.argsort(reference))
    #  a_order = np.argsort(a)
    #  return a_order[ref_meta_order]
    # Everything else you see here is there in order to cope with arrays
    # and with variations in how the reference is supplied.

    # normalize types
    a = np.array(a)
    reference = np.array(reference)

    # tile reference out to match shape of a
    if reference.ndim < a.ndim and axis is not None:
        reshape_shape = [1] * len(a.shape)
        reshape_shape[axis] = len(reference)
        reps = list(a.shape)
        reps[axis] = 1
        reference = np.reshape(reference, reshape_shape)
        reference = np.tile(reference, reps)

    # get meta-order from reference
    ref_meta_order = np.argsort(np.argsort(reference, axis=axis), axis=axis)

    # determine ordering of a
    a_order = np.argsort(a, axis=axis)
    indices = np.take_along_axis(a_order, ref_meta_order, axis=axis)
    return indices


def _standardize_order(counts, sums, terms):
    """
    Reverses the ordering effects of filter_classifications().
    Args:
        counts, sums, terms - all must have same shape: value_shape + (terms,)
    Returns:
        counts, sums, terms - reordered along terms axis so that all entries all in the same
            order as returned by classify_terms().
    """
    # (TODO should do some sanity checks)
    terms_list = classify_terms(counts)
    sort_order = _fixargsort(terms, terms_list, axis=-1)
    counts = np.take_along_axis(counts, sort_order, axis=-1)
    sums = np.take_along_axis(sums, sort_order, axis=-1)
    terms = np.take_along_axis(terms, sort_order, axis=-1)

    return counts, sums, terms


def _safe_divide(x, y):
    """
    Element-wise divide x by y, or zero if y is zero.
    Intended as a div-by-zero-safe version for computing means from a sums and counts.
    """
    return np.divide(x, y, out=np.zeros_like(x, dtype=float), where=(y != 0))


def _format_decimal(value, significant_digits=4, scale=None, return_scale=False):
    """
    Variant of the standard number formatting that is optimised first for easier visual comparison
    across multiple numbers potentially ranging wildly across different scales,
    and for compactness second.
    This is achieved by targeting the number of displayed significant digits, regardless
    of scale, and by avoiding scientific notation except for the largest values.

    Can be used to construct a shared scale across multiple numbers, eg:
    > max_value = np.max(values)
    > _, scale = _format_decimal(max_value, return_scale=True)
    > formatted = [_format_decimal(value, scale=scale) for value in values]

    Args:
      value: the value to format
      significant_digits: number of non-zero digits wanted for display
      scale: use this scale instead of calculating from the given value.
    """
    if scale is None:
        scale = int(np.floor(np.log10(abs(value))))

    if scale < 0:
        p = significant_digits - scale + 1  # more digits as the number gets smaller
        res = f"{value:.{p}f}"
    else:
        p = max(0, significant_digits - scale - 1)  # less digits as the number gets larger
        res = f"{value:.{p}f}"

    # todo: maybe switch to scientific notation if length of standard display is some multiple of the length
    # of scientific notation

    if return_scale:
        return res, scale
    else:
        return res
