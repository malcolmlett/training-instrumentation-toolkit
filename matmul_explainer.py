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


def classify_terms():
    return ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']


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

    "Near zero" is defined as any value having an absolute value less than or equal a threshold.
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
      threshold1: abs(X1) values less than or equal this are considered near-zero
      threshold2: abs(X2) values less than or equal this are considered near-zero

    Returns:
      (counts, sums) where each is an np-array with shape (n, m, 9)
    """

    # standardise on data format
    x1 = np.array(x1)
    x2 = np.array(x2)

    # determine thresholds
    # (note: on small matrices with few discrete numbers, np.percentile will find a value <= percentile_level,
    #  so subsequently we need to use this threshold via the rule "zero if value <= threshold", otherwise we might
    #  miss values on the threshold that are still < percentile_level.)
    if threshold1 is None:
        threshold1 = np.percentile(np.abs(x1), 100 * (1 - confidence))
    if threshold2 is None:
        threshold2 = np.percentile(np.abs(x2), 100 * (1 - confidence))
    print(f"X1 threshold: {threshold1}")
    print(f"X2 threshold: {threshold2}")

    # create masks that classify each input individually
    x1_p = x1 > threshold1
    x1_z = np.abs(x1) <= threshold1
    x1_n = x1 < -threshold1

    x2_p = x2 > threshold2
    x2_z = np.abs(x2) <= threshold2
    x2_n = x2 < -threshold2

    # print(f"X1_p: {x1_p}")
    # print(f"X1_z: {x1_z}")
    # print(f"X1_n: {x1_n}")

    # print(f"X2_p: {x2_p}")
    # print(f"X2_z: {x2_z}")
    # print(f"X2_n: {x2_n}")

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
