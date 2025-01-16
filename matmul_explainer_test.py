from matmul_explainer import *
import numpy as np

def run_test_suite():
    matmul_classify_test()
    conv_classify_2d_test()


def matmul_classify_test():
    a = np.arange(0.0, 1.0, 0.1)
    a = np.tile(a, (10, 1))

    counts, sums = matmul_classify(a, a, confidence=0.75)

    expected_counts = [490, 210, 0, 210, 90, 0, 0, 0, 0]
    actual_counts = np.sum(counts, axis=(0, 1))
    assert np.all(actual_counts == expected_counts),\
      f"Expected counts {expected_counts}, got: {actual_counts}"

    real_matmul = np.matmul(a, a)
    derived_matmul = np.sum(sums, axis=-1)
    assert np.allclose(real_matmul, derived_matmul), "real matmul and derived matmul are different"


def conv_classify_2d_test():
    a = np.arange(0.0, 0.9, 0.1)
    a = np.tile(a, (9, 1)).astype(np.float32)
    a = tf.reshape(a, shape=(1, 9, 9, 1))

    k = np.array([
        [-1, 0, -1],
        [+1, +1, +1],
        [-1, 0, -1]
    ]).astype(np.float32)
    k = tf.reshape(k, shape=(3, 3, 1, 1))

    counts, sums = conv_classify(a, k, confidence=0.95)

    expected_counts = [140, 0, 182, 0, 0, 0, 0, 0, 0]
    actual_counts = np.sum(counts, axis=(0, 1, 2, 3))
    assert np.all(actual_counts == expected_counts), f"Expected counts {expected_counts}, got: {actual_counts}"

    expected_sums = [58.79999, 0., -78.40001, 0., 0., 0., 0., 0., 0.]
    actual_sums = np.sum(sums, axis=(0, 1, 2, 3))
    assert np.allclose(actual_sums == expected_sums), f"Expected sums {expected_sums}, got: {actual_sums}"

    real_matmul = np.matmul(a, a)
    derived_matmul = np.sum(sums, axis=-1)
    expected_conv = tf.nn.convolution(a, k)
    derived_conv = tf.reduce_sum(sums, axis=-1)
    assert np.allclose(expected_conv, derived_conv), "real conv and derived conv are different"

