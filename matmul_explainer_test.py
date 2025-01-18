from matmul_explainer import *
import tensorflow as tf
import numpy as np


def run_test_suite():
    classification_mask_test()
    tensor_classify_test()
    multiply_classify_test()
    matmul_classify_test()
    conv_classify_1d_test()
    conv_classify_2d_test()
    print("All matmul_explainer tests passed.")


def classification_mask_test():
    a = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p, z, n, t = classification_mask(a, confidence=0.75)
    assert t == 0.25, f"Expected threshold 0.25, got: {t}"
    expected_p = [False, False, False, True, True, True, True, True, True, True]
    expected_z = [True, True, True, False, False, False, False, False, False, False]
    expected_n = [False, False, False, False, False, False, False, False, False, False]
    assert np.all(p == expected_p), f"Expected p {expected_p}, got: {p}"
    assert np.all(z == expected_z), f"Expected z {expected_z}, got: {z}"
    assert np.all(n == expected_n), f"Expected n {expected_n}, got: {n}"

    a = np.array([-0.5, 0, -0.5])
    p, z, n, t = classification_mask(a, confidence=0.95)
    assert t == 0.25, f"Expected threshold 0.25, got: {t}"
    expected_p = [False, False, False]
    expected_z = [False, True, False]
    expected_n = [True, False, True]
    assert np.all(p == expected_p), f"Expected p {expected_p}, got: {p}"
    assert np.all(z == expected_z), f"Expected z {expected_z}, got: {z}"
    assert np.all(n == expected_n), f"Expected n {expected_n}, got: {n}"

    a = np.array([-0.5, 0.25, 0, 0.25, -0.5])
    p, z, n, t = classification_mask(a, confidence=0.95)
    assert t == 0.125, f"Expected threshold 0.125, got: {t}"
    expected_p = [False, True, False, True, False]
    expected_z = [False, False, True, False, False]
    expected_n = [True, False, False, False, True]
    assert np.all(p == expected_p), f"Expected p {expected_p}, got: {p}"
    assert np.all(z == expected_z), f"Expected z {expected_z}, got: {z}"
    assert np.all(n == expected_n), f"Expected n {expected_n}, got: {n}"


def tensor_classify_test():
    a = np.arange(0.0, 1.0, 0.1)
    a = np.tile(a, (10, 1))

    counts, sums = tensor_classify(a, confidence=0.75)

    expected_counts = [80, 20, 0]
    actual_counts = np.sum(counts, axis=(0, 1))
    assert np.all(actual_counts == expected_counts), f"Expected counts {expected_counts}, got: {actual_counts}"

    expected_sums = [44., 1., 0.]
    actual_sums = np.sum(sums, axis=(0, 1))
    assert np.allclose(actual_sums, expected_sums), f"Expected sums {expected_sums}, got: {actual_sums}"

    derived_value = np.sum(sums, axis=-1)
    assert np.allclose(a, derived_value), "real value and derived value are different"


def multiply_classify_test():
    a = np.arange(0.0, 1.0, 0.1)
    a = np.tile(a, (10, 1))

    counts, sums = multiply_classify(a, a.transpose(), confidence=0.75)

    expected_counts = [64, 16, 0, 16, 4, 0, 0, 0, 0]
    actual_counts = np.sum(counts, axis=(0, 1))
    assert np.all(actual_counts == expected_counts), f"Expected counts {expected_counts}, got: {actual_counts}"

    expected_sums = [19.36, 0.44, 0, 0.44, 0.01, 0., 0., 0., 0.]
    actual_sums = np.sum(sums, axis=(0, 1))
    assert np.allclose(actual_sums, expected_sums), f"Expected sums {expected_sums}, got: {actual_sums}"

    real_matmul = np.multiply(a, a.transpose())
    derived_matmul = np.sum(sums, axis=-1)
    assert np.allclose(real_matmul, derived_matmul), "real multiply and derived multiply are different"


def matmul_classify_test():
    a = np.arange(0.0, 1.0, 0.1)
    a = np.tile(a, (10, 1))

    counts, sums = matmul_classify(a, a, confidence=0.75)

    expected_counts = [640, 160, 0, 160, 40, 0, 0, 0, 0]
    actual_counts = np.sum(counts, axis=(0, 1))
    assert np.all(actual_counts == expected_counts), f"Expected counts {expected_counts}, got: {actual_counts}"

    expected_sums = [193.6, 4.4, 0, 4.4, 0.1, 0., 0., 0., 0.]
    actual_sums = np.sum(sums, axis=(0, 1))
    assert np.allclose(actual_sums, expected_sums), f"Expected sums {expected_sums}, got: {actual_sums}"

    real_matmul = np.matmul(a, a)
    derived_matmul = np.sum(sums, axis=-1)
    assert np.allclose(real_matmul, derived_matmul), "real matmul and derived matmul are different"


def conv_classify_1d_test():
    a = np.arange(0.0, 0.9, 0.1).astype(np.float32)
    a = tf.reshape(a, shape=(1,9,1))

    k = np.array([-1, 0, -1]).astype(np.float32)
    k = tf.reshape(k, shape=(3,1,1))

    counts, sums = conv_classify(a, k, confidence=0.90)

    expected_counts = [0, 7, 13, 0, 0, 1, 0, 0, 0]
    actual_counts = np.sum(counts, axis=(0, 1, 2))
    assert np.all(actual_counts == expected_counts), f"Expected counts {expected_counts}, got: {actual_counts}"

    expected_sums = [0., 0., -5.6, 0., 0., 0., 0., 0., 0.]
    actual_sums = np.sum(sums, axis=(0, 1, 2))
    assert np.allclose(actual_sums, expected_sums), f"Expected sums {expected_sums}, got: {actual_sums}"

    expected_conv = tf.nn.convolution(a, k)
    derived_conv = tf.reduce_sum(sums, axis=-1)
    assert np.allclose(expected_conv, derived_conv), "real conv and derived conv are different"


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

    counts, sums = conv_classify(a, k, confidence=0.75)

    expected_counts = [126, 84, 168, 21, 14, 28, 0, 0, 0]
    actual_counts = np.sum(counts, axis=(0, 1, 2, 3))
    assert np.all(actual_counts == expected_counts), f"Expected counts {expected_counts}, got: {actual_counts}"

    expected_sums = [57.4, 0., -77., 1.4000002, 0., -1.4000001, 0., 0., 0.,]
    actual_sums = np.sum(sums, axis=(0, 1, 2, 3))
    assert np.allclose(actual_sums, expected_sums), f"Expected sums {expected_sums}, got: {actual_sums}"

    expected_conv = tf.nn.convolution(a, k)
    derived_conv = tf.reduce_sum(sums, axis=-1)
    assert np.allclose(expected_conv, derived_conv), "real conv and derived conv are different"

