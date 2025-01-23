from matmul_explainer import *
from matmul_explainer import _partial_filter_by_sum, _partial_filter_by_count, _fixargsort, _standardize_order
from matmul_explainer import _safe_divide
import tensorflow as tf
import numpy as np


def run_test_suite():
    classify_terms_test()
    classification_mask_test()
    tensor_classify_test()
    multiply_classify_test()
    matmul_classify_test()
    conv_classify_1d_test()
    conv_classify_2d_test()

    _partial_filter_by_sum_test()
    _partial_filter_by_count_test()
    _fixargsort_test()
    filter_classifications_test()
    _standardize_order_test()
    _safe_divide_test()
    group_classifications_test()
    print("All matmul_explainer tests passed.")


def classify_terms_test():
    # gets default list without args
    terms = classify_terms()
    expected_terms = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    assert terms == expected_terms, f"Expected terms {expected_terms}, got: {terms}"

    # gets simple list from tensor classification
    a = np.tile(np.arange(0.0, 1.0, 0.1), (10, 1))
    counts, sums = tensor_classify(a)
    terms = classify_terms(counts)
    expected_terms = ['P', 'Z', 'N']
    assert terms == expected_terms, f"Expected terms {expected_terms}, got: {terms}"

    # gets simple list from matmul classification - counts
    a = np.tile(np.arange(0.0, 1.0, 0.1), (10, 1))
    counts, sums = matmul_classify(a, a)
    terms = classify_terms(counts)
    expected_terms = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    assert terms == expected_terms, f"Expected terms {expected_terms}, got: {terms}"

    # gets simple list from matmul classification - sums
    a = np.tile(np.arange(0.0, 1.0, 0.1), (10, 1))
    counts, sums = matmul_classify(a, a)
    terms = classify_terms(counts)
    expected_terms = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    assert terms == expected_terms, f"Expected terms {expected_terms}, got: {terms}"

    # gets simple list from matmul classification - tuple
    a = np.tile(np.arange(0.0, 1.0, 0.1), (10, 1))
    terms = classify_terms(matmul_classify(a, a))
    expected_terms = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    assert terms == expected_terms, f"Expected terms {expected_terms}, got: {terms}"

    # retains shape from tensor classification
    a = np.tile(np.arange(0.0, 1.0, 0.1), (8, 1))
    counts, sums = tensor_classify(a)
    terms = classify_terms(counts, retain_shape=True)
    expected_terms = np.tile(np.array([[['P', 'Z', 'N']]]), reps=(8, 10, 1))
    assert terms.shape == expected_terms.shape, f"Expected shape {expected_terms.shape}, got: {terms.shape}"
    assert np.all(terms == expected_terms), f"Expected terms {expected_terms}, got: {terms}"

    # retains shape from matmul classification
    a = np.tile(np.arange(0.0, 1.0, 0.1), (8, 1))
    counts, sums = matmul_classify(a, a.T)
    terms = classify_terms(counts, retain_shape=True)
    expected_terms = np.tile(np.array([[['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']]]), reps=(8, 8, 1))
    assert terms.shape == expected_terms.shape, f"Expected shape {expected_terms.shape}, got: {terms.shape}"
    assert np.all(terms == expected_terms), f"Expected terms {expected_terms}, got: {terms}"


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


def _partial_filter_by_sum_test():
    counts = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
    ])
    sums = np.array([
        [0, 1, 3, 6, 10, 15, 21, 28, 36],
        [0, 36, 1, 28, 3, 21, 6, 15, 10],
        [8, 15, 21, 26, 30, 33, 35, 36, 0],
    ])
    expected_counts = np.array([
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [1, 3, 5, 7, 8, 6, 4, 2, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 0],
    ])
    expected_sums = np.array([
        [36, 28, 21, 15, 10, 6, 3, 1, 0],
        [36, 28, 21, 15, 10, 6, 3, 1, 0],
        [36, 35, 33, 30, 26, 21, 15, 8, 0],
    ])
    expected_terms = np.array([
        ['NN', 'NZ', 'NP', 'ZN', 'ZZ', 'ZP', 'PN', 'PZ', 'PP'],
        ['PZ', 'ZP', 'ZN', 'NZ', 'NN', 'NP', 'ZZ', 'PN', 'PP'],
        ['NZ', 'NP', 'ZN', 'ZZ', 'ZP', 'PN', 'PZ', 'PP', 'NN'],
    ])
    expected_masks = np.array([
        [True, True, True, True, False, False, False, False, False],
        [True, True, True, True, False, False, False, False, False],
        [True, True, True, True, True, False, False, False, False],
    ])

    terms = classify_terms(counts, retain_shape=True)
    masks = np.zeros_like(counts, dtype=bool)
    counts, sums, terms, masks = _partial_filter_by_sum(counts, sums, terms, masks, completeness=0.75)
    assert np.all(counts == expected_counts), f"Expected counts {expected_counts}, got: {counts}"
    assert np.all(sums == expected_sums), f"Expected sums {expected_sums}, got: {sums}"
    assert np.all(terms == expected_terms), f"Expected terms {expected_terms}, got: {terms}"
    assert np.all(masks == expected_masks), f"Expected masks {expected_masks}, got: {masks}"


def _partial_filter_by_count_test():
    counts = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
    ])
    sums = np.array([
        [0, 1, 3, 6, 10, 15, 21, 28, 36],
        [0, 36, 1, 28, 3, 21, 6, 15, 10],
        [8, 15, 21, 26, 30, 33, 35, 36, 0],
    ])
    expected_counts = np.array([
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
    ])
    expected_sums = np.array([
        [36, 28, 21, 15, 10, 6, 3, 1, 0],
        [10, 15, 6, 21, 3, 28, 1, 36, 0],
        [8, 15, 21, 26, 30, 33, 35, 36, 0],
    ])
    expected_terms = np.array([
        ['NN', 'NZ', 'NP', 'ZN', 'ZZ', 'ZP', 'PN', 'PZ', 'PP'],
        ['NN', 'NZ', 'NP', 'ZN', 'ZZ', 'ZP', 'PN', 'PZ', 'PP'],
        ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN'],
    ])
    expected_masks = np.array([
        [True, True, True, True, True, False, False, False, False],
        [True, True, True, True, True, False, False, False, False],
        [True, True, True, True, True, False, False, False, False],
    ])

    terms = classify_terms(counts, retain_shape=True)
    masks = np.zeros_like(counts, dtype=bool)
    counts, sums, terms, masks = _partial_filter_by_count(counts, sums, terms, masks, completeness=0.75)
    assert np.all(counts == expected_counts), f"Expected counts {expected_counts}, got: {counts}"
    assert np.all(sums == expected_sums), f"Expected sums {expected_sums}, got: {sums}"
    assert np.all(terms == expected_terms), f"Expected terms {expected_terms}, got: {terms}"
    assert np.all(masks == expected_masks), f"Expected masks {expected_masks}, got: {masks}"


def filter_classifications_test():
    counts = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
    ])
    sums = np.array([
        [0, 1, 3, 6, 10, 15, 21, 28, 36],
        [0, 36, 1, 28, 3, 21, 6, 15, 10],
        [8, 15, 21, 26, 30, 33, 35, 36, 0],
    ])
    expected_counts = np.array([
        [8, 7, 6, 5, 4, 0, 0, 0, 0],
        [8, 7, 6, 5, 4, 3, 1, 0, 0],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
    ])
    expected_sums = np.array([
        [36, 28, 21, 15, 10, 0, 0, 0, 0],
        [10, 15, 6, 21, 3, 28, 36, 0, 0],
        [8, 15, 21, 26, 30, 33, 35, 36, 0],
    ])
    expected_terms = np.array([
        ['NN', 'NZ', 'NP', 'ZN', 'ZZ', 'ZP', 'PN', 'PZ', 'PP'],
        ['NN', 'NZ', 'NP', 'ZN', 'ZZ', 'ZP', 'PZ', 'PN', 'PP'],
        ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN'],
    ])

    counts, sums, terms = filter_classifications(counts, sums, completeness=0.75)
    assert np.all(counts == expected_counts), f"Expected counts {expected_counts}, got: {counts}"
    assert np.all(sums == expected_sums), f"Expected sums {expected_sums}, got: {sums}"
    assert np.all(terms == expected_terms), f"Expected terms {expected_terms}, got: {terms}"


def _fixargsort_test():
    # sort simple list
    a = ['ZZ', 'NN', 'ZP', 'NZ', 'PZ', 'PP', 'NP', 'PN', 'ZN']
    ref = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    order = _fixargsort(a, ref)
    fixed = np.array(a)[order]
    assert np.all(fixed == ref), f"Expected fixed: {ref}, got: {fixed}"

    # sort array given list-type reference
    a = np.array([
        ['ZZ', 'NN', 'ZP', 'NZ', 'PZ', 'PP', 'NP', 'PN', 'ZN'],
        ['ZN', 'PN', 'NP', 'PP', 'PZ', 'NZ', 'ZP', 'NN', 'ZZ']
    ])
    ref = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    order = _fixargsort(a, ref)
    fixed = np.take_along_axis(a, order, axis=-1)
    expected = np.array([
        ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN'],
        ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    ])
    assert np.all(fixed == expected), f"Expected fixed: {ref}, got: {fixed}"

    # sort array given array reference
    a = np.array([
        ['ZZ', 'NN', 'ZP', 'NZ', 'PZ', 'PP', 'NP', 'PN', 'ZN'],
        ['ZN', 'PN', 'NP', 'PP', 'PZ', 'NZ', 'ZP', 'NN', 'ZZ']
    ])
    ref = np.array([
        ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN'],
        ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN'],
    ])
    order = _fixargsort(a, ref)
    fixed = np.take_along_axis(a, order, axis=-1)
    expected = np.array([
        ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN'],
        ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    ])
    assert np.all(fixed == expected), f"Expected fixed: {ref}, got: {fixed}"

    # sort array as flattened list
    a = np.array([
        ['ZZ', 'NN', 'ZP'],
        ['NZ', 'PZ', 'PP'],
        ['NP', 'PN', 'ZN'],
    ])
    ref = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    order = _fixargsort(a, ref, axis=None)
    fixed = np.take_along_axis(a, order, axis=None)
    expected = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    assert np.all(fixed == expected), f"Expected fixed: {ref}, got: {fixed}"

    # sort array as flattened list with array-type reference
    a = np.array([
        ['ZZ', 'NN', 'ZP'],
        ['NZ', 'PZ', 'PP'],
        ['NP', 'PN', 'ZN'],
    ])
    ref = np.array([
        ['PP', 'PZ', 'PN'],
        ['ZP', 'ZZ', 'ZN'],
        ['NP', 'NZ', 'NN'],
    ])
    order = _fixargsort(a, ref, axis=None)
    fixed = np.take_along_axis(a, order, axis=None)
    expected = ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']
    assert np.all(fixed == expected), f"Expected fixed: {ref}, got: {fixed}"


def _standardize_order_test():
    np.random.seed(36)
    a = np.random.uniform(-5, +5, (100, 100)).astype(int).astype(float)
    b = np.random.uniform(-5, +5, (100, 100)).astype(int).astype(float)
    counts0, sums0 = matmul_classify(a,b)
    counts, sums, terms = filter_classifications(counts0, sums0, completeness=1.0)
    counts, sums, terms = _standardize_order(counts, sums, terms)
    assert np.allclose(counts, counts0), "Normalized counts aren't same as original"
    assert np.allclose(sums, sums0), "Normalized sums aren't same as original"


def _safe_divide_test():
    sums = np.array([3.0, 5.0, 6.0, 0.0])
    counts = np.array([2.0, 3.0, 0.0, 0.0])
    expected = np.array([3 / 2, 5 / 3, 0., 0.])
    actual = _safe_divide(sums, counts)
    assert np.all(actual == expected), f"Expected {expected}, got: {actual}"


def group_classifications_test():
    counts = np.array([
        # group 1
        [99, 87, 65, 30, 0, 0, 0, 0, 0],
        [89, 77, 55, 20, 0, 0, 0, 0, 0],

        # group 2: no difference in counts from group 1
        [99, 87, 65, 30, 0, 0, 0, 0, 0],
        [89, 77, 55, 20, 0, 0, 0, 0, 0],

        # group 3
        [77, 65, 80, 0, 0, 0, 0, 0, 0],
    ])
    sums = counts / 10
    terms = np.array([
        # group 1: differences after 4th term should be ignored
        ['PP', 'PZ', 'NN', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ'],
        ['PP', 'PZ', 'NN', 'PN', 'ZZ', 'ZN', 'NP', 'NZ', 'ZP'],

        # group 2: first four terms are different to group 1
        ['ZN', 'PP', 'PZ', 'NN', 'PN', 'ZP', 'ZZ', 'NP', 'NZ'],
        ['ZN', 'PP', 'PZ', 'NN', 'PN', 'ZZ', 'NP', 'NZ', 'ZP'],

        # group 3: same ordered terms as one of the lines above, but term length is different
        ['PP', 'PZ', 'NN', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ'],
    ])

    count_groups, sum_groups, term_groups = group_classifications(counts, sums, terms)

    expected_count_groups = [
        np.array([
            [99, 87, 65, 30, 0, 0, 0, 0, 0],
            [89, 77, 55, 20, 0, 0, 0, 0, 0]
        ]),
        np.array([
            [99, 87, 65, 30, 0, 0, 0, 0, 0],
            [89, 77, 55, 20, 0, 0, 0, 0, 0]
        ]),
        np.array([
            [77, 65, 80, 0, 0, 0, 0, 0, 0]
        ])
    ]
    expected_sum_groups = [counts / 10 for counts in expected_count_groups]
    expected_term_groups = [
        ['PP', 'PZ', 'NN', 'PN'],
        ['ZN', 'PP', 'PZ', 'NN'],
        ['PP', 'PZ', 'NN']
    ]
    expected_group_count = len(expected_count_groups)
    assert len(count_groups) == expected_group_count,\
        f"Expected {expected_group_count} count groups, got {len(count_groups)}"
    assert len(sum_groups) == expected_group_count,\
        f"Expected {expected_group_count} sum groups, got {len(sum_groups)}"
    assert len(term_groups) == expected_group_count,\
        f"Expected {expected_group_count} term groups, got {len(term_groups)}"

    for i, (actual_counts, actual_sums, actual_terms, expected_counts, expected_sums, expected_terms) in\
            enumerate(zip(count_groups, sum_groups, term_groups, expected_count_groups, expected_sum_groups,
                          expected_term_groups)):
        assert np.all(actual_counts == expected_counts),\
            f"Expected group {i} to have counts {expected_counts}, got {actual_counts}"
        assert np.all(actual_sums == expected_sums),\
            f"Expected group {i} to have sums {expected_sums}, got {actual_sums}"
        assert np.all(actual_terms == expected_terms),\
            f"Expected group {i} to have terms {expected_terms}, got {actual_terms}"
        # still need to check shape because the above can miss issues due to broadcasting
        assert actual_counts.shape == expected_counts.shape,\
            f"Expected group {i} to have counts shape {expected_counts.shape}, got {actual_counts.shape}: " \
            f"{actual_counts}"
        assert actual_sums.shape == expected_sums.shape,\
            f"Expected group {i} to have sums shape {expected_sums.shape}, got {actual_sums.shape}: {actual_sums}"
