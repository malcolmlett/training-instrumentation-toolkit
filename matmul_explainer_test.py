from matmul_explainer import *
import numpy as np

def run_test_suite():
    matmul_classify_test()


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

