from matmul_explainer import *


def run_test_suite():
    matmul_classify_test()


def matmul_classify_test():
    a = np.arange(0.0, 1.0, 0.1)
    a = np.tile(a, (10, 1))

    counts, sums = matmul_classify(a, a, confidence=0.75)

    assert np.sum(counts, axis=(0, 1)) == np.array([490, 210, 0, 210, 90, 0, 0, 0, 0])

    print(f"True matmul: {np.matmul(a, a)}")
    print(f"Derived matmul: {np.sum(sums, axis=-1)}")
    np.allclose(np.matmul(a, a), np.sum(sums, axis=-1))

