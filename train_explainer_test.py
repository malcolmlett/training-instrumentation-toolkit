from train_explainer import *
import matmul_explainer as me
import numpy as np


def run_test_suite():
    describe_top_near_zero_explanandum_test()
    print("All train_explainer tests passed.")


def describe_top_near_zero_explanandum_test():
    a = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 1.]
    ])
    b = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
    description, fraction = describe_top_near_zero_explanandum(me.matmul_classify(a, b))
    assert description == "near-zero values from first input", f"Got {description=}"
    assert np.allclose(fraction, 0.89, 0.01), f"Got {fraction=}"

    a = np.array([
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]
    ])
    b = np.array([
        [1., 0., 1.],
        [-1., 1., -1.],
        [0., -1., 0.]
    ])
    description, fraction = describe_top_near_zero_explanandum(me.matmul_classify(a, b))
    assert description == "positive/negatives from second input cancelling out (PP ~= PN)", f"Got {description=}"
    assert np.allclose(fraction, 0.67, 0.01), f"Got {fraction=}"

    a = np.array([
        [3., -2., -2., 2.],
        [3., -2., -2., 2.],
        [3., -2., -2., 2.]
    ])
    b = np.array([
        [2., 2., 2.],
        [-3., -3., -3.],
        [4., 4., 4.],
        [-2., -2., -1.]
    ])
    description, fraction = describe_top_near_zero_explanandum(me.matmul_classify(a, b))
    assert description == "sums of positive/negatives from both inputs cancelling out (PP+NN ~= NP+PN)", f"Got {description=}"
    assert np.allclose(fraction, 0.67, 0.01), f"Got {fraction=}"

