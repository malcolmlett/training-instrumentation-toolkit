from conv_tools import *


def run_test_suite():
    dilate_test()
    print("All conv_tools tests passed.")


def dilate_test():
    x = np.ones((32, 3, 6))
    expected = np.array([1, 0, 1, 0, 1])
    expected = np.tile(expected[np.newaxis, :, np.newaxis], reps=(32, 1, 6))
    actual = dilate(x, [2])
    assert np.allclose(actual, expected), f"Got: {actual}"

    x = np.ones((32, 3, 3, 6))
    expected = np.array([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1]
    ])
    expected = np.tile(expected[np.newaxis, :, :, np.newaxis], reps=(32, 1, 1, 6))
    actual = dilate(x, [2, 2])
    assert np.allclose(actual, expected), f"Got: {actual}"

    x = np.ones((32, 3, 3, 3, 6))
    expected_on = np.array([
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1]
    ])
    expected_off = np.zeros_like(expected_on)
    expected = np.stack([expected_on, expected_off, expected_on, expected_off, expected_on], axis=-1)
    expected = np.tile(expected[np.newaxis, :, :, :, np.newaxis], reps=(32, 1, 1, 1, 6))
    actual = dilate(x, [2, 2, 2])
    assert np.allclose(actual, expected), f"Got: {actual}"
