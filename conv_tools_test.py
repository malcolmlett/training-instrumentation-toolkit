from conv_tools import *
import tensorflow as tf
import tensorflow_probability as tfp


def run_test_suite():
    dilate_test()
    conv_backprop_filter_test()
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


def conv_backprop_filter_test():
    def test_scenario(counters, A_0_shape, W_shape, strides, padding, description):
        tf.random.set_seed(36)
        A_0 = tf.random.normal(A_0_shape)
        W = tf.random.normal(W_shape)

        # forward pass plus ground-truth
        with tf.GradientTape() as tape:
            tape.watch(A_0)
            tape.watch(W)
            if len(A_0.shape) == 3:
                Z = tf.nn.conv1d(A_0, W, stride=strides, padding=padding)
            elif len(A_0.shape) == 4:
                Z = tf.nn.conv2d(A_0, W, strides=strides, padding=padding)
            elif len(A_0.shape) == 5:
                # note: conv3d() strictly requires a list with len(strides) = number of dims
                if not isinstance(strides, int) and len(strides) < 5:
                    strides = [1] + strides + [1]
                Z = tf.nn.conv3d(A_0, W, strides=strides, padding=padding)
            else:
                raise ValueError(f"Unsupported number of dims in A_0: {A_0.shape}")
            # simulate being part of a network by computing mse loss against an 'expected_Z'
            expected_Z = tf.random.normal(Z.shape)
            loss = tf.reduce_mean(tf.square(Z - expected_Z))
        dJdZ, true_dJdW = tape.gradient(loss, [Z, W])

        count = counters[0]
        counters[0] = count + 1

        dJdW = conv_backprop_filter(A_0, dJdZ, W_shape, strides, padding)
        if dJdW.shape != true_dJdW.shape:
            print(f"[{count}] A_0: {A_0.shape}, W: {W.shape}, strides: {strides}, padding: {padding} -> "
                  f"Z: {expected_Z.shape}")
            raise ValueError(f"[{count}] {description}: got wrong shape for dJdW, expected {true_dJdW.shape}, "
                             f"got: {dJdW.shape}")
        else:
            err = dJdW - true_dJdW
            rmse = tf.sqrt(tf.reduce_mean(tf.square(err)))
            if not np.allclose(dJdW, true_dJdW, atol=1e-07):
                print(f"[{count}] A_0: {A_0.shape}, W: {W.shape}, strides: {strides}, padding: {padding} -> "
                      f"Z: {expected_Z.shape}")
                raise ValueError(f"[{count}] {description}: dJdW has incorrect values. Rmse={rmse}. "
                                 f"Percentiles={tfp.stats.percentile(err, [0, 25, 50, 75, 100])}")

    counters = [0]
    # 1d
    test_scenario(counters, A_0_shape=(32, 100, 5), W_shape=(3, 5, 16), strides=[1], padding='VALID',
                  description='Basic 2D')
    test_scenario(counters, A_0_shape=(32, 100, 5), W_shape=(3, 5, 16), strides=[1], padding='SAME',
                  description='Basic 2D')
    test_scenario(counters, A_0_shape=(32, 100, 4), W_shape=(3, 4, 16), strides=[1], padding='VALID',
                  description='Basic 2D')
    test_scenario(counters, A_0_shape=(32, 100, 4), W_shape=(3, 4, 16), strides=[1], padding='SAME',
                  description='Basic 2D')
    test_scenario(counters, A_0_shape=(32, 100, 5), W_shape=(3, 5, 16), strides=[2], padding='VALID',
                  description='Basic 2D')
    test_scenario(counters, A_0_shape=(32, 100, 5), W_shape=(3, 5, 16), strides=[2], padding='SAME',
                  description='Basic 2D')

    # 2D
    test_scenario(counters, A_0_shape=(32, 8, 9, 5), W_shape=(3, 3, 5, 16), strides=[1, 1], padding='VALID',
                  description='Basic 2D')
    test_scenario(counters, A_0_shape=(32, 8, 9, 5), W_shape=(3, 3, 5, 16), strides=[1, 1], padding='SAME',
                  description='Basic 2D')
    test_scenario(counters, A_0_shape=(32, 17, 16, 5), W_shape=(4, 4, 5, 16), strides=[1, 1], padding='VALID',
                  description='Basic 2D')
    test_scenario(counters, A_0_shape=(32, 17, 16, 5), W_shape=(4, 4, 5, 16), strides=[1, 1], padding='SAME',
                  description='Basic 2D')
    test_scenario(counters, A_0_shape=(32, 8, 9, 5), W_shape=(3, 3, 5, 16), strides=[2, 2], padding='VALID',
                  description='2D with strides')
    test_scenario(counters, A_0_shape=(32, 8, 9, 5), W_shape=(3, 3, 5, 16), strides=[2, 2], padding='SAME',
                  description='2D with strides')
    test_scenario(counters, A_0_shape=(32, 17, 16, 5), W_shape=(4, 4, 5, 16), strides=[2, 2], padding='VALID',
                  description='2D with strides')
    test_scenario(counters, A_0_shape=(32, 17, 16, 5), W_shape=(4, 4, 5, 16), strides=[2, 2], padding='SAME',
                  description='2D with strides')
    test_scenario(counters, A_0_shape=(32, 8, 9, 5), W_shape=(3, 4, 5, 16), strides=[1, 2], padding='VALID',
                  description='Mixed 2D')
    test_scenario(counters, A_0_shape=(32, 8, 9, 5), W_shape=(3, 4, 5, 16), strides=[2, 1], padding='SAME',
                  description='Mixed 2D')
    test_scenario(counters, A_0_shape=(32, 100, 160, 8), W_shape=(3, 3, 8, 32), strides=[2, 2], padding='VALID',
                  description='Large 2D')
    test_scenario(counters, A_0_shape=(32, 100, 160, 8), W_shape=(3, 3, 8, 32), strides=[2, 2], padding='SAME',
                  description='Large 2D')

    # 3D
    test_scenario(counters, A_0_shape=(32, 8, 9, 8, 5), W_shape=(3, 3, 4, 5, 16), strides=[1, 1, 1],
                  padding='VALID', description='Basic 3D')
    test_scenario(counters, A_0_shape=(32, 8, 9, 8, 5), W_shape=(3, 3, 4, 5, 16), strides=[1, 1, 1],
                  padding='SAME', description='Basic 3D')
    test_scenario(counters, A_0_shape=(32, 17, 16, 17, 5), W_shape=(4, 3, 4, 5, 16), strides=[1, 1, 1],
                  padding='VALID', description='Basic 3D')
    test_scenario(counters, A_0_shape=(32, 17, 16, 17, 5), W_shape=(4, 3, 4, 5, 16), strides=[1, 1, 1],
                  padding='SAME', description='Basic 3D')
    test_scenario(counters, A_0_shape=(32, 8, 9, 8, 5), W_shape=(3, 4, 4, 5, 16), strides=[2, 2, 2],
                  padding='VALID', description='3D with strides')
    test_scenario(counters, A_0_shape=(32, 8, 9, 8, 5), W_shape=(3, 4, 4, 5, 16), strides=[2, 2, 2],
                  padding='SAME', description='3D with strides')
    test_scenario(counters, A_0_shape=(32, 7, 8, 9, 5), W_shape=(3, 3, 3, 5, 16), strides=[1, 2, 3],
                  padding='VALID', description='3D with mixed strides')
    test_scenario(counters, A_0_shape=(32, 7, 8, 9, 5), W_shape=(3, 3, 3, 5, 16), strides=[1, 2, 3],
                  padding='VALID', description='3D with mixed strides')
    test_scenario(counters, A_0_shape=(32, 7, 8, 9, 5), W_shape=(4, 4, 4, 5, 16), strides=[1, 2, 3],
                  padding='VALID', description='3D with mixed strides')
    test_scenario(counters, A_0_shape=(32, 7, 8, 9, 5), W_shape=(4, 4, 4, 5, 16), strides=[1, 2, 3],
                  padding='VALID', description='3D with mixed strides')

    # Argument forms
    test_scenario(counters, A_0_shape=(32, 8, 5), W_shape=(3, 5, 16), strides=2, padding='VALID',
                  description='Scalar strides')
    test_scenario(counters, A_0_shape=(32, 8, 9, 5), W_shape=(3, 3, 5, 16), strides=2, padding='VALID',
                  description='Scalar strides')
    test_scenario(counters, A_0_shape=(32, 8, 9, 5), W_shape=(3, 3, 5, 16), strides=[1, 1, 1, 1], padding='VALID',
                  description='Full strides')
    test_scenario(counters, A_0_shape=(32, 8, 9, 6, 5), W_shape=(3, 3, 3, 5, 16), strides=[1, 1, 1, 1, 1],
                  padding='VALID', description='Full strides')
