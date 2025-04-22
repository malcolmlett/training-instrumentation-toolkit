import unittest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from train_instrumentation import *
from train_instrumentation import _normalize_collection_sets_for_layers, _normalize_collection_sets_for_variables


def run_test_suite():
    suite = unittest.defaultTestLoader.loadTestsFromName(__name__)
    unittest.TextTestRunner(verbosity=2).run(suite)


class IndexConversions(unittest.TestCase):
    def test_variable_indices_by_layer(self):
        model = self._create_test_model()

        res = variable_indices_by_layer(model)
        expected = [[0, 1], [2, 3], [4], [5, 6], [7, 8], [9], [10, 11], [12, 13, 14, 15], [16, 17], [18, 19]]
        self.assertEqual(res, expected, f"include_trainable_only=Default: expected {expected}, but got {res}")

        res = variable_indices_by_layer(model, include_trainable_only=False)
        expected = [[0, 1], [2, 3], [4], [5, 6], [7, 8], [9], [10, 11], [12, 13, 14, 15], [16, 17], [18, 19]]
        self.assertEqual(res, expected, f"include_trainable_only=False: expected {expected}, but got {res}")

        res = variable_indices_by_layer(model, include_trainable_only=True)
        expected = [[0, 1], [2, 3], [], [5, 6], [7, 8], [], [10, 11], [12, 13], [16, 17], [18, 19]]
        self.assertEqual(res, expected, f"include_trainable_only=True: expected {expected}, but got {res}")

    def test_trainable_variable_indices_by_layer(self):
        model = self._create_test_model()

        res = trainable_variable_indices_by_layer(model)
        expected = [[0, 1], [2, 3], [], [4, 5], [6, 7], [], [8, 9], [10, 11], [12, 13], [14, 15]]
        self.assertEqual(res, expected, f"Expected {expected}, but got {res}")

    @staticmethod
    def _create_test_model():
        # 20 variables total, 16 trainable
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),  # omitted from layers
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dropout(rate=0.2),  # 1 non-trainable var
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dropout(rate=0.2),  # 1 non-trainable var
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.BatchNormalization(),  # 2 trainable vars + 2 non-trainable vars
            tf.keras.layers.Dense(5, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dense(1, activation='sigmoid')  # 2 trainable vars
        ])
        return model


class CollectionSetHandling(unittest.TestCase):
    def test_normalize_collection_sets_for_layers(self):
        model = self._create_test_model()

        # singleton closed collection sets
        res = _normalize_collection_sets_for_layers(model, [{'layer_indices': [0, 3]}])
        expected = [{'layer_indices': [0, 3]}]
        self.assertEqual(res, expected, f"Accepts layer_indices as is: expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_layers(model, [{'layers': [model.layers[0], model.layers[3]]}])
        expected = [{'layers': [model.layers[0], model.layers[3]], 'layer_indices': [0, 3]}]
        self.assertEqual(res, expected, f"Translates layers: expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_layers(model, [{'layer_names': [model.layers[0].name, model.layers[3].name]}])
        expected = [{'layer_names': [model.layers[0].name, model.layers[3].name], 'layer_indices': [0, 3]}]
        self.assertEqual(res, expected, f"Translates layer names: expected {expected}, but got {res}")

        # multiple closed collection sets
        res = _normalize_collection_sets_for_layers(model, [
            {'layer_indices': [0, 3]},
            {'layers': [model.layers[1], model.layers[4]]},
            {'layer_names': [model.layers[2].name, model.layers[5].name]}])
        expected = [
            {'layer_indices': [0, 3]},
            {'layers': [model.layers[1], model.layers[4]], 'layer_indices': [1, 4]},
            {'layer_names': [model.layers[2].name, model.layers[5].name], 'layer_indices': [2, 5]}]
        self.assertEqual(res, expected, f"Translates across multiple collection sets: expected {expected}, but got {res}")

        # open-ended collection sets
        res = _normalize_collection_sets_for_layers(model, [{}])
        expected = [{'layer_indices': [0, 1, 3, 4, 6, 7, 8, 9]}]
        self.assertEqual(res, expected, f"Expands single all-layers collection set with all trainable layers: expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_layers(model, [{'include_non_trainable': True}])
        expected = [{'include_non_trainable': True, 'layer_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]
        self.assertEqual(res, expected, f"Expands single all-layers collection set with all layers: expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_layers(model, [{'layer_indices': [2, 3, 4]}, {}])
        expected = [{'layer_indices': [2, 3, 4]},
                    {'layer_indices': [0, 1, 6, 7, 8, 9]}]
        self.assertEqual(res, expected, f"Expands open-ended collection set with remaining layers: expected {expected}, but got {res}")

        # error conditions
        with self.assertRaises(ValueError, msg="Expected duplicate error given variable_indices and "
                                               "trainable_variable_indices"):
            _normalize_collection_sets_for_variables(model, [
                {'variable_indices': [2, 3, 4]},
                {'trainable_variable_indices': [4, 5, 6, 2]}])

        with self.assertRaises(ValueError, msg="Expected duplicate error given variable_indices and layer_indices"):
            _normalize_collection_sets_for_variables(model, [
                {'variable_indices': [2, 3, 4]},
                {'layer_indices': [1]}])

    def test_normalize_collection_sets_for_variables(self):
        model = self._create_test_model()

        # singleton closed collection sets
        res = _normalize_collection_sets_for_variables(model, [{'variable_indices': [2, 3, 4]}])
        expected = [{'variable_indices': [2, 3, 4]}]
        self.assertEqual(res, expected, f"Accepts as is when variable_indices given directly: expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_variables(model, [{'trainable_variable_indices': [2, 3, 4, 5]}])
        expected = [{'trainable_variable_indices': [2, 3, 4, 5], 'variable_indices': [2, 3, 5, 6]}]
        self.assertEqual(res, expected, f"Translates trainable_variable_indices: expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_variables(model, [{'layers': [model.layers[0], model.layers[3]]}])
        expected = [{'layers': [model.layers[0], model.layers[3]], 'variable_indices': [0, 1, 5, 6]}]
        self.assertEqual(res, expected, f"Translates layers: expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_variables(model, [{'layer_indices': [0, 3]}])
        expected = [{'layer_indices': [0, 3], 'variable_indices': [0, 1, 5, 6]}]
        self.assertEqual(res, expected, f"Translates layer indices: expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_variables(model, [{'layer_names': [model.layers[0].name, model.layers[3].name]}])
        expected = [{'layer_names': [model.layers[0].name, model.layers[3].name], 'variable_indices': [0, 1, 5, 6]}]
        self.assertEqual(res, expected, f"Translates layer names: expected {expected}, but got {res}")

        # multiple closed collection sets
        res = _normalize_collection_sets_for_variables(model, [
            {'variable_indices': [2, 3, 4]},
            {'layer_indices': [0]},
            {'trainable_variable_indices': [10, 12, 14]}])
        expected = [{'variable_indices': [2, 3, 4]},
                    {'layer_indices': [0], 'variable_indices': [0, 1]},
                    {'trainable_variable_indices': [10, 12, 14], 'variable_indices': [12, 16, 18]}]
        self.assertEqual(res, expected, f"Translates across multiple collection sets: expected {expected}, but got {res}")

        # open-ended collection sets
        res = _normalize_collection_sets_for_variables(model, [{}])
        expected = [{'variable_indices': [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 19]}]
        self.assertEqual(res, expected, f"Expands single all-variables collection set with all trainable variables: " \
                                        f"expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_variables(model, [{'variable_indices': [2, 3, 4]}, {}])
        expected = [
            {'variable_indices': [2, 3, 4]},
            {'variable_indices': [0, 1, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 19]}]
        self.assertEqual(res, expected, f"Expands open-ended collection set with remaining trainable variables: " \
                                        f"expected {expected}, but got {res}")

        res = _normalize_collection_sets_for_variables(model, [
            {'variable_indices': [2, 3, 4]},
            {'include_non_trainable': True}])
        expected = [
            {'variable_indices': [2, 3, 4]},
            {'include_non_trainable': True, 'variable_indices': [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}]
        self.assertEqual(res, expected, f"Expands open-ended collection set with all remaining variables: " \
                                        f"expected {expected}, but got {res}")

        # error conditions
        with self.assertRaises(ValueError, msg="Expected duplicate error given given variable_indices and "
                                               "trainable_variable_indices"):
            _normalize_collection_sets_for_variables(model, [{'variable_indices': [2, 3, 4]},
                                                             {'trainable_variable_indices': [4, 5, 6, 2]}])

        with self.assertRaises(ValueError, msg="Expected duplicate error given variable_indices and layer_indices"):
            _normalize_collection_sets_for_variables(model, [{'variable_indices': [2, 3, 4]},
                                                             {'layer_indices': [1]}])

    @staticmethod
    def _create_test_model():
        # 20 variables total, 16 trainable
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),  # omitted from layers
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dropout(rate=0.2),  # 1 non-trainable var
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dropout(rate=0.2),  # 1 non-trainable var
            tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
            tf.keras.layers.BatchNormalization(),  # 2 trainable vars + 2 non-trainable vars
            tf.keras.layers.Dense(5, activation='relu'),  # 2 trainable vars
            tf.keras.layers.Dense(1, activation='sigmoid')  # 2 trainable vars
        ])
        return model


class NormAccumulatorStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        # minimise chance of random variations to cause sporadic test failures
        tf.keras.utils.set_random_seed(1)

        self.var_list = [tf.random.normal((5, 10, 10), dtype=tf.float32), None,
                         tf.random.normal((5, 3, 3, 32, 20), dtype=tf.float32)]

    def test_accumulated_norms(self):
        # expected
        expected = [self.one_expected_norm(v) if v is not None else None for v in self.var_list]

        # actual
        accumulator = NormAccumulatorStrategy()
        for iteration in range(self.var_list[0].shape[0]):
            iteration_data = [v[iteration] if v is not None else None for v in self.var_list]
            accumulator.accumulate(iteration == 0, iteration_data)
        actual = accumulator.accumulated_norms

        same_states = [close_or_none(e, a) for e, a in zip(expected, actual)]
        msg = f"Differences found. Matches by variable: {same_states}\n"\
              f"-- Expected: {expected}\n"\
              f"-- Actual: {actual}"
        self.assertEqual(np.all(same_states), True, msg)

    def test_immediate_norms(self):
        expected = [self.one_expected_norm(v) if v is not None else None for v in self.var_list]

        accumulator = NormAccumulatorStrategy()
        actual = accumulator.norms(self.var_list)

        same_states = [close_or_none(e, a) for e, a in zip(expected, actual)]
        msg = f"Differences found. Matches by variable: {same_states}\n"\
              f"-- Expected: {expected}\n"\
              f"-- Actual: {actual}"
        self.assertEqual(np.all(same_states), True, msg)

    @staticmethod
    def one_expected_norm(tensor):
        return tf.sqrt(tf.reduce_mean(tf.square(tensor)))


class BasicStatsAccumulatorStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        # minimise chance of random variations to cause sporadic test failures
        tf.keras.utils.set_random_seed(1)

        self.var_list = [tf.random.normal((5, 10, 10), dtype=tf.float32), None,
                         tf.random.normal((5, 3, 3, 32, 20), dtype=tf.float32)]

    def test_quantiles(self):
        accumulator = BasicStatsAccumulatorStrategy(abs_log_scale=False)
        self.assertEqual(accumulator.quantiles, [0., 32., 50., 68., 100.], "standard")

        accumulator = BasicStatsAccumulatorStrategy(abs_log_scale=True)
        self.assertEqual(accumulator.quantiles, [0., 32., 50., 68., 100.], "abs-log-scale")

    def test_accumulated_percentiles_given_linear_scale(self):
        # expected
        expected = [self.one_expected_percentile(v, abs_log_scale=False)
                    if v is not None else None for v in self.var_list]

        # actual
        accumulator = BasicStatsAccumulatorStrategy(abs_log_scale=False)
        for iteration in range(self.var_list[0].shape[0]):
            iteration_data = [v[iteration] if v is not None else None for v in self.var_list]
            accumulator.accumulate(iteration == 0, iteration_data)
        actual = accumulator.accumulated_percentiles

        same_states = [close_or_none(e, a) for e, a in zip(expected, actual)]
        msg = f"Differences found. Matches by variable: {same_states}\n"\
              f"-- Expected: {expected}\n"\
              f"-- Actual: {actual}"
        self.assertEqual(np.all(same_states), True, msg)

    def test_accumulated_percentiles_given_abs_log_scale(self):
        # expected
        expected = [self.one_expected_percentile(v, abs_log_scale=True)
                    if v is not None else None for v in self.var_list]

        # actual
        accumulator = BasicStatsAccumulatorStrategy(abs_log_scale=True)
        for iteration in range(self.var_list[0].shape[0]):
            iteration_data = [v[iteration] if v is not None else None for v in self.var_list]
            accumulator.accumulate(iteration == 0, iteration_data)
        actual = accumulator.accumulated_percentiles

        same_states = [close_or_none(e, a) for e, a in zip(expected, actual)]
        msg = f"Differences found. Matches by variable: {same_states}\n"\
              f"-- Expected: {expected}\n"\
              f"-- Actual: {actual}"
        self.assertEqual(np.all(same_states), True, msg)

    def test_accumulated_percentiles_given_abs_log_scale_and_zeros(self):
        def randomly_zerofy_one(tensor):
            zero_out = tf.random.uniform(tensor.shape, maxval=1.0, dtype=tf.float32) < 0.5
            return tf.where(zero_out, tf.zeros_like(tensor), tensor)
        var_list_with_zeros = [randomly_zerofy_one(v) if v is not None else None for v in self.var_list]

        # expected
        expected = [self.one_expected_percentile(v, abs_log_scale=True)
                    if v is not None else None for v in var_list_with_zeros]

        # actual
        accumulator = BasicStatsAccumulatorStrategy(abs_log_scale=True)
        for iteration in range(self.var_list[0].shape[0]):
            iteration_data = [v[iteration] if v is not None else None for v in var_list_with_zeros]
            accumulator.accumulate(iteration == 0, iteration_data)
        actual = accumulator.accumulated_percentiles

        same_states = [close_or_none(e, a) for e, a in zip(expected, actual)]
        msg = f"Differences found. Matches by variable: {same_states}\n"\
              f"-- Expected: {expected}\n"\
              f"-- Actual: {actual}"
        self.assertEqual(np.all(same_states), True, msg)

    def test_immediate_percentiles_given_linear_scale(self):
        expected = [self.one_expected_percentile(v, abs_log_scale=False)
                    if v is not None else None for v in self.var_list]

        accumulator = BasicStatsAccumulatorStrategy(abs_log_scale=False)
        actual = accumulator.percentiles(self.var_list)

        same_states = [close_or_none(e, a) for e, a in zip(expected, actual)]
        msg = f"Differences found. Matches by variable: {same_states}\n"\
              f"-- Expected: {expected}\n"\
              f"-- Actual: {actual}"
        self.assertEqual(np.all(same_states), True, msg)

    def test_immediate_percentiles_given_abs_log_scale(self):
        expected = [self.one_expected_percentile(v, abs_log_scale=True)
                    if v is not None else None for v in self.var_list]

        accumulator = BasicStatsAccumulatorStrategy(abs_log_scale=True)
        actual = accumulator.percentiles(self.var_list)

        same_states = [close_or_none(e, a) for e, a in zip(expected, actual)]
        msg = f"Differences found. Matches by variable: {same_states}\n"\
              f"-- Expected: {expected}\n"\
              f"-- Actual: {actual}"
        self.assertEqual(np.all(same_states), True, msg)

    @staticmethod
    def one_expected_percentile(tensor, abs_log_scale):
        if abs_log_scale:
            # using different log(zero) handling than in actual implementation
            # in order to verify that the handling is accurate enough
            epsilon = 1e-12
            has_zero = np.any(tensor == 0.0)

            tensor = tf.where(tensor == 0.0, tf.ones_like(tensor) * epsilon, tensor)
            tensor = tf.math.log(tf.abs(tensor))
            min = tf.reduce_min(tensor)
            max = tf.reduce_max(tensor)
            mean = tf.reduce_mean(tensor)
            sd = tf.math.reduce_std(tensor)
            p = tf.stack([min, mean - sd, mean, mean + sd, max])
            p = tf.math.exp(p)
            if has_zero:
                first_q = tf.where(p[1] < epsilon, 0.0, p[1])
                p = tf.stack([0.0, first_q, p[2], p[3], p[4]])
            return p
        else:
            min = tf.reduce_min(tensor)
            max = tf.reduce_max(tensor)
            mean = tf.reduce_mean(tensor)
            sd = tf.math.reduce_std(tensor)
            return tf.stack([min, mean-sd, mean, mean+sd, max])


class PercentileAccumulatorStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        # minimise chance of random variations to cause sporadic test failures
        tf.keras.utils.set_random_seed(1)

        self.quantiles = [0., 24., 52., 76.5, 100.]
        self.var_list = [tf.random.normal((5, 10, 10), dtype=tf.float32), None,
                         tf.random.normal((5, 3, 3, 32, 20), dtype=tf.float32)]

    def test_accumulated_percentiles(self):
        # expected
        def compute_one(var_data):
            raw_percentiles = tfp.stats.percentile(var_data, self.quantiles, interpolation='linear',
                                                   axis=tf.range(1, tf.rank(var_data)))  # shape: (quantiles, epochs)
            raw_percentiles = tf.transpose(raw_percentiles)  # shape: (epochs, quantiles)
            return tf.reduce_mean(raw_percentiles, axis=0)  # shape: (quantiles,)
        expected = [compute_one(v) if v is not None else None for v in self.var_list]

        # actual
        accumulator = PercentileAccumulatorStrategy(quantiles=self.quantiles)
        for iteration in range(self.var_list[0].shape[0]):
            iteration_data = [v[iteration] if v is not None else None for v in self.var_list]
            accumulator.accumulate(iteration == 0, iteration_data)
        actual = accumulator.accumulated_percentiles

        self.assertEquals(accumulator.quantiles, self.quantiles)

        same_states = [close_or_none(e, a) for e, a in zip(expected, actual)]
        msg = f"Differences found. Matches by variable: {same_states}\n"\
              f"-- Expected: {expected}\n"\
              f"-- Actual: {actual}"
        self.assertEqual(np.all(same_states), True, msg)

    def test_immediate_percentiles(self):
        expected = [tfp.stats.percentile(v, self.quantiles, interpolation='linear')
                    if v is not None else None for v in self.var_list]

        accumulator = PercentileAccumulatorStrategy(quantiles=self.quantiles)
        actual = accumulator.percentiles(self.var_list)

        same_states = [close_or_none(e, a) for e, a in zip(expected, actual)]
        msg = f"Differences found. Matches by variable: {same_states}\n"\
              f"-- Expected: {expected}\n"\
              f"-- Actual: {actual}"
        self.assertEqual(np.all(same_states), True, msg)


class CallbackTrainingTestMixin:
    """
    Bunch of capabilities for creating models, doing fake training runs, for selecting data along various
    axis, and for asserting on the results.
    """

    def fake_train(self, model, variable_data=None, gradient_data=None, output_data=None, output_gradient_data=None,
                   batch_size=32, callbacks=None, use_variable_data_before=False, verbose=0):
        """
        Fakes the behaviour of model training w.r.t. what gets passed to callbacks at each event.

        Params:
            model: model supposedly being trained
            batch_size: used to break data into batches
            variable_data: optional, list (by model variable) of tensor of shape (epochs, steps, ..variable-dims..).
                Includes non-trainable variables.
            gradient_data: optional, list (by trainable variable) of tensor with shape
                (epochs, steps, ..variable-dims..), where:
                    epochs = number of epochs
                    steps = number of steps in each epoch
            output_data: optional, list (by layer) of tensor with shape (epochs, dataset-size, ..data-dims..),
                where:
                    epochs = number of epochs
                    dataset-size = total number of samples in entire dataset (must be same for all variables)
            output_gradient_data: optional, list (by layer) of tensor with shape (epochs, dataset-size, ..data-dims..),
                where:
                    epochs = number of epochs
                    dataset-size = total number of samples in entire dataset (must be same for all variables)
            use_variable_data_before: bool.
                Whether to apply variable state before training step, or after (default).
        """
        # sanity checks - requirements:
        # - variable_data must have same length as model.variables, and same shapes and types
        # - gradient_data must have same length as trainable variables (ideally same shapes and types)
        # - layer_output_data must have same length as model.layers (ideally same shapes and types)
        # - layer_gradient_data must have same length as model.layers (ideally same shapes and types)
        if gradient_data is not None:
            if len(gradient_data) != len(model.trainable_variables):
                raise ValueError(
                    f"gradient_data has wrong number of variables. "
                    f"Expected {len(model.trainable_variables)}, got {len(gradient_data)}")

        # determine basic coordinates
        epochs, n_steps = None, None
        if gradient_data is not None:
            for var_data in gradient_data:
                epochs, n_steps = var_data.shape[0:2]
        elif output_data is not None:
            dataset_size = None
            for layer_data in output_data:
                if layer_data is not None:
                    epochs, dataset_size = layer_data.shape[0:2]
            n_steps = math.ceil(dataset_size / batch_size)
        elif output_gradient_data is not None:
            dataset_size = None
            for layer_data in output_gradient_data:
                if layer_data is not None:
                    epochs, dataset_size = layer_data.shape[0:2]
            n_steps = math.ceil(dataset_size / batch_size)

        # prepare callbacks tracking
        gradient_callbacks = []
        if callbacks is not None and not isinstance(callbacks, tf.keras.callbacks.CallbackList):
            gradient_callbacks = [callback for callback in callbacks if
                                  reload_safe_isinstance(callback, BaseGradientCallback)]
            callbacks = [callback for callback in callbacks if
                         not reload_safe_isinstance(callback, BaseGradientCallback)]
        if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
            callbacks = tf.keras.callbacks.CallbackList(
                callbacks, add_history=False, add_progbar=False, verbose=False, epochs=epochs,
                steps=n_steps, model=model)
        for gradient_callback in gradient_callbacks:
            gradient_callback.set_params({'epochs': epochs, 'steps': n_steps})
            gradient_callback.set_model(model)

        # train begin
        if verbose > 0:
            print(f"Fake training for {epochs} epochs x {n_steps} batches of {batch_size} or less")
        callbacks.on_train_begin()
        for gradient_callback in gradient_callbacks:
            gradient_callback.on_train_begin()

        # training loop by epoch
        logs = {}
        batch_gradients, batch_outputs, batch_outgrads = None, None, None
        for epoch in range(epochs):
            # simulate variable state BEFORE update step
            batch_variables = self.select_variable_batch(variable_data, epoch, 0)
            if use_variable_data_before:
                self.set_model_variables(model, batch_variables)

            # notify callbacks
            callbacks.on_epoch_begin(epoch)
            for gradient_callback in gradient_callbacks:
                gradient_callback.on_epoch_begin(epoch)

            # training loop by step
            for step in range(n_steps):
                # simulate variable state BEFORE update step
                # - technically already done for step 0 but no harm in repeating
                batch_variables = self.select_variable_batch(variable_data, epoch, step)
                if use_variable_data_before:
                    self.set_model_variables(model, batch_variables)

                # notify callbacks
                callbacks.on_train_batch_begin(step)
                for gradient_callback in gradient_callbacks:
                    gradient_callback.on_train_batch_begin(step)

                # simulate training update step
                logs = {'loss': 1.0}
                batch_gradients = self.select_variable_batch(gradient_data, epoch, step)
                batch_outputs = self.select_layer_batch(output_data, epoch, step, batch_size)
                batch_outgrads = self.select_layer_batch(output_gradient_data, epoch, step, batch_size)
                if verbose > 0:
                    print(f"  epoch: {epoch}, step: {step}", end='')
                    if batch_variables is not None:
                        print(f", variables: {describe(batch_variables)}", end='')
                    if batch_gradients is not None:
                        print(f", gradients: {describe(batch_gradients, verbose=2)}", end='')
                    if batch_outputs is not None:
                        print(f", outputs: {describe(batch_outputs)}", end='')
                    if batch_outgrads is not None:
                        print(f", outgrads: {describe(batch_outgrads)}", end='')
                    print('')

                # simulate variable state AFTER update step
                if use_variable_data_before:
                    self.set_model_variables(model, batch_variables)

                # notify callbacks
                callbacks.on_train_batch_end(step, logs)
                for gradient_callback in gradient_callbacks:
                    gradient_callback.on_train_batch_end(
                        batch=step,
                        loss=logs['loss'],
                        gradients=batch_gradients,
                        trainable_variables=model.trainable_variables,
                        activations=batch_outputs,
                        output_gradients=batch_outgrads if gradient_callback.needs_output_gradients else None)

            # end of epoch
            callbacks.on_epoch_end(epoch, logs)  # should be passing loss and mse
            for gradient_callback in gradient_callbacks:
                gradient_callback.on_epoch_end(
                    epoch=epoch,
                    loss=logs['loss'],
                    gradients=batch_gradients,
                    trainable_variables=model.trainable_variables,
                    activations=batch_outputs,
                    output_gradients=batch_outgrads if gradient_callback.needs_output_gradients else None)

        # end of training
        callbacks.on_train_end(logs)  # should be passing loss and mse
        for gradient_callback in gradient_callbacks:
            gradient_callback.on_train_end()

    @staticmethod
    def make_model(input_shape, kernel_shapes, biases=None):
        """
        Streamlined way of creating a model suitable for the test scenarios. The API is designed to make
        the data shapes explicit, so that it's easier to eyeball the test code to verify that it's correct.
        Params:
          kernel_shapes: shape of layer kernels
            It has the following forms:
            `(in_channels,out_channels)` - creates a Dense layer
            `(k1,k2,in_channels,out_channels)` - creates a Conv2D layer with the specified kernel shape
            `dropout` - creates a Dropout layer that has only non-trainable parameters
            `flatten` - creates a Flatten layer that has no parameters
          biases: optional, list of bool.
            To reduce simulation effort, layers are created without biases by default.
            This specifies which layers should have biases.
        """
        layer_list = [tf.keras.layers.InputLayer(shape=input_shape)]
        cur_input_shape = input_shape

        if biases is not None and len(biases) != len(kernel_shapes):
            raise ValueError(f"biases (len {len(biases)}) not same length as kernel_shapes (len {len(kernel_shapes)})")

        for l_idx, kernel_shape in enumerate(kernel_shapes):
            use_bias = biases[l_idx] if biases is not None else False

            if kernel_shape == 'dropout':
                # non-trainable layer with non-trainable parameters
                layer_list.append(tf.keras.layers.Dropout(0.2))
                cur_input_shape = cur_input_shape  # no change
            elif kernel_shape == 'flatten':
                # non-trainable layer without any parameters
                layer_list.append(tf.keras.layers.Flatten())
                cur_input_shape = (math.prod(cur_input_shape),)
            elif kernel_shape == 'batch-norm':
                # layer with mixture of trainable and non-trainable parameters
                layer_list.append(tf.keras.layers.BatchNormalization())
                cur_input_shape = cur_input_shape  # no change
            elif len(kernel_shape) == 2:
                # simple dense layer
                if kernel_shape[-2] != cur_input_shape[-1]:
                    raise ValueError(f"kernel_shape {kernel_shape} not compatible with "
                                     f"layer input shape {cur_input_shape}")
                layer_list.append(tf.keras.layers.Dense(units=kernel_shape[-1], use_bias=use_bias))
                cur_input_shape = (kernel_shape[-1],)  # simplistic but will work for our needs
            elif len(kernel_shape) == 4:
                # standard conv layer
                if kernel_shape[-2] != cur_input_shape[-1]:
                    raise ValueError(f"kernel_shape {kernel_shape} not compatible with "
                                     f"layer input shape {cur_input_shape}")
                layer_list.append(tf.keras.layers.Conv2D(
                    filters=kernel_shape[3], kernel_size=kernel_shape[0:2], use_bias=use_bias))
                cur_input_shape = (cur_input_shape[0] - 2, cur_input_shape[1] - 2,
                                   kernel_shape[-1])  # simplistic but will work for our needs
            else:
                raise ValueError(f"Unknown layer shape: {kernel_shape}")
        return tf.keras.Sequential(layer_list)

    @staticmethod
    def get_layer_output_shapes(model):
        """
        Gets the output shapes of each layer in the given model.
        """
        return [l.compute_output_shape(l.input.shape) for l in model.layers]

    @staticmethod
    def set_model_variables(model, new_variable_states):
        """
        Correctly sets all model variables, and copes with minor type coercion
        """
        if new_variable_states is None:
            return
        for variable, new_variable_state in zip(model.variables, new_variable_states):
            variable.assign(new_variable_state)

    @staticmethod
    def expand_to_all_variables(model, list_by_trainable_var):
        """
        Takes a list-by-trainable-variable and expands it to a list-by-variable, inserting Nones where needed.
        """
        indices = trainable_variable_indices_to_variable_indices(model)
        res = [None for i in range(len(model.variables))]
        for i, var in enumerate(list_by_trainable_var):
            res[indices[i]] = var
        return res

    @staticmethod
    def select_epoch(list_of_epochs_of_datasets, epoch_index):
        """
        Dataset operation that selects datasets at indexed epoch.
        Params:
          list_of_epochs_of_datasets: list (by variable/layer) of tensor with shape (epochs, ..other-dims..)
        Returns:
          list (by variable/layer) of tensor with shape (..other-dims..)
        """
        if list_of_epochs_of_datasets is None:
            return None
        return [dataset[epoch_index] if dataset is not None else None for dataset in list_of_epochs_of_datasets]

    @staticmethod
    def select_variable_batch(list_of_epochs_of_datasets, epoch_index, batch_index):
        """
        Dataset operation that selects datasets at indexed epoch and step, suitable for use against variable datasets,
        which have one sample per step.
        Params:
          list_of_epochs_of_datasets: list (by variable) of tensor with shape (epochs, steps, ..variable-dims..)
        Returns:
          list (by variable) of tensor with shape (..variable-dims..)
        """
        if list_of_epochs_of_datasets is None:
            return None
        return [dataset[epoch_index, batch_index] for dataset in list_of_epochs_of_datasets]

    @staticmethod
    def select_layer_batch(list_of_epochs_of_datasets, epoch_index, batch_index, batch_size):
        """
        Dataset operation that selects datasets at indexed epoch and step, suitable for use against layer datasets,
        which have multiple samples per step.
        Params:
          list_of_epochs_of_datasets: list (by layer) of tensor with shape (<optional>epochs, dataset-size, ..layer-output-dims..)
          epoch_index: indexing by epoch, or None if there is no epoch dimension
          batch_index: indexing by batch
          batch_size: size of each batch
        Returns:
          list (by layer) of tensor with shape (batch_size, ..data-dims..)
        """
        if list_of_epochs_of_datasets is None:
            return None
        if epoch_index is None:
            # no epoch dimension
            # - note: clipping to dimension length is done implicitly
            return [dataset[batch_index * batch_size:(batch_index + 1) * batch_size]
                    if dataset is not None else None
                    for dataset in list_of_epochs_of_datasets]
        else:
            # epoch dimension present
            # - note: clipping to dimension length is done implicitly
            return [dataset[epoch_index, batch_index * batch_size:(batch_index + 1) * batch_size]
                    if dataset is not None else None
                    for dataset in list_of_epochs_of_datasets]

    @staticmethod
    def map_each_item(list_of_item_datasets, item_fn):
        """
        Dataset operation that applies a mapping function to each non-None item in the provided list
        """
        if list_of_item_datasets is None:
            return None
        return [item_fn(item) if item is not None else None for item in list_of_item_datasets]

    @staticmethod
    def map_each_epoch(list_of_epochs_of_datasets, dataset_fn):
        """
        Dataset operation that applies a mapping function against each epoch of data.
        Params:
          list_of_epochs_of_datasets: list (by variable/layer) of tensor with shape (epochs, ..other-dims..),
            may have Nones for some variables/layers
          dataset_fn: function applied to each dataset at each epoch.
            Takes a tensor of shape (..other-dims..) and returns anything
        Returns:
          list (by variable/layer) of list (by epoch) of results of fn() against each epoch of data,
            with Nones for some variables/layers
        """
        if list_of_epochs_of_datasets is None:
            return None

        def map_one_item(dataset):
            return [dataset_fn(dataset[epoch]) for epoch in range(dataset.shape[0])]

        return [map_one_item(dataset) if dataset is not None else None for dataset in list_of_epochs_of_datasets]

    def map_each_layer_batch(self, epoch_of_datasets, batch_fn, batch_size=32, batch_first=False):
        """
        Dataset operation that applies a mapping function against each batch of data within a single epoch.
        Params:
          epoch_of_datasets: list (by layer) of tensor with shape (dataset-size, ..other-dims..),
            may have Nones for some layers
          batch_fn: function applied to each batch of data for each layer.
            Takes a tensor of shape (batch-size, ..other-dims..) and returns anything
          batch_first: whether to have list axes as (batch, layer), or otherwise as (layer, batch) (default).
        Returns:
          list (by layer) of list (by batch) of results of fn() against each batch of data,
            with Nones for some layers,
          OR if batch_first is True then
          list (by batch) of list (by layer) of results of fn() against each batch of data,
            with Nones for some layers.
        """
        if epoch_of_datasets is None:
            return None

        dataset_size = None
        for layer_data in epoch_of_datasets:
            if layer_data is not None:
                dataset_size = layer_data.shape[0]
        n_steps = math.ceil(dataset_size / batch_size)

        if batch_first:
            res = []  # list with shape: (n_steps, n_layers), of Any
            for step in range(n_steps):
                # batch_datas shape: list (by items) of tensor (batch-size-or-less, ..other-dims..)
                batch_datas = self.select_layer_batch(
                    epoch_of_datasets, epoch_index=None, batch_index=step, batch_size=batch_size)
                batch_datas = [batch_fn(data) if data is not None else None for data in batch_datas]
                res.append(batch_datas)
            return res

        else:
            # list with shape: (n_layers, n_steps), of Any
            res = [[] if v is not None else None for v in epoch_of_datasets]
            for step in range(n_steps):
                # batch_datas shape: list (by items) of tensor (batch-size-or-less, ..other-dims..)
                batch_datas = self.select_layer_batch(
                    epoch_of_datasets, epoch_index=None, batch_index=step, batch_size=batch_size)
                batch_datas = [batch_fn(data) if data is not None else None for data in batch_datas]
                for l_idx, data in enumerate(batch_datas):
                    if data is not None:
                        res[l_idx].append(data)
            return res

    @staticmethod
    def map_each_variable_batch(epoch_of_datasets, batch_fn):
        """
        Dataset operation that applies a mapping function against each batch of data within a single epoch.
        Params:
          epoch_of_datasets: list (by variable) of tensor with shape (steps, ..other-dims..),
            may have Nones for some variables
          batch_fn: function applied to each batch of data for each variable.
            Takes a tensor of shape (..other-dims..) and returns anything
        Returns:
          list (by variable) of list (by batch) of results of fn() against each batch of data,
            with Nones for some variables
        """
        if epoch_of_datasets is None:
            return None

        def map_one_item(dataset):
            return [batch_fn(dataset[step]) for step in range(dataset.shape[0])]

        return [map_one_item(dataset) if dataset is not None else None for dataset in epoch_of_datasets]

    def flatmap_epoch_batches(self, list_of_epochs_of_datasets, datasets_fn):
        """
        Dataset operation that applies a mapping function against each batch of data, flattening with epoch dimension.
        Usually the mapping function will be one of map_each_layer_batch() or map_each_variable_batch().
        Params:
          list_of_epochs_of_datasets: list (by variable/layer) of tensor with shape (epochs, batch-dim, ..other-dims..),
            may have Nones for some variables/layers,
            where batch-dim is usually either a full dataset or one entry for each step in an epoch.
          datasets_fn: function applied to lists of datasets, for each epoch.
            Takes a list (by variable/layer) of tensors of shape (batch-dim, ..other-dims..) and returns
            and return list (by variable/layer) of list (by batch) of Anything
        Returns:
          list (by variable/layer) of list (by batch) of Anything,
            with Nones for some variables/layers
        """
        if list_of_epochs_of_datasets is None:
            return None

        n_epochs = None
        for item_dataset in list_of_epochs_of_datasets:
            if item_dataset is not None:
                n_epochs = item_dataset.shape[0]

        # list with shape: (n_layers, n_steps), of Any
        res = [[] if v is not None else None for v in list_of_epochs_of_datasets]
        for epoch in range(n_epochs):
            epoch_of_datasets = self.select_epoch(list_of_epochs_of_datasets, epoch)
            list_of_batches = datasets_fn(epoch_of_datasets)
            for item_idx, data in enumerate(list_of_batches):
                if data is not None:
                    res[item_idx].extend(data)
        return res


class CallbackTrainingTestMixinTest(unittest.TestCase):
    """
    Validate helper functionality for the purpose of helping with tests.
    """

    def setUp(self):
        self.target = CallbackTrainingTestMixin()

    def test_make_model_given_fnn(self):
        # layers:
        # - input:   output (64,)
        # - dropout: seed variable of shape (2,), output (64,)
        # - dense:   kernel (64, 128), no bias, output (128,)
        tgt = self.target
        model = tgt.make_model((64,), ['dropout', (64, 128)])
        self.assertEqual(describe(model.variables), [(2,), (64, 128)])
        self.assertEqual(describe(model.trainable_variables), [(64, 128)])
        self.assertEqual(tgt.get_layer_output_shapes(model), [(None, 64), (None, 128)])

    def test_make_model_given_cnn(self):
        # layers:
        # - input:     output (5,5,1)
        # - conv2d:    kernel (3,3,1,4), bias (4,), output (5-2,5-2,4)
        # - flatten:   no variables, output (36,)
        # - dense:     kernel (36,128), no bias, output (128,)
        # - batchNorm: two trainable variables (128,) each, two non-trainable variables (128,) each, output (128,)
        tgt = self.target
        model = tgt.make_model((5, 5, 1), [(3, 3, 1, 4), 'flatten', (36, 128), 'batch-norm'],
                               biases=[True, False, False, False])
        self.assertEqual(describe(model.variables), [(3, 3, 1, 4), (4,), (36, 128), (128,), (128,), (128,), (128,)])
        self.assertEqual(describe(model.trainable_variables), [(3, 3, 1, 4), (4,), (36, 128), (128,), (128,)])
        self.assertEqual(tgt.get_layer_output_shapes(model), [(None, 3, 3, 4), (None, 36), (None, 128), (None, 128)])

    def test_expand_to_all_variables(self):
        tgt = self.target
        model = self.target.make_model((64,), ['dropout', (64, 128)])
        gradients = [tf.random.normal((2, 2, 64, 128))]
        self.assertEqual(describe(tgt.expand_to_all_variables(model, model.trainable_variables)), [None, (64, 128)])
        self.assertEqual(describe(tgt.expand_to_all_variables(model, gradients)), [None, (2, 2, 64, 128)])

    def test_dataset_selection(self):
        tgt = self.target
        gradients = [tf.random.normal((2, 2, 64, 128))]
        outputs = [None, tf.random.normal((2, 40, 128))]

        self.assertEqual(describe(tgt.select_epoch(gradients, 0)), [(2, 64, 128)])
        self.assertEqual(describe(tgt.select_epoch(gradients, 1)), [(2, 64, 128)])
        self.assertEqual(describe(tgt.select_variable_batch(gradients, 0, 0)), [(64, 128)])
        self.assertEqual(describe(tgt.select_variable_batch(gradients, 0, 1)), [(64, 128)])
        self.assertEqual(describe(tgt.select_epoch(outputs, 0)), [None, (40, 128)])
        self.assertEqual(describe(tgt.select_epoch(outputs, 1)), [None, (40, 128)])
        self.assertEqual(describe(tgt.select_layer_batch(outputs, 0, 0, batch_size=32)), [None, (32, 128)])
        self.assertEqual(describe(tgt.select_layer_batch(outputs, 0, 1, batch_size=32)), [None, (8, 128)])

    def test_dataset_mapping(self):
        tgt = self.target
        model = tgt.make_model((64,), ['dropout', (64, 128)])
        gradients = [tf.random.normal((2, 2, 64, 128))]
        outputs = [None, tf.random.normal((2, 40, 128))]
        all_gradients = tgt.expand_to_all_variables(model, gradients)
        first_epoch_gradients = tgt.select_epoch(all_gradients, 0)
        first_epoch_outputs = tgt.select_epoch(outputs, 0)

        # basic sanity checks
        self.assertEqual(describe(gradients), [(2, 2, 64, 128)])
        self.assertEqual(describe(all_gradients), [None, (2, 2, 64, 128)])

        self.assertEqual(describe(first_epoch_gradients), [None, (2, 64, 128)])
        self.assertEqual(describe(tgt.map_each_variable_batch(
            first_epoch_gradients, lambda batch_data: batch_data)),
            [None, [(64, 128), (64, 128)]])
        self.assertEqual(describe(first_epoch_outputs), [None, (40, 128)])
        self.assertEqual(describe(tgt.map_each_layer_batch(
            first_epoch_outputs, lambda batch_data: batch_data)),
            [None, [(32, 128), (8, 128)]])

        self.assertEqual(describe(all_gradients), [None, (2, 2, 64, 128)])
        self.assertEqual(describe(tgt.map_each_epoch(
            all_gradients, lambda epoch_data: epoch_data)),
            [None, [(2, 64, 128), (2, 64, 128)]])
        self.assertEqual(describe(tgt.flatmap_epoch_batches(
            all_gradients, lambda datasets: tgt.map_each_variable_batch(datasets, lambda batch_data: batch_data))),
            [None, [(64, 128), (64, 128), (64, 128), (64, 128)]])

        self.assertEqual(describe(outputs), [None, (2, 40, 128)])
        self.assertEqual(describe(tgt.map_each_epoch(
            outputs, lambda epoch_data: epoch_data)),
            [None, [(40, 128), (40, 128)]])
        self.assertEqual(describe(tgt.flatmap_epoch_batches(
            outputs, lambda datasets: tgt.map_each_layer_batch(datasets, lambda batch_data: batch_data))),
            [None, [(32, 128), (8, 128), (32, 128), (8, 128)]])


class GradientHistoryCallbackTest(unittest.TestCase, CallbackTrainingTestMixin):
    """
    Validate helper functionality for the purpose of helping with tests.
    """
    def setUp(self):
        self.model = self.make_model((64,), ['dropout', (64, 128)])
        self.variable_datasets = [
            tf.random.uniform((2, 2, 2), maxval=100, dtype=tf.int64), tf.random.normal((2, 2, 64, 128))]
        self.gradient_datasets = [tf.random.normal((2, 2, 64, 128))]
        self.output_datasets = [tf.random.normal((2, 40, 64)), tf.random.normal((2, 40, 128))]
        self.outgrad_datasets = [tf.random.normal((2, 40, 64)), tf.random.normal((2, 40, 128))]

    def test_basic_setup(self):
        model = self.model
        self.assertEqual(describe(model.variables), [(2,), (64, 128)])
        self.assertEqual(describe(model.trainable_variables), [(64, 128)])
        self.assertEqual(self.get_layer_output_shapes(model), [(None, 64), (None, 128)])
        self.assertEquals(describe(self.variable_datasets), [(2, 2, 2), (2, 2, 64, 128)])
        self.assertEquals(describe(self.gradient_datasets), [(2, 2, 64, 128)])
        self.assertEquals(describe(self.output_datasets), [(2, 40, 64), (2, 40, 128)])
        self.assertEquals(describe(self.outgrad_datasets), [(2, 40, 64), (2, 40, 128)])

    # Norms, value_stats, and magnitude_stats ared calculated on sum(steps) over gradient data.
    # Based on notion that there is a single "overall" gradient for the epoch: the sum of the gradients at each step.
    def test_per_epoch_norms(self):
        cb = GradientHistoryCallback()
        model = self.model
        self.fake_train(model, variable_data=self.variable_datasets, gradient_data=self.gradient_datasets,
                        output_data=self.output_datasets, output_gradient_data=self.outgrad_datasets,
                        callbacks=[cb])

        # basic shapes
        self.assertEqual(describe(cb.model_norm_stats), (2, 5))
        self.assertEqual(describe(cb.value_norms), [None, (2,)])

        # calculate expected values
        all_gradients = self.expand_to_all_variables(model, self.gradient_datasets)
        norm_accumulator = NormAccumulatorStrategy()
        expected = self.map_each_epoch(
            all_gradients, lambda epoch_data: norm_accumulator.single(tf.reduce_sum(epoch_data, axis=0)))
        expected = [np.stack(v) if v is not None else None for v in expected]

        # asserts
        actual = cb.value_norms
        matches = np.all([close_or_none(a, b) for a, b in zip(expected, actual)])
        if not matches:
            sources = self.map_each_epoch(all_gradients, lambda epoch_data: tf.reduce_sum(epoch_data, axis=0))
            print()
            print(f"test_per_epoch_norms:")
            print(f"- sources:  {describe(sources, verbose=2)}")
            print(f"- expected: {expected}")
            print(f"- actual:   {cb.value_norms}")
        self.assertTrue(matches, "see logs for details")

    # Calculated based on sum(steps), as for norms
    def test_per_epoch_value_stats(self):
        cb = GradientHistoryCallback()
        model = self.model
        self.fake_train(model, variable_data=self.variable_datasets, gradient_data=self.gradient_datasets,
                        output_data=self.output_datasets, output_gradient_data=self.outgrad_datasets,
                        callbacks=[cb])

        # basic shapes
        self.assertEqual(describe(cb.value_stats), [None, (2, 9)])

        # calculate expected values
        expected_quantiles = [0., 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
        all_gradients = self.expand_to_all_variables(model, self.gradient_datasets)
        percentile_accumulator = PercentileAccumulatorStrategy(quantiles=expected_quantiles)
        expected = self.map_each_epoch(
            all_gradients, lambda epoch_data: percentile_accumulator.single(tf.reduce_sum(epoch_data, axis=0)))
        expected = [np.stack(v) if v is not None else None for v in expected]

        # asserts
        self.assertEqual(list(cb.value_stats[1].columns), expected_quantiles)
        actual = self.map_each_item(cb.value_stats, lambda v: v.to_numpy())
        matches = np.all([close_or_none(a, b) for a, b in zip(expected, actual)])
        if not matches:
            sources = self.map_each_epoch(all_gradients, lambda epoch_data: tf.reduce_sum(epoch_data, axis=0))
            print()
            print(f"test_per_epoch_value_stats:")
            print(f"- sources:  {describe(sources, verbose=2)}")
            print(f"- expected: {expected}")
            print(f"- actual:   {cb.value_norms}")
        self.assertTrue(matches, "see logs for details")

    # Calculated based on sum(steps), as for norms
    def test_per_epoch_magnitude_stats(self):
        cb = GradientHistoryCallback()
        model = self.model
        self.fake_train(model, variable_data=self.variable_datasets, gradient_data=self.gradient_datasets,
                        output_data=self.output_datasets, output_gradient_data=self.outgrad_datasets,
                        callbacks=[cb])

        # basic shapes
        self.assertEqual(describe(cb.model_magnitude_stats), (2, 5))
        self.assertEqual(describe(cb.magnitude_stats), [None, (2, 9)])

        # calculate expected values
        expected_quantiles = [0., 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
        all_gradients = self.expand_to_all_variables(model, self.gradient_datasets)
        percentile_accumulator = PercentileAccumulatorStrategy(quantiles=expected_quantiles, magnitudes=True)
        expected = self.map_each_epoch(
            all_gradients, lambda epoch_data: percentile_accumulator.single(tf.reduce_sum(epoch_data, axis=0)))
        expected = [np.stack(v) if v is not None else None for v in expected]

        # assert
        self.assertEqual(list(cb.magnitude_stats[1].columns), expected_quantiles)
        actual = self.map_each_item(cb.magnitude_stats, lambda v: v.to_numpy())
        matches = np.all([close_or_none(a, b) for a, b in zip(expected, actual)])
        if not matches:
            sources = self.map_each_epoch(all_gradients, lambda epoch_data: tf.reduce_sum(epoch_data, axis=0))
            print()
            print(f"test_per_epoch_magnitude_stats:")
            print(f"- sources:  {describe(sources, verbose=2)}")
            print(f"- expected: {expected}")
            print(f"- actual:   {cb.value_norms}")
        self.assertTrue(matches, "see logs for details")

    def test_per_epoch_activity_rates(self):
        cb = GradientHistoryCallback()
        model = self.model
        self.fake_train(model, variable_data=self.variable_datasets, gradient_data=self.gradient_datasets,
                        output_data=self.output_datasets, output_gradient_data=self.outgrad_datasets,
                        callbacks=[cb])

        # basic shapes
        self.assertEqual(describe(cb.activity_stats), [None, (2, 3)])

        # TODO assert on values

    def test_per_step_norms(self):
        cb = GradientHistoryCallback(per_step=True)
        model = self.model
        self.fake_train(model, variable_data=self.variable_datasets, gradient_data=self.gradient_datasets,
                        output_data=self.output_datasets, output_gradient_data=self.outgrad_datasets,
                        callbacks=[cb])

        # basic shapes
        self.assertEqual(describe(cb.model_norm_stats), (4, 5))
        self.assertEqual(describe(cb.value_norms), [None, (4,)])

        all_gradients = self.expand_to_all_variables(model, self.gradient_datasets)
        norm_accumulator = NormAccumulatorStrategy()
        expected = self.flatmap_epoch_batches(
            all_gradients, lambda epoch_data: self.map_each_variable_batch(
                epoch_data, lambda batch_data: norm_accumulator.single(batch_data)))
        expected = [np.stack(v) if v is not None else None for v in expected]

        # assert
        actual = cb.value_norms
        matches = np.all([close_or_none(a, b) for a, b in zip(expected, actual)])
        if not matches:
            sources = self.flatmap_epoch_batches(
                all_gradients, lambda epoch_data: self.map_each_variable_batch(
                    epoch_data, lambda batch_data: batch_data))
            print()
            print(f"test_per_step_norms:")
            print(f"- sources:  {describe(sources, verbose=2)}")
            print(f"- expected: {expected}")
            print(f"- actual:   {cb.value_norms}")
        self.assertTrue(matches, "see logs for details")

    def test_per_step_value_stats(self):
        cb = GradientHistoryCallback(per_step=True)
        model = self.model
        self.fake_train(model, variable_data=self.variable_datasets, gradient_data=self.gradient_datasets,
                        output_data=self.output_datasets, output_gradient_data=self.outgrad_datasets,
                        callbacks=[cb])

        # basic shapes
        self.assertEqual(describe(cb.value_stats), [None, (4, 9)])

        # calculate expected values
        expected_quantiles = [0., 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
        all_gradients = self.expand_to_all_variables(model, self.gradient_datasets)
        percentile_accumulator = PercentileAccumulatorStrategy(quantiles=expected_quantiles)
        expected = self.flatmap_epoch_batches(
            all_gradients, lambda epoch_data: self.map_each_variable_batch(
                epoch_data, lambda batch_data: percentile_accumulator.single(batch_data)))
        expected = [np.stack(v) if v is not None else None for v in expected]

        # assert
        self.assertEqual(list(cb.value_stats[1].columns), expected_quantiles)
        actual = self.map_each_item(cb.value_stats, lambda v: v.to_numpy())
        matches = np.all([close_or_none(a, b) for a, b in zip(expected, actual)])
        if not matches:
            sources = self.flatmap_epoch_batches(
                all_gradients, lambda epoch_data: self.map_each_variable_batch(
                    epoch_data, lambda batch_data: batch_data))
            print()
            print(f"test_per_step_value_stats:")
            print(f"- sources:  {describe(sources, verbose=2)}")
            print(f"- expected: {expected}")
            print(f"- actual:   {cb.value_norms}")
        self.assertTrue(matches, "see logs for details")

    def test_per_step_magnitude_stats(self):
        cb = GradientHistoryCallback(per_step=True)
        model = self.model
        self.fake_train(model, variable_data=self.variable_datasets, gradient_data=self.gradient_datasets,
                        output_data=self.output_datasets, output_gradient_data=self.outgrad_datasets,
                        callbacks=[cb])

        # basic shapes
        self.assertEqual(describe(cb.model_magnitude_stats), (4, 5))
        self.assertEqual(describe(cb.magnitude_stats), [None, (4, 9)])

        # calculate expected values
        expected_quantiles = [0., 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
        all_gradients = self.expand_to_all_variables(model, self.gradient_datasets)
        percentile_accumulator = PercentileAccumulatorStrategy(quantiles=expected_quantiles, magnitudes=True)
        expected = self.flatmap_epoch_batches(
            all_gradients, lambda epoch_data: self.map_each_variable_batch(
                epoch_data, lambda batch_data: percentile_accumulator.single(batch_data)))
        expected = [np.stack(v) if v is not None else None for v in expected]

        # assert
        self.assertEqual(list(cb.magnitude_stats[1].columns), expected_quantiles)
        actual = self.map_each_item(cb.magnitude_stats, lambda v: v.to_numpy())
        matches = np.all([close_or_none(a, b) for a, b in zip(expected, actual)])
        if not matches:
            sources = self.flatmap_epoch_batches(
                all_gradients, lambda epoch_data: self.map_each_variable_batch(
                    epoch_data, lambda batch_data: batch_data))
            print()
            print(f"test_per_step_magnitude_stats:")
            print(f"- sources:  {describe(sources, verbose=2)}")
            print(f"- expected: {expected}")
            print(f"- actual:   {cb.value_norms}")
        self.assertTrue(matches, "see logs for details")

    def test_per_step_activity_rates(self):
        cb = GradientHistoryCallback(per_step=True)
        model = self.model
        self.fake_train(model, variable_data=self.variable_datasets, gradient_data=self.gradient_datasets,
                        output_data=self.output_datasets, output_gradient_data=self.outgrad_datasets,
                        callbacks=[cb])

        # basic shapes
        self.assertEqual(describe(cb.activity_stats), [None, (4, 3)])

        # TODO assert on values


def describe(thing, verbose=1):
    """
    Helpful generic tool for turning potentially nested lists of tensors and other things into simple
    descriptions that can be nicely asserted on.
    Example output: [None, (32, 128)], given a list containing a None and a tensor of shape (32, 128).
    """
    if thing is None:
        return None
    elif hasattr(thing, 'shape'):
        if verbose == 1:
            return np.array(thing).shape
        elif verbose >= 2:
            return f"{np.array(thing).shape} in {np.min(thing)}..{np.mean(thing)}..{np.max(thing)}"
    elif isinstance(thing, list):
        return [describe(it, verbose) for it in thing]
    else:
        return thing


def close_or_none(expected_one, actual_one):
    if expected_one is None and actual_one is not None:
        return False
    elif expected_one is not None and actual_one is None:
        return False
    elif expected_one is None:
        return True
    return np.allclose(expected_one, actual_one)
