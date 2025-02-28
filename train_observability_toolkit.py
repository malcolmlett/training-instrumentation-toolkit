import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import tree
from keras.src.backend.tensorflow.trainer import TFEpochIterator
from keras.src import optimizers as optimizers_module
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
import keras
import math
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tqdm
from enum import Enum

# tip: to get output shape of a layer:
#  model.layers[l].compute_output_shape(model.layers[l].input.shape)


# Tries to replicate keras.backend.tensorflow.TensorFlowTrainer.fit() (trainer.py, keras 3.5.0)
# as much as possible.
def fit(model, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None,
        validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None,
        initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1):
    """
    A custom training loop mimicking model.fit() that makes raw gradient and layer output
    information available for tracking.

    Honours the state of `tf.config.run_functions_eagerly(bool)`.

    All args are the same as for model.fit(), with the following exceptions:
    - callbacks: additionally can be passed instances of BaseGradientCallback

    Returns:
         HistoryCallback
    """
    model._assert_compile_called("fit")
    if model.steps_per_execution != 1:
        raise ValueError(f"Only supports steps_per_execution=1, got: {model.steps_per_execution}")

    model._eval_epoch_iterator = None
    if validation_split and validation_data is None:
        # Create the validation data using the training data. Only supported
        # for TF/numpy/jax arrays.
        (x, y, sample_weight), validation_data = array_slicing.train_validation_split(
            (x, y, sample_weight), validation_split=validation_split)
    if validation_data is not None:
        (val_x, val_y, val_sample_weight) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

    # Create an iterator that yields batches for one epoch.
    epoch_iterator = TFEpochIterator(
        x=x,
        y=y,
        sample_weight=sample_weight,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        shuffle=shuffle,
        class_weight=class_weight,
        distribute_strategy=model.distribute_strategy,
        steps_per_execution=model.steps_per_execution,
    )
    model._maybe_symbolic_build(iterator=epoch_iterator)

    # prepare callbacks tracking
    gradient_callbacks = []
    if callbacks is not None and not isinstance(callbacks, tf.keras.callbacks.CallbackList):
        gradient_callbacks = [callback for callback in callbacks if reload_safe_isinstance(callback, BaseGradientCallback)]
        callbacks = [callback for callback in callbacks if not reload_safe_isinstance(callback, BaseGradientCallback)]
    if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
        callbacks = tf.keras.callbacks.CallbackList(
            callbacks, add_history=True, add_progbar=verbose != 0, verbose=verbose, epochs=epochs,
            steps=epoch_iterator.num_batches, model=model)
    for gradient_callback in gradient_callbacks:
        gradient_callback.set_params({'epochs': epochs, 'steps': epoch_iterator.num_batches})
        gradient_callback.set_model(model)

    needs_output_gradients = any(cb.needs_output_gradients for cb in gradient_callbacks)

    # prepare model for layer output collection
    # - original model output(s) will be first entry of new outputs array, it will have single tensor or list
    #   accordingly
    monitoring_model = tf.keras.Model(
        inputs=_original_inputs(model),
        outputs=[model.outputs] + [layer.output for layer in model.layers])

    # prepare train function
    if tf.config.functions_run_eagerly():
        train_step_fn = _gradient_returning_train_step
    else:
        train_step_fn = tf.function(_gradient_returning_train_step)

    # latest values
    # - these are computed at each step, but we need to provide values at the end of each epoch,
    #   so we'll just hold onto the last value computed at each step
    training_logs = None
    logs = {}
    trainable_gradients = None
    output_gradients = None
    activations = None

    # train begin
    model.stop_training = False
    callbacks.on_train_begin()
    for gradient_callback in gradient_callbacks:
        gradient_callback.on_train_begin()

    # training loop by epoch
    initial_epoch = model._initial_epoch or initial_epoch
    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        for gradient_callback in gradient_callbacks:
            gradient_callback.on_epoch_begin(epoch)

        # training loop by step
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator.enumerate_epoch():
                callbacks.on_train_batch_begin(step)
                for gradient_callback in gradient_callbacks:
                    gradient_callback.on_train_batch_begin(step)

                x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(next(iterator))
                logs, trainable_gradients, output_gradients, activations = train_step_fn(
                    model, monitoring_model, x, y, sample_weight, needs_output_gradients)

                callbacks.on_train_batch_end(step, logs)
                for gradient_callback in gradient_callbacks:
                    gradient_callback.on_train_batch_end(
                        batch=step,
                        loss=logs['loss'],
                        gradients=trainable_gradients,
                        trainable_variables=model.trainable_variables,
                        activations=activations,
                        output_gradients=output_gradients if gradient_callback.needs_output_gradients else None)

                if model.stop_training:
                    break

        # Override with model metrics instead of last step logs if needed.
        epoch_logs = dict(model._get_metrics_result_or_logs(logs))

        # Run validation.
        if validation_data is not None and model._should_eval(epoch, validation_freq):
            # Create EpochIterator for evaluation and cache it.
            if getattr(model, "_eval_epoch_iterator", None) is None:
                model._eval_epoch_iterator = TFEpochIterator(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    distribute_strategy=model.distribute_strategy,
                    steps_per_execution=model.steps_per_execution,
                    steps_per_epoch=validation_steps,
                    shuffle=False,
                )
            val_logs = model.evaluate(
                x=val_x,
                y=val_y,
                sample_weight=val_sample_weight,
                batch_size=validation_batch_size or batch_size,
                steps=validation_steps,
                callbacks=callbacks,
                return_dict=True,
                _use_cached_eval_dataset=True,
            )
            val_logs = {"val_" + name: val for name, val in val_logs.items()}
            epoch_logs.update(val_logs)

        # end of epoch
        training_logs = epoch_logs
        callbacks.on_epoch_end(epoch, epoch_logs)  # should be passing loss and mse
        for gradient_callback in gradient_callbacks:
            gradient_callback.on_epoch_end(
                epoch=epoch,
                loss=epoch_logs['loss'],
                gradients=trainable_gradients,
                trainable_variables=model.trainable_variables,
                activations=activations,
                output_gradients=output_gradients if gradient_callback.needs_output_gradients else None)
        if model.stop_training:
            break

    if (isinstance(model.optimizer, optimizers_module.Optimizer) and epochs > 0):
        model.optimizer.finalize_variable_values(model.trainable_weights)

    # train end
    # If _eval_epoch_iterator exists, delete it after all epochs are done.
    if getattr(model, "_eval_epoch_iterator", None) is not None:
        del model._eval_epoch_iterator
    callbacks.on_train_end(training_logs)
    for gradient_callback in gradient_callbacks:
        gradient_callback.on_train_end()

    return model.history


# Tries to replicate keras.backend.tensorflow.TensorFlowTrainer.train_step() (trainer.py, keras 3.5.0)
# as much as possible.
def _gradient_returning_train_step(model, monitoring_model, x, y, sample_weight, compute_output_gradients):
    # Forward pass
    with tf.GradientTape() as tape:
        monitoring_outputs = monitoring_model(x, training=True)
        y_pred = monitoring_outputs[0]
        layer_outputs = monitoring_outputs[1:]

        loss = model.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=True)
        model._loss_tracker.update_state(loss, sample_weight=tf.shape(tree.flatten(x)[0])[0])
        loss = model.optimizer.scale_loss(loss)

    # Backward pass
    if compute_output_gradients:
        gradients = tape.gradient(loss, model.trainable_weights + layer_outputs)
        trainable_grads = gradients[:len(model.trainable_variables)]
        output_grads = gradients[len(model.trainable_variables):]
    else:
        trainable_grads = tape.gradient(loss, model.trainable_variables)
        output_grads = None
    model.optimizer.apply_gradients(zip(trainable_grads, model.trainable_variables))

    # Metrics
    metrics = model.compute_metrics(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)

    return metrics, trainable_grads, output_grads, layer_outputs


def _original_inputs(model):
    """
    Applies some heuristics to identify the inputs for the model in their originally supplied structure.
    This isn't straightforward because model.inputs gets normalized into a list, disregarding whether
    the original input structure was a single tensor, a list of tensors, a dict, or some other nested structure.
    This matters because the actual input batches we use during training are from the end user and they will
    match the input structure when originally created.
    """
    # extract Functional API from Sequence or otherwise
    functional = None
    if isinstance(model, keras.src.Functional):
        functional = model
    elif hasattr(model, '_functional'):
        functional = model._functional

    # extract original input structure from Functional
    inputs = None
    if functional is not None and hasattr(functional, '_inputs_struct'):
        inputs = functional._inputs_struct

    # fallback
    if inputs is None:
        inputs = model.inputs

    return inputs


class LessVerboseProgressLogger(tf.keras.callbacks.Callback):
    """
    Progress logger for when running models that train via thousands of very short epochs.
    By default, automatically logs progress 10 times during training.

    Use as:
    ```python
    model.fit(...., verbose=0, callbacks=[LessVerboseProgressLogger()])
    ```
    """

    def __init__(self, display_interval=None, display_total=10):
        super().__init__()
        self.display_interval = display_interval
        self.display_total = display_total
        self.epoch_count = None
        self.group_start_time = None
        self.group_start_epoch = None
        self.epoch_start = None

    def set_params(self, params):
        self.epoch_count = params['epochs']
        self.group_start_epoch = -1
        self.group_start_time = tf.timestamp()
        if self.display_interval is None:
            self.display_interval = math.floor(self.epoch_count / self.display_total)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        if self.display_interval == 0 or ((epoch + 1) % self.display_interval == 0) or epoch == self.epoch_count - 1:
            now = tf.timestamp()
            group_dur = now - self.group_start_time
            rate = group_dur / (epoch - self.group_start_epoch)
            self.group_start_time = now
            self.group_start_epoch = epoch

            print(f'Epoch {epoch + 1:5d} - {self._format_duration(rate)}/epoch:', end=' ')
            for key, value in logs.items():
                print(f'{key}: {value:.4f}', end='  ')
            print()

    @staticmethod
    def _format_duration(seconds):
        if seconds < 1e-3:
            return f"{seconds * 1e6:.2f}µs"
        elif seconds < 1:
            return f"{seconds * 1e3:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            mins, secs = divmod(seconds, 60)
            return f"{int(mins)}m {secs:.2f}s"


class HistoryStats(tf.keras.callbacks.History):
    """
    Extended version of the History callback that collects stats of the loss and metric values
    over the course of each epoch, rather than only recording the value of the loss at the end
    the epoch. This allows for more quickly identifying training problems.

    Other properties:
        - steps: list of step numbers corresponding to results from step_history,
            or None if per_step is False
        - step_history: like `history` but added to each step,
            or None if per_step is False

    Example:
    ```python
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    model.compile(tf.keras.optimizers.SGD(), loss='mse')
    history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                 epochs=10, verbose=1, callbacks=[HistoryStats()])
    print(history.params)
    # out: {'verbose': 1, 'epochs': 10, 'steps': 1}
    print(history.history.keys())
    # out: dict_keys(['loss'])
    print(history.stats.keys())
    # out: dict_keys(['loss'])
    ```
    """

    def __init__(self, per_step=False, quantiles=None):
        """
        Args:
            per_step: bool.
                Whether to additionally keep raw metrics on a per_step basis.
            quantiles: list of int/float.
                List of quantiles to collect percentile data for, in range 0 .. 100.
                Default: [0, 25, 50, 75, 100]
        """
        super().__init__()
        self.per_step = per_step
        self.quantiles = quantiles or [0, 25, 50, 75, 100]
        if per_step:
            self.steps = []
            self.step_history = {}
        else:
            self.steps = None
            self.step_history = None

        # internal
        self._epoch_start_step = 0
        self._raw_epoch_stats = {}
        self._converted_epoch_stats = None
        self._this_epoch_history = None

    @property
    def epoch_stats(self):
        """
        Mirrors the built-in `history` property, but contains a dict of pandas dataframes
        containing percentile data of the loss and metrics each epoch.
        """
        # convert and cache
        # - from: dict (by loss/metric) of list (by epoch) of TF tensors (by percentile)
        # - to: dict (by loss/metric) of dataframes with shape (epochs, percentiles)
        if self._converted_epoch_stats is None:
            self._converted_epoch_stats = {}
            for k, tensor_list in self._raw_epoch_stats.items():
                table = np.stack(tensor_list, axis=0)
                self._converted_epoch_stats[k] = pd.DataFrame(table, columns=self.quantiles)
        return self._converted_epoch_stats

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self._this_epoch_history = {}
        if self.steps is not None:
            self._epoch_start_step = (self.steps[-1] + 1) if len(self.steps) > 0 else 0

    def on_train_batch_end(self, step, logs=None):
        super().on_train_batch_end(step, logs)
        if self.steps is not None:
            # step starts from zero each epoch, so add to existing
            self.steps.append(self._epoch_start_step + step)
        logs = logs or {}
        for k, v in logs.items():
            self._this_epoch_history.setdefault(k, []).append(v)
            if self.step_history is not None:
                self.step_history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        # compute stats over the epoch
        for k, values in self._this_epoch_history.items():
            percentiles = tfp.stats.percentile(values, self.quantiles)
            self._raw_epoch_stats.setdefault(k, []).append(percentiles)
            self._converted_epoch_stats = None  # invalidate cache


class BaseGradientCallback:
    """
    Supply a subclass instance to the custom fit() method in order to collect gradient and layer activation
    information.
    This implementation does nothing, and is suitable for use as a no-op.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = None
        self._model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    @property
    def needs_output_gradients(self):
        """
        Indicates whether this callback wants to receive "layer output gradients" in its callbacks.
        These require extra computation, so they are only supplied to callbacks that request them.

        Default: False.
        """
        return False

    def on_train_begin(self):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.
        """

    def on_train_end(self):
        """Called at the end of training.

        Subclasses should override for any actions to run.
        """

    def on_epoch_begin(self, epoch):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run.

        Args:
            epoch: Integer, index of epoch.
        """

    def on_epoch_end(self, epoch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Called at the end of an epoch during training.
        Subclasses should override for any actions to run.

        Supplied with training parameters from the last batch of the epoch. Can be a convenient alternative where
        training doesn't use batches, or when you only want to sample the occasional update. However, be aware
        that because it is supplied values from only the last batch, the results can be quite skewed and noisy.
        You will often need to accumulate values over the epoch and compute means.

        The first dimension of activations and output_gradients is `batch_size`, which can vary during training.
        In particular, the last batch of an epoch can be less than the usual size.

        Args:
            epoch: Integer, index of epoch.
            loss: float. The loss value of the batch.
            gradients: list of gradients for each trainable variable.
            trainable_variables: list of trainable variables.
            activations: activation outputs from each layer.
            output_gradients: list of gradients w.r.t. to the outputs of each layer, or None if not requested.
                Always same size as the number of layers, but some gradients may be None (last layer, and any layers
                that don't contribute to the loss).
        """

    def on_train_batch_begin(self, batch):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in `Model` is set to `N`, this method will only
        be called every `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
        """

    def on_train_batch_end(self, batch, loss, gradients, trainable_variables, activations, output_gradients):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in `Model` is set to `N`, this method will
        only be called every `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            loss: float. The loss value of the batch.
            gradients: list of gradients for each trainable variable.
            trainable_variables: list of trainable variables.
            activations: activation outputs from each layer.
            output_gradients: list of gradients w.r.t. to the outputs of each layer, or None if not requested.
                Always same size as the number of layers, but some gradients may be None (last layer, and any layers
                that don't contribute to the loss).
        """


class ValueStatsCollectingMixin:
    """
    Mixin for callback classes the need to measure the value statistics over the items that they capture.
    "Items" are typically something related to layers or variables.
    """

    def __init__(self, value_norms=True, value_stats=True, value_stats_quantiles=None, *args, **kwargs):
        """
        Args:
            value_norms: bool.
                Whether to enable collection of value norms.
            value_stats: bool.
                Whether to enable collection of value stats.
            value_stats_quantiles: list of percentiles in range 0 .. 100.
                Default: [0., 12.5, 25., 37.5, 50., 62.5, 75., 87.5, 100.]
        """
        super().__init__(*args, **kwargs)
        self.value_norms_enabled = value_norms
        self.value_stats_enabled = value_stats
        self.value_stats_quantiles = value_stats_quantiles or [0., 12.5, 25., 37.5, 50., 62.5, 75., 87.5, 100.]
        self._collected_value_indices = None

        # - initially: list (by item) of list (by iteration) of scalar norms of gradients
        # - finally:   list (by item) of np array with shape (iterations,)
        self._value_norms = None

        # shape of data structure is optimised for speed of accumulation during training
        # so needs to be converted to a more usable data structure after training
        # - initially: list (by item) of list (by iteration) of tensors (by quantile)
        # - finally:   list (by item) of pandas dataframes with shape (iterations, stats)
        self._value_stats = None
        self._magnitude_stats = None

    @property
    def model_norm_stats(self):
        """
        Stats across the norms of values across all measured variables/layers in the model.
        Pandas data-frame with shape (iterations, percentiles).
        """
        if self._value_norms is not None:
            scales = np.stack(self.collected_value_norms, axis=-1)
            return _compute_model_summary_stats(scales)
        else:
            return None

    @property
    def model_magnitude_stats(self):
        """
        Stats across the general magnitudes of values across all measured variables/layers in the model.
        Pandas data-frame with shape (iterations, percentiles).
        """
        if self._magnitude_stats is not None:
            scales = get_scales_across_stats_list(self._magnitude_stats, 50)
            return _compute_model_summary_stats(scales)
        else:
            return None

    @property
    def value_norms(self):
        """
        List (by variable/layer) of array containing the norms of each variable at each iteration,
        or None if not enabled. Each list entry is either a numpy array of shape (iterations,), or
        None if data is not available for that variable/layer.
        Norms are calculated as the frobenius (aka euclidean) norm, and then divided by `sqrt(size)` so that tensors
        of different sizes have the same scale. Mathematically equivalent to the RMS of the individual values.
        """
        return self._value_norms

    @property
    def value_stats(self):
        """
        List (by variable/layer) of dataframes containing value stats, or None if not enabled.
        Each list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that variable/layer.
        """
        return self._value_stats

    @property
    def magnitude_stats(self):
        """
        List (by variable/layer) of dataframes containing value magnitude stats, or None if not enabled.
        Each list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that variable/layer.
        """
        return self._magnitude_stats

    @property
    def collected_value_norms(self):
        """
        Value norms filtered only on those that have been collected.
        None if value norms collection is disabled.
        Use collected_value_stats_indices() to identify which model variables/layers are included.
        """
        if self._value_norms is not None:
            return [norms for norms in self._value_norms if norms is not None]
        else:
            return None

    @property
    def collected_value_stats(self):
        """
        Value stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_value_stats_indices() to identify which model variables/layers are included.
        """
        if self._value_stats is not None:
            return [stats for stats in self._value_stats if stats is not None]
        else:
            return None

    @property
    def collected_magnitude_stats(self):
        """
        Magnitude stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_value_stats_indices() to identify which model variables/layers are included.
        """
        if self._magnitude_stats is not None:
            return [stats for stats in self._magnitude_stats if stats is not None]
        else:
            return None

    @property
    def collected_value_stats_indices(self):
        """
        Indices of variables/layers for which value and magnitude stats are returned.
        Indices are as per the variable's position in model.variables, or layer's position
        in model.layers, whichever is applicable for the callback.
        """
        return self._collected_value_indices

    def _init_value_stats(self, values):
        """
        Initialisation of tracking once we have a first example of what the values look like.
        Automatically called on first data collection.
        Safe to call this every iteration. Only takes effect on the first call.
        """
        if self.value_norms_enabled and self._value_norms is None:
            self._value_norms = [[] if value is not None else None for value in values]
        if self.value_stats_enabled and self._value_stats is None:
            self._value_stats = [[] if value is not None else None for value in values]
            self._magnitude_stats = [[] if value is not None else None for value in values]
            self._collected_value_indices = [i_idx for i_idx, value in enumerate(values) if value is not None]

    def _finalize_value_stats(self):
        # convert data structures
        # - from: list (by item) of list (by iteration) of tensor (by stat)
        # - to:   list (by item) of pd-dataframe: iterations x percentiles
        if self._value_norms is not None:
            self._value_norms = [np.array(item_norms) if item_norms is not None else None
                                 for item_norms in self._value_norms]
        if self._value_stats is not None:
            self._value_stats = self._stats_tensor_list_to_dataframes(
                self._value_stats, self.value_stats_quantiles)
            self._magnitude_stats = self._stats_tensor_list_to_dataframes(
                self._magnitude_stats, self.value_stats_quantiles)

    def _collect_value_stats(self, values):
        self._init_value_stats(values)
        if self._value_stats is not None:
            # compute value and magnitude percentile stats for each individual variable
            # - returns tuples (norm, magnitude_percentiles, value_percentiles)
            stat_tuples = self._compute_iteration_value_stats(values, self.value_stats_quantiles)

            # append to stats list
            # (performance note: this loop doesn't seem to cost much)
            for item_value_norms, item_value_stats, item_magnitude_stats,\
                    (norm, value_percentiles, magnitude_percentiles) \
                    in zip(self._value_norms, self._value_stats, self._magnitude_stats, stat_tuples):
                if item_value_norms is not None:
                    item_value_norms.append(norm)
                if item_value_stats is not None:
                    item_value_stats.append(value_percentiles)
                if item_magnitude_stats is not None:
                    item_magnitude_stats.append(magnitude_percentiles)

    # Percentile calculation is fairly expensive. It could be worth defaulting to calculation of simpler stats and only
    # doing percentiles if requested. However, simple mean + stddev isn't very good for heavily skewed distributions
    # like gradient magnitudes.
    @tf.function
    def _compute_iteration_value_stats(self, tensors, quantiles):
        """
        Computes a set of stats across the raw values and magnitudes of each provided tensor.
        For tensors that commonly have values distributed either side of zero, the median will
        typically be around zero, and the 25th and 75th percentiles represent the respective medians
        in the positive and negative halves.

        Norms are calculated as a size-normalized euclidean norm. This makes for easy comparison across layers.
        If not size normalised then the scale would be proportional to the sqrt of its total number of elements
        (often in the thousands). This is mathematically equivalent to the root-mean-square of the tensor values.
        It also happens that computing RMS is more efficient in TF than using tf.norm() (particularly on GPU).
        So we use that to calculate the norm.

        Args:
            tensors: list of tensors for which percentiles should be calculated, some of which may be None.
            quantiles: list of quantiles to compute values for
        Returns:
            - norm - scalar norm
            - value_percentiles - tensor of percentile values across the tensor values, None for None tensors
            - magnitude_percentile - tensor of percentile values across the tensor magnitudes, None for None tensors
        """
        def computation(tensor):
            if tensor is not None:
                norm = tf.sqrt(tf.reduce_mean(tf.square(tensor)))
                value_percentiles = tfp.stats.percentile(tensor, quantiles, interpolation='linear')
                magnitude_percentiles = tfp.stats.percentile(tf.abs(tensor), quantiles, interpolation='linear')
            else:
                norm, value_percentiles, magnitude_percentiles = None, None, None
            return norm, value_percentiles, magnitude_percentiles
        return [computation(tensor) for tensor in tensors]

    @staticmethod
    def _stats_tensor_list_to_dataframes(stats_by_item, columns):
        """
        Args:
            stats_by_item: list (by item) of list (by iteration) of TF tensor (by stat)
        Returns:
            list (by item) of pandas dataframe (iterations x stats)
        """
        item_dataframes = []
        for item_stats in stats_by_item:
            if item_stats is not None:
                item_data = [iteration_stats.numpy() for iteration_stats in item_stats]
                df = pd.DataFrame(item_data, columns=columns)
            else:
                df = None
            item_dataframes.append(df)
        return item_dataframes


class ActivityStatsCollectingMixin:
    """
    Mixin for callback classes the need to measure the activity rates of the items that they capture.
    "Items" are typically something related to layers or variables.

    Assumes the following for the calculation of "activation rates" across layers:
    * All layers are assumed to produce outputs with shapes of form: `(batch_size, ..spatial_dims.., channels)`.
    * We focus on the channels dimension as representing a set of output neurons, or "units",
      each producing a single float output value. The output value may vary across batch and spatial dimensions.

    Assumes the following for the calculation of "activation rates" across variables:
    * All variables are assumed to have shapes of form: `(..spatial_dims.., channels)`
    * Then treat these the same as for layer outputs, but without the batch dimension.

    Then assumes and computes:
    * A unit or channel is "active" if it has any non-zero value.
    * A unit or channel is "dead" if it always has zero values across batch and spatial dimensions.
    * The activation rate for a unit is the fraction of batch/spatial positions for which that unit is active.
    * The activation rate for the whole layer is the mean activation rate across all units.
      Alternatively, this can be seen as the fraction of active outputs across all batch, spatial, and channel
      dimensions.
    * The "dead rate" is the fraction of units that are dead.
    * The "spatial dead rate" uses the concept, but measures across discrete spatial positions.
        It identifies the fraction of spatial positions that are always dead across the batch and channel dims.

    """
    def __init__(self, activity_stats=True, data_format="BSC", *args, **kwargs):
        """
        Args:
            activity_stats: bool.
                Whether to enable collection of activity stats.

            data_format: string.
                Indicates the structure of the data being analysed. This depends on
                whether it's layer or variable data. Ideally we'd also support the standard
                data_format variations supported by TF, but not yet.
                Allowed values:
                    'BSC' - indicates data values will have shape `(batches, ..spatial_dims.., channels)`
                    'SC' - indicates data values will have shape `(..spatial_dims.., channels)`
        """
        super().__init__(*args, **kwargs)
        if data_format not in ('BSC', 'SC'):
            raise ValueError(f"data_format '{data_format}' is not one of 'BSC' or 'SC'")

        self.activity_stats_enabled = activity_stats
        self.activity_data_format = data_format

        # shape of data structure is optimised for speed of accumulation during training
        # so needs to be converted to a more usable data structure after training
        # - initially: list (by iteration) of list (by item) of tuples (by stat)
        # - finally:   list (by item) of pandas dataframes with shape (iterations, stats)
        self._activity_stats = None
        self._model_activity_stats = None

        # the following gets initialised only within _init_activity_stats()
        self._collected_activity_indices = None  # variable or layer indices of collected data
        self._item_shapes = None  # shapes for each item
        self._channel_sizes = None  # for each item: number of channels
        self._spatial_shapes = None  # for each item: subset of shape relating just to spatial dims
        self._channel_activity_sums = None  # tf.Variable accumulators for each item
        self._spatial_activity_sums = None  # tf.Variable accumulators for each item

    @property
    def model_activity_stats(self):
        """
        Model activity stats, or None if not enabled.
        A pandas dataframe of shape (iterations, stats), with the following
        stats:
            - min_activation_rate - min value of activation_rate across all variables/layers
            - mean_activation_rate - mean value of activation_rate across all variables/layers
            - max_activation_rate - max value of activation_rate across all variables/layers
            - min_dead_rate - min value of dead_rate across all variables/layers
            - mean_dead_rate - min value of dead_rate across all variables/layers
            - max_dead_rate - min value of dead_rate across all variables/layers
            - min_spatial_dead_rate - min value of spatial_dead_rate across all variables/layers
            - mean_spatial_dead_rate - min value of spatial_dead_rate across all variables/layers
            - max_spatial_dead_rate - max value of spatial_dead_rate across all variables/layers
        """
        return self._model_activity_stats

    @property
    def activity_stats(self):
        """
        List (by variable/layer) of dataframes containing activity stats, or None if not enabled.
        Each item's list entry is either a pandas dataframe of shape (iterations, stats), with the following
        stats, or None if stats are not collected for that variable/layer:
            - activation_rate - fraction of non-zero values
            - dead_rate - fraction of output channels that are all zero across all other dims
            - spatial_dead_rate - fraction across other dims where all channels are zero
        """
        return self._activity_stats

    @property
    def collected_activity_stats(self):
        """
        Activity stats filtered only on those that have been collected.
        None if activity collection is disabled.
        Use collected_activity_stats_indices() to identify which model variables/layers are included.
        """
        if self._activity_stats is not None:
            return [stats for stats in self._activity_stats if stats is not None]
        else:
            return None

    @property
    def collected_activity_stats_indices(self):
        """
        Indices of variables/layers for which activity stats are returned.
        Indices are as per the variable's position in model.variables, or layer's position within model.layers,
        whichever is applicable for the callback.
        """
        return self._collected_activity_indices

    def _init_activity_stats(self, values):
        """
        Final initialisation of tracking that has to be deferred until we have our first activation data.
        There seems to be some dynamism in determining the output shape of a Layer object, and we need that
        accurate for this final initialisation.
        Automatically called on first data collection.
        Safe to call this every iteration. Only takes effect on the first call.
        Args:
            values: list of tensors of raw tracked values, some of which may be None
        """
        if self.activity_stats_enabled and self._activity_stats is None:
            self._collected_activity_indices = [i_idx for i_idx, value in enumerate(values) if value is not None]
            self._activity_stats = []

            # TODO either expose some of these as properties or don't save in fields
            has_batch_dim = 'B' in self.activity_data_format
            self._item_shapes = [tensor.shape if tensor is not None else None for tensor in values]
            self._channel_sizes = [tensor.shape[-1] if tensor is not None else None
                                   for tensor in values]
            if has_batch_dim:
                self._spatial_shapes = [tensor.shape[1:-1] if tensor is not None and len(tensor.shape) > 2
                                        else () if tensor is not None else None
                                        for tensor in values]
            else:
                self._spatial_shapes = [tensor.shape[:-1] if tensor is not None and len(tensor.shape) > 1
                                        else () if tensor is not None else None
                                        for tensor in values]

            self._channel_activity_sums = [tf.Variable(tf.zeros(size, dtype=tf.float32))
                                           if size is not None else None
                                           for size in self._channel_sizes]  # by channel
            self._spatial_activity_sums = [tf.Variable(tf.zeros(shape, dtype=tf.float32))
                                           if shape is not None else None
                                           for shape in self._spatial_shapes]  # by spatial pos

    def _finalize_activity_stats(self):
        """
        To be called once training is complete. Does data format conversions.
        """
        # TODO to support live-monitoring, consider doing on-demand in property,
        #  saving to a different variable, and invalidating if more data is added

        # convert per-item stats
        if self._activity_stats is not None:
            def convert(i_idx, columns):
                if self._activity_stats[0][i_idx] is not None:
                    item_data = [[stat.numpy() for stat in iteration_stats[i_idx]]
                                 for iteration_stats in self._activity_stats]
                    return pd.DataFrame(item_data, columns=columns)
                return None
            num_items = len(self._activity_stats[0])
            cols = self._activity_stat_keys()
            self._activity_stats = [convert(i_idx, cols) for i_idx in range(num_items)]

        # compute model-whole stats
        if self._activity_stats is not None:
            dic = {}
            for key in self._activity_stat_keys():
                key_stats = [item_stats[key].to_numpy() for item_stats in self._activity_stats
                             if item_stats is not None]
                dic[f"min_{key}"] = np.min(key_stats, axis=0)
                dic[f"max_{key}"] = np.max(key_stats, axis=0)
                dic[f"mean_{key}"] = np.mean(key_stats, axis=0)
            self._model_activity_stats = pd.DataFrame(dic)

    def _accum_activity_stats(self, values, is_accum):
        """
        Called each iteration and accumulates stats into TF Variables.
        For example, very much needed when collecting stats across epochs as some
        spatial/channel positions are only active for some batches.
        """
        self._init_activity_stats(values)
        if self._activity_stats is not None:
            self._accum_activity_stats_internal(
                values, is_accum,
                self._channel_activity_sums,
                self._spatial_activity_sums)

    def _reset_per_epoch_activity_stats(self):
        """
        Resets accumulators for collection of data per-epoch.
        """
        if self._activity_stats is not None:
            for activity_sum in self._channel_activity_sums:
                if activity_sum is not None:
                    activity_sum.assign(tf.zeros_like(activity_sum))
            for activity_sum in self._spatial_activity_sums:
                if activity_sum is not None:
                    activity_sum.assign(tf.zeros_like(activity_sum))

    # TODO to support live-monitoring, should also compute and append model stats
    def _collect_activity_stats(self, num_batches):
        """
        Args:
            num_batches - number of batches that have gone into the accumulated activity sums
                (usually 1 for per-step collection, and steps_per_epoch for per-epoch collection)
        """
        if self._activity_stats is not None:
            iteration_activity_stats = self._compute_activity_stats(
                self._channel_activity_sums,
                self._spatial_activity_sums,
                num_batches)
            self._activity_stats.append(iteration_activity_stats)

    @staticmethod
    def _activity_stat_keys():
        """
        Gets the list of stats that will be computed.
        Currently static but may be computed based on configuration in the future.
        """
        return ['activation_rate', 'dead_rate', 'spatial_dead_rate']

    @tf.function
    def _accum_activity_stats_internal(self, values, is_accum, outs_by_channel, outs_by_spatial):
        """
        Auto-graphed function that accumulates partially computed stats each iteration, in preparation
        for calculating target stats either each step or each epoch.
        Args:
            values: list of tensors, some of which may be None
        """
        has_batch_dim = 'B' in self.activity_data_format
        for i_idx, tensor in enumerate(values):
            if tensor is not None:
                active_mask = tf.cast(tf.not_equal(tensor, 0.0), tf.float32)
                rate_by_channel = tf.reduce_mean(active_mask, axis=tf.range(tf.rank(active_mask) - 1))
                if has_batch_dim:
                    rate_by_spatial = tf.reduce_mean(active_mask, axis=(0, -1))
                else:
                    rate_by_spatial = tf.reduce_mean(active_mask, axis=-1)
                if is_accum:
                    outs_by_channel[i_idx].assign_add(rate_by_channel)
                    outs_by_spatial[i_idx].assign_add(rate_by_spatial)
                else:
                    outs_by_channel[i_idx].assign(rate_by_channel)
                    outs_by_spatial[i_idx].assign(rate_by_spatial)

    @tf.function
    def _compute_activity_stats(self, channel_activity_sums, spatial_activity_sums, num_batches):
        """
        Auto-graphed function that calculates the set of activity stats for a single iteration.
        Returns a list of tuples, one for each item, with Nones for those without data.
        """
        def computation(channel_activity_sum, spatial_activity_sum):
            # item-wide activation rate
            # - mean rate of activation (non-zero) across batches, spatial dims, and channels
            # - already taken mean across other dims, so take final mean across channels
            # - note: have been summing over each step, so must divide by that
            active_rate = tf.reduce_mean(channel_activity_sum / num_batches)

            # per-channel dead rate
            # - fraction of channels that are always zero
            dead_channel_rate = tf.reduce_mean(tf.cast(tf.equal(channel_activity_sum, 0.0), tf.float32))

            # per-spatial dead rate
            # - fraction of spatial positions that are always zero
            dead_spatial_rate = tf.reduce_mean(tf.cast(tf.equal(spatial_activity_sum, 0.0), tf.float32))

            # return in same order as per _activity_stat_keys()
            return active_rate, dead_channel_rate, dead_spatial_rate

        return [computation(channel_activity_sum, spatial_activity_sum)
                if channel_activity_sum is not None else None
                for channel_activity_sum, spatial_activity_sum
                in zip(channel_activity_sums, spatial_activity_sums)]


class PerEpochAccumulatorStrategy:
    """
    Helps with efficiently summing up gradients etc. each update step within an epoch, so that results
    can be computed on the sum or mean values at the end the epoch.
    Tip: instances may consume memory that isn't needed after training, including GCP memory.
    Instances of this class can usually be discarded automatically at the end of training.
    """

    def __init__(self, batch_reduction=False, keep_dims=False):
        """
        Args:
            batch_reduction: bool.
                Whether to reduce along the batch dimension (always assumed to be the first dimension).
                Needed if a batch dimension is present and varies by step - which is often the case
                for the last batch of each epoch.
            keep_dims: bool.
                Whether to keep the batch dimension when applying batch_reduction.
        """
        super().__init__()
        self.batch_reduction = batch_reduction
        self.keep_dims = keep_dims
        self._accumulators = None
        self._count = None

    @property
    def sum(self):
        """
        An immutable tensor containing the current sums
        """
        return [tf.identity(a) if a is not None else None for a in self._accumulators]

    @property
    def mean(self):
        """
        An immutable tensor containing the current means
        """
        return [a / tf.cast(self._count, dtype=a.dtype) if a is not None else None for a in self._accumulators]

    def accumulate(self, batch: int, tensors: list):
        """
        Args:
            batch - batch index within epoch; used to determine whether this is the first in the epoch
            tensors - the tensors to add to the accumulator
        """
        # init
        if self._accumulators is None:
            if batch > 0:
                raise RuntimeError("Second or subsequent tensor received but not initialised")
            self._accumulators = [tf.Variable(self._transform_batch_dim(t), dtype=t.dtype) if t is not None else None
                                  for t in tensors]
            self._count = self._get_divisor(tensors)
        elif batch == 0:
            self._reset(self._accumulators, tensors)
            self._count = self._get_divisor(tensors)
        else:
            self._add(self._accumulators, tensors)
            self._count += self._get_divisor(tensors)

    @tf.function
    def _reset(self, accumulators, values):
        for accumulator, value in zip(accumulators, values):
            if value is not None:
                accumulator.assign(self._transform_batch_dim(value))

    @tf.function
    def _add(self, accumulators, values):
        for accumulator, value in zip(accumulators, values):
            if value is not None:
                accumulator.assign_add(self._transform_batch_dim(value))

    def _transform_batch_dim(self, tensor):
        if self.batch_reduction:
            return tf.reduce_sum(tensor, axis=0, keepdims=self.keep_dims)
        else:
            return tensor
        
    def _get_divisor(self, tensors):
        """
        Computes the divisor component for this iteration to be used when computing the global mean.
        If reducing the batch dimension, then returns the batch size for the given batch, otherwise returns 1.
        Note that the batch size can vary for the last batch in an epoch.
        """
        if self.batch_reduction:
            for tensor in tensors:
                if tensor is not None:
                    return tf.shape(tensor)[0]
        return 1


class VariableHistoryCallback(tf.keras.callbacks.Callback, ValueStatsCollectingMixin, ActivityStatsCollectingMixin):
    """
    Standard model.fit() callback that collects various statistics and/or raw values of
    the model variables during training.
    Variable states may be captured BEFORE or AFTER each update step or epoch, depending on the needs.

    Other properties:
        model: the model captured
        epochs: list of int. Epoch numbers correlated to captured gradients/gradient-stats
            (only available if per_step is False)
        steps: list of int. Step numbers correlated to captured gradients/gradient-stats
            (only available if per_step is True)
    """

    def __init__(self, per_step=False, before_updates=False, trainable_only=True, value_stats_quantiles=None,
                 value_stats=True, activity_stats=True, collection_sets=None):
        """
        Args:
            per_step: bool. Whether to collect per-step stats and raw values, or per-epoch otherwise.
                By default, data is accumulated per-epoch, and an 'epochs' list is available
                that tracks the display indices of each sample.
                If per-step is set, then a `steps` list is available instead, and activity
                is collected on each update step.
                The same applies to layer output capture if enabled.
            before_updates: bool. Whether to sample variables BEFORE they are updated, or AFTER otherwise.
                If False, this reflects the notion of capturing the weights and biases as a RESULT of each update step.
                If True, this reflects the notion of capturing the weights and biases that were used DURING
                the update step.
            value_norms: bool, default: True.
                Whether to collect the norms of values.
            value_stats: bool, default: True.
                Whether to collect value and magnitude stats.
            activity_stats: bool, default: True.
                Whether to collect activity stats.
            trainable_only: bool, default: True.
                Whether to only include stats for trainable variables, or all variables otherwise.
            value_stats_quantiles: list of percentiles to collect stats for, in range 0 .. 100.
                Default: [0., 12.5, 25., 37.5, 50., 62.5, 75., 87.5, 100.]
            collection_sets: list of dicts. Enables collection of raw layer outputs
                and provides fine-grained control over which layer outputs are collected.
                If omitted, this callback collects only stats.
                See _normalize_collection_sets_for_variables() for format details.
        """
        # Callback doesn't honour python 3's MRO, but the mixins do, do call first mixin directly and
        # it'll call all the rest. Also, init base Callback class via python 2 syntax to prevent regressions
        # if Callback updated to honour MRO.
        ValueStatsCollectingMixin.__init__(self, value_stats=value_stats, value_stats_quantiles=value_stats_quantiles,
                                           data_format='SC', activity_stats=activity_stats)
        tf.keras.callbacks.Callback.__init__(self)
        self.per_step = per_step
        self.before_updates = before_updates
        self.trainable_only = trainable_only
        self.collection_sets = collection_sets

        # results variable creation
        if per_step:
            self.steps = []
        else:
            self.epochs = []
        self._variable_values = None

        # internal tracking
        self._epoch = 0
        self._variable_stats_mask = None
        self._filtered_value_variable_indices = None

    @property
    def item_name(self):
        """
        Used in plot functions.
        """
        return "Variable"

    @property
    def item_type(self):
        """
        Used in plot functions.
        """
        return ItemType.VARIABLE

    @property
    def variables(self):
        """
        Raw captured variable states, if any.
        None if raw data is not being captured, and each variable entry is None if that variable is not captured.
        Returns:
            list (by model.variable) of list (by step/epoch) of variable tensors.
            None if variable capturing not enabled.
        """
        return self._variable_values

    @property
    def collected_variables(self):
        """
        Gets the list of collected variables, containing only those that are collected.
        The indices of the returned variables relative to the original model can only be
        determined via collected_variable_indices().

        Returns:
            list (by captured variable) of list (by step/epoch) of variable tensors.
            None if variable capturing not enabled.
        """
        if self._variable_values is not None:
            return [var_list for var_list in self._variable_values if var_list is not None]
        else:
            return None

    @property
    def collected_variable_indices(self):
        """
        Gets the indices of variables returned by collected_variables() as they
        were on the original model and returned by model.variables.
        """
        return self._filtered_value_variable_indices

    def on_train_begin(self, logs=None):
        """
        Initialises tracking, now that we know the model etc.
        """
        # collect and filter first set of values
        # - precompute _collected_stats_indices and _variable_stats_mask independent of
        #   data collection mixins because they might be disabled
        tracked_variables = self.model.trainable_variables if self.trainable_only else self.model.variables
        values = [value if _index_by_identity(tracked_variables, value) >= 0 else None
                  for value in self.model.variables]
        self._variable_stats_mask = [value is not None for value in values]

        # init stats
        self._init_value_stats(values)
        self._init_activity_stats(values)

        # expand collection_sets and initialise variable storages
        # (TODO also prepare slicing rules)
        if self.collection_sets:
            self.collection_sets = _normalize_collection_sets_for_variables(self.model, self.collection_sets)
            self._filtered_value_variable_indices = [index for collection_set in self.collection_sets
                                                     for index in collection_set['variable_indices']]
            self._variable_values = [[] if var_idx in self._filtered_value_variable_indices else None
                                     for var_idx in range(len(self.model.variables))]

    def on_train_end(self, logs=None):
        """
        Cleans up tracking, and converts some things to numpy arrays for easier consumption.
        High-level indices and stats are all converted to pandas dataframes.
        Raw values are retained as TF values.
        """
        if self.per_step:
            self.steps = np.array(self.steps)
        else:
            self.epochs = np.array(self.epochs)

        # convert data structures to output formats
        self._finalize_value_stats()
        self._finalize_activity_stats()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        if not self.per_step and self.before_updates:
            self.epochs.append(epoch)
            self._do_collection()

    def on_epoch_end(self, epoch, logs=None):
        if not self.per_step and not self.before_updates:
            self.epochs.append(epoch)
            self._do_collection()

    def on_train_batch_begin(self, batch, logs=None):
        if self.per_step and self.before_updates:
            self.steps.append(self.params['steps'] * self._epoch + batch)
            self._do_collection()

    def on_train_batch_end(self, batch, logs=None):
        if self.per_step and not self.before_updates:
            self.steps.append(self.params['steps'] * self._epoch + batch)
            self._do_collection()

    def _do_collection(self):
        values = [value if included else None for value, included
                  in zip(self.model.variables, self._variable_stats_mask)]

        # value stats
        self._collect_value_stats(values)

        # activity stats
        self._accum_activity_stats(values, is_accum=False)
        self._collect_activity_stats(1)  # always summing over 1 sample

        # raw data capture
        self._collect_raw_values(self.model.variables)

    def _collect_raw_values(self, values):
        # TODO do slicing
        if self._variable_values:
            for var_idx, (val_list, value) in enumerate(zip(self._variable_values, values)):
                if val_list is not None:
                    val_list.append(tf.identity(value))  # take copy of current state


class GradientHistoryCallback(BaseGradientCallback, ValueStatsCollectingMixin, ActivityStatsCollectingMixin):
    """
    Custom tot.fit() gradient callback that collects various statistics and/or raw values of the gradients during
    training.
    Only works for the standard gradients of trainable variables.
    See `LayerOutputGradientHistoryCallback` for gradients w.r.t layer outputs.

    When collecting per-epoch stats and values, gradients used are the sum of per-step gradients over
    the course of each epoch. This smooths out the fact that batched stochastic gradient descent tends
    to jump-around as it iterates through the batches in each epoch.

    Other properties:
        model: the model captured
        epochs: list of int. Epoch numbers correlated to captured gradients/gradient-stats
            (only available if per_step is False)
        steps: list of int. Step numbers correlated to captured gradients/gradient-stats
            (only available if per_step is True)
    """

    def __init__(self, per_step=False, collection_sets=None, *args, **kwargs):
        """
        Args:
            per_step: bool. Whether to collect per-step stats and raw values, or per-epoch otherwise.
                By default, data is accumulated per-epoch, and an 'epochs' list is available
                that tracks the display indices of each sample.
                If per-step is set, then a `steps` list is available instead, and activity
                is collected on each update step.
                The same applies to layer output capture if enabled.
            value_norms: bool, default: True.
                Whether to collect the norms of values.
            value_stats: bool, default: True.
                Whether to collect value and magnitude stats.
            activity_stats: bool, default: True.
                Whether to collect activity stats.
            value_stats_quantiles: list of percentiles to collect stats for, in range 0 .. 100.
                Default: [0., 12.5, 25., 37.5, 50., 62.5, 75., 87.5, 100.]
            collection_sets: list of dicts. Fine-grained control over how data is collected across the variables.
              If omitted, this callback collects only stats.
              See _normalize_collection_sets_for_variables() for format details.
        """
        # Callback doesn't honour python 3's MRO, so init mixins directly
        super().__init__(data_format='SC', *args, **kwargs)
        self.per_step = per_step
        self.collection_sets = collection_sets

        # results variable creation
        if per_step:
            self.steps = []
        else:
            self.epochs = []
        self._gradient_values = None

        # internal tracking
        self._epoch = 0
        self._collected_stats_indices_transpose = None  # from model.variable to model.trainable_variable
        self._filtered_value_variable_indices = None
        self._gradients_accumulator = PerEpochAccumulatorStrategy() if not per_step else None

    @property
    def item_name(self):
        """
        Used in plot functions.
        """
        return "Gradient"

    @property
    def item_type(self):
        """
        Used in plot functions.
        """
        return ItemType.VARIABLE

    @property
    def gradients(self):
        """
        Raw captured gradient states, if any.
        None if raw data is not being captured, and each gradient entry is None if that gradient is not captured.
        Returns:
            list (by model.variable) of list (by step/epoch) of gradient tensors.
            None if gradient capturing not enabled.
        """
        return self._gradient_values

    @property
    def collected_gradients(self):
        """
        Gets the list of collected gradients, containing only those that are collected.
        The indices of the returned gradients relative to the original model can only be
        determined via collected_gradient_indices().

        Returns:
            list (by captured gradient) of list (by step/epoch) of gradient tensors.
            None if variable capturing not enabled.
        """
        if self._gradient_values is not None:
            return [var_list for var_list in self._gradient_values if var_list is not None]
        else:
            return None

    @property
    def collected_gradient_indices(self):
        """
        Gets the indices of variables returned by collected_gradients() as they
        were on the original model and returned by model.variables.
        """
        if self._filtered_value_variable_indices is not None:
            return self._filtered_value_variable_indices
        else:
            return None

    def on_train_begin(self):
        """
        Initialises tracking, now that we know the model and number of epochs and steps per epoch.
        """
        # pre-compute lookups
        # - compute independent of data collection mixins because they might be disabled
        v_to_t = [_index_by_identity(self.model.trainable_variables, var) for var in self.model.variables]
        self._collected_stats_indices_transpose = [idx if idx >= 0 else None for idx in v_to_t]

        # expand collection_sets and initialise gradient storages
        # (TODO also prepare slicing rules)
        if self.collection_sets:
            self.collection_sets = _normalize_collection_sets_for_variables(self.model, self.collection_sets)

            # compute variable indices being captured
            # note: we only ever get trainable variables, so we must enforce that filter
            allowed_indices = [v_idx for v_idx in range(len(self.model.variables))
                               if self.model.variables[v_idx].trainable]
            filtered = [index for collection_set in self.collection_sets
                        for index in collection_set['variable_indices']]
            filtered = [v_idx for v_idx in filtered if v_idx in allowed_indices]
            self._filtered_value_variable_indices = filtered

            self._gradient_values = [[] if var_idx in filtered else None
                                     for var_idx in range(len(self.model.variables))]

    def on_train_end(self):
        """
        Cleans up tracking, and converts some things to numpy arrays for easier consumption.
        High-level indices and stats are all converted to pandas dataframes.
        Raw values are retained as TF values.
        """
        if self.per_step:
            self.steps = np.array(self.steps)
        else:
            self.epochs = np.array(self.epochs)

        # convert data structures to output formats
        self._finalize_value_stats()
        self._finalize_activity_stats()

        # free memory
        del self._gradients_accumulator

    def on_epoch_begin(self, epoch):
        """
        Just tracks the current epoch number
        """
        self._epoch = epoch

    def on_epoch_end(self, epoch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Collects gradient stats and raw gradients after each epoch, if configured.
        """
        if not self.per_step:
            self.epochs.append(epoch)
            self._do_collection(self._gradients_accumulator.sum)

    def on_train_batch_end(self, batch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Collects gradient stats and raw gradients after each update step, if configured.
        """
        # per-epoch mode only: accumulate gradients over course of epoch
        if not self.per_step:
            self._gradients_accumulator.accumulate(batch, gradients)

        # per-step mode only: collect stats at each training step
        if self.per_step:
            step = self.params['steps'] * self._epoch + batch
            self.steps.append(step)
            self._do_collection(gradients)

    def _do_collection(self, gradients):
        # note: gradients list is always relative to model.trainable_variables, but I use
        # the model.variables list as the internal reference point, so we must convert the list.
        v_to_t = self._collected_stats_indices_transpose
        values = [gradients[v_to_t[v_idx]] if v_to_t[v_idx] is not None else None
                  for v_idx in range(len(self.model.variables))]

        # value stats
        self._collect_value_stats(values)

        # activity stats
        self._accum_activity_stats(values, is_accum=False)
        self._collect_activity_stats(1)  # always summing over 1 sample

        # raw data capture
        self._collect_raw_values(values)

    def _collect_raw_values(self, values):
        # note: assumes 'values' has been converted into a list relative to model.variables.
        # TODO do slicing
        if self._gradient_values:
            for var_idx, (val_list, value) in enumerate(zip(self._gradient_values, values)):
                if val_list is not None:
                    val_list.append(value)


class LayerOutputHistoryCallback(BaseGradientCallback, ValueStatsCollectingMixin,
                                 ActivityStatsCollectingMixin):
    """
    Custom tot.fit() gradient callback function that collects various statistics and/or raw values of
    layer outputs during training.

    When collecting data per-epoch, by default, raw values and value stats are calculated based on the mean layer
    output over all samples in the epoch. This ensures consistency with `LayerOutputGradientHistoryCallback`.
    Activity stats are always calculated over all individual samples.

    Other properties:
        model: the model captured
        epochs: list of int. Epoch numbers correlated to captured gradients/gradient-stats
            (only available if per_step is False)
        steps: list of int. Step numbers correlated to captured gradients/gradient-stats
            (only available if per_step is True)
    """

    def __init__(self, per_step=False, batch_reduction='auto', keep_dims=False, collection_sets=None, *args, **kwargs):
        """
        Args:
            per_step: bool. Whether to collect per-step stats, or per-epoch otherwise.
                By default, activity is accumulated per-epoch, and an 'epochs' list is available
                that tracks the display indices of each sample.
                If per-step is set, then a `steps` list is available instead, and activity
                is collected on each update step.
                The same applies to layer output capture if enabled.
            batch_reduction: one of 'auto' (default), 'mean', 'sum', or None.
                When doing per-epoch collection, determines how values are accumulated over the course of the epoch
                and over each sample in each batch when computing value norms, value stats, and raw values.
                - 'auto' applies no reduction in per-step mode and 'mean' in per-epoch mode.
                - 'mean' uses the mean value across all samples in the batch/epoch.
                    The batch-dim in returned raw values is either dropped (keep_dims==False)
                    or reduced to size 1 (keep_dims==True).
                - 'sum' uses the sum across all samples in the batch/epoch
                    The batch-dim in returned raw values is either dropped (keep_dims==False)
                    or reduced to size 1 (keep_dims==True).
                - None applies no reduction. For per-step data collection, uses and retains all samples in each batch.
                    For per-epoch data collection, uses and retains only the last batch of each epoch. 
                    Note that the last batch often has less samples than for all other batches in an epoch.
            keep_dims: bool.
                Whether to retain the batch-dim if using 'mean' or 'sum' batch reduction.
            value_norms: bool, default: True.
                Whether to collect the norms of values.
            value_stats: bool, default: True.
                Whether to collect value and magnitude stats.
            activity_stats: bool, default: True.
                Whether to collect activity stats.
            value_stats_quantiles: list of percentiles to collect stats for, in range 0 .. 100.
                Default: [0., 12.5, 25., 37.5, 50., 62.5, 75., 87.5, 100.]
            collection_sets: list of dicts. Enables collection of raw layer outputs
                and provides fine-grained control over which layer outputs are collected.
                If omitted, this callback collects only stats.
                See _normalize_collection_sets_for_layers() for format details.
        """
        super().__init__(data_format='BSC', *args, **kwargs)

        if batch_reduction and batch_reduction not in ('auto', 'mean', 'sum'):
            raise ValueError(f"Invalid batch_reduction: '{batch_reduction}'")
        if batch_reduction == 'auto':
            batch_reduction = None if per_step else 'mean'

        self.per_step = per_step
        self.batch_reduction = batch_reduction
        self.keep_dims = keep_dims
        self.collection_sets = collection_sets

        # results variable creation
        if per_step:
            self.steps = []
        else:
            self.epochs = []
        self._output_values = None  # initially list (by layer) of list (by step/epoch) of layer output tensors

        # internal tracking
        self._epoch = 0
        self._layer_shapes = None
        self._filtered_value_layer_indices = None
        self._activations_accumulator = PerEpochAccumulatorStrategy(batch_reduction=True, keep_dims=keep_dims) \
            if not per_step and self.batch_reduction else None

    @property
    def item_name(self):
        """
        Used in plot functions.
        """
        return "Layer output"

    @property
    def item_type(self):
        """
        Used in plot functions.
        """
        return ItemType.LAYER

    @property
    def layer_shapes(self):
        """
        List of shapes of each layer, regardless of what's being collected.
        Provided for convenience when displaying results and doing other analysis,
        because this information is hard to obtain directly from the model.
        """
        return self._layer_shapes

    @property
    def layer_outputs(self):
        """
        Raw captured layer outputs, if any.
        None if raw data is not being captured, and each layer entry is None if that
        layer is not captured.
        Returns:
            list (by model.layers) of list (by step/epoch) of layer output tensors.
            None if variable capturing not enabled.
        """
        return self._output_values

    @property
    def collected_layer_outputs(self):
        """
        Gets the list of collected layer outputs, containing only those that are collected.
        The indices of the returned layer outputs relative to the original model can only be
        determined via collected_layer_output_indices().

        Returns:
            list (by captured output) of list (by step/epoch) of layer output tensors.
            None if variable capturing not enabled.
        """
        if self._output_values is not None:
            return [var_list for var_list in self._output_values if var_list is not None]
        else:
            return None

    @property
    def collected_layer_output_indices(self):
        """
        Gets the layer indices for outputs returned by collected_layer_outputs() as they
        were on the original model and returned by model.layers.
        """
        if self._output_values is not None:
            return [l_idx for l_idx, var_list in enumerate(self._output_values) if var_list is not None]
        else:
            return None

    def on_train_begin(self):
        """
        Validates configuration and partially expands it, now that we have access to the model.
        """
        # expand collection_sets and initialise variable storages
        # (TODO also prepare slicing rules)
        if self.collection_sets:
            self.collection_sets = _normalize_collection_sets_for_layers(self.model, self.collection_sets)
            self._filtered_value_layer_indices = [index for collection_set in self.collection_sets
                                                  for index in collection_set['layer_indices']]
            self._output_values = [[] if l_idx in self._filtered_value_layer_indices else None
                                   for l_idx in range(len(self.model.layers))]

    def on_train_end(self):
        """
        Cleans up tracking, and converts some things to numpy arrays for easier consumption.
        High-level indices and stats are all converted to pandas dataframes.
        Raw values are retained as TF values.
        """
        if self.per_step:
            self.steps = np.array(self.steps)
        else:
            self.epochs = np.array(self.epochs)
        self._finalize_value_stats()
        self._finalize_activity_stats()

        # free memory
        del self._activations_accumulator

    def on_epoch_begin(self, epoch, logs=None):
        """
        Tracks the current epoch number and resets sums across each epoch
        """
        self._epoch = epoch
        if not self.per_step:
            self._reset_per_epoch_activity_stats()

    def on_train_batch_end(self, batch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Accumulates activations from each step. Also emits stats, if configured at per-step level.
        """
        # initialisation on first access to raw data
        if self._layer_shapes is None:
            self._layer_shapes = [activation.shape for activation in activations]

        # accumulate activation data
        # - always add each batch, regardless of emitting stats per-step or per-epoch
        # - accum when per-epoch, overwrite when per-step
        is_accum = (not self.per_step)
        self._accum_activity_stats(activations, is_accum)

        # per-epoch mode only: accumulate activations over course of epoch
        if not self.per_step:
            self._activations_accumulator.accumulate(batch, activations)

        # stats calculations for each step, if configured
        if self.per_step:
            if self.batch_reduction == 'sum':
                activations = [tf.reduce_sum(t, keepdims=self.keep_dims) for t in activations]
            elif self.batch_reduction == 'mean':
                activations = [tf.reduce_mean(t, keepdims=self.keep_dims) for t in activations]

            self.steps.append(self.params['steps'] * self._epoch + batch)
            self._collect_value_stats(activations)
            self._collect_raw_values(activations)

            # activity stats calculated based on accumulated partial stats
            self._collect_activity_stats(1)

    def on_epoch_end(self, epoch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Collects gradient stats and raw gradients after each epoch, if configured at per-epoch level.
        """
        if not self.per_step:
            # compute aggregated activations, otherwise use last batch
            if self.batch_reduction == 'sum':
                activations = self._activations_accumulator.sum
            elif self.batch_reduction == 'mean':
                activations = self._activations_accumulator.mean

            self.epochs.append(epoch)
            self._collect_value_stats(activations)
            self._collect_raw_values(activations)

            # activity stats calculated based on accumulated partial stats
            self._collect_activity_stats(self.params['steps'])

    def _collect_raw_values(self, activations):
        # TODO do slicing
        if self._output_values:
            for l_idx, val_list in enumerate(self._output_values):
                if val_list is not None:
                    val_list.append(activations[l_idx])


class LayerOutputGradientHistoryCallback(BaseGradientCallback, ValueStatsCollectingMixin,
                                         ActivityStatsCollectingMixin):
    """
    Custom tot.fit() gradient callback function that collects various statistics and/or raw values of
    layer output gradients during training.
    See `GradientHistoryCallback` for the standard gradients w.r.t variables.

    When collecting data per-epoch, by default, raw values and value stats are calculated based on the mean gradient
    over all samples in the epoch. This smooths out the fact that batched stochastic gradient descent tends to
    jump-around as it iterates through the batches in each epoch. It also resolves the fact that the last batch of
    each epoch often has fewer samples than for other batches.
    Activity stats are always calculated over all individual samples.

    Other properties:
        model: the model captured
        epochs: list of int. Epoch numbers correlated to captured gradients/gradient-stats
            (only available if per_step is False)
        steps: list of int. Step numbers correlated to captured gradients/gradient-stats
            (only available if per_step is True)
    """

    def __init__(self, per_step=False, batch_reduction='auto', keep_dims=False, collection_sets=None, *args, **kwargs):
        """
        Args:
            per_step: bool. Whether to collect per-step stats and raw values, or per-epoch otherwise.
                By default, data is accumulated per-epoch, and an 'epochs' list is available
                that tracks the display indices of each sample.
                If per-step is set, then a `steps` list is available instead, and activity
                is collected on each update step.
                The same applies to layer output capture if enabled.
            batch_reduction: one of 'auto' (default), 'mean', 'sum', or None.
                When doing per-epoch collection, determines how values are accumulated over the course of the epoch
                and over each sample in each batch when computing value norms, value stats, and raw values.
                - 'auto' applies no reduction in per-step mode and 'mean' in per-epoch mode.
                - 'mean' uses the mean value across all samples in the batch/epoch.
                    The batch-dim in returned raw values is either dropped (keep_dims==False)
                    or reduced to size 1 (keep_dims==True).
                - 'sum' uses the sum across all samples in the batch/epoch
                    The batch-dim in returned raw values is either dropped (keep_dims==False)
                    or reduced to size 1 (keep_dims==True).
                - None applies no reduction. For per-step data collection, uses and retains all samples in each batch.
                    For per-epoch data collection, uses and retains only the last batch of each epoch. 
                    Note that the last batch often has less samples than for all other batches in an epoch.
            value_norms: bool, default: True.
                Whether to collect the norms of values.
            value_stats: bool, default: True.
                Whether to collect value and magnitude stats.
            activity_stats: bool, default: True.
                Whether to collect activity stats.
            value_stats_quantiles: list of percentiles to collect stats for, in range 0 .. 100.
                Default: [0., 12.5, 25., 37.5, 50., 62.5, 75., 87.5, 100.]
            collection_sets: list of dicts. Enables collection of raw gradients
                and provides fine-grained control over which gradients are collected.
                If omitted, this callback collects only stats.
                See _normalize_collection_sets_for_layers() for format details.
        """
        super().__init__(data_format='BSC', *args, **kwargs)

        if batch_reduction and batch_reduction not in ('auto', 'mean', 'sum'):
            raise ValueError(f"Invalid batch_reduction: '{batch_reduction}'")
        if batch_reduction == 'auto':
            batch_reduction = None if per_step else 'mean'

        self.per_step = per_step
        self.batch_reduction = batch_reduction
        self.keep_dims = keep_dims
        self.collection_sets = collection_sets

        # results variable creation
        if per_step:
            self.steps = []
        else:
            self.epochs = []
        self._gradient_values = None
        self._layer_shapes = None

        # internal tracking
        self._epoch = 0
        self._layer_shapes = None
        self._filtered_value_layer_indices = None
        self._gradients_accumulator = PerEpochAccumulatorStrategy(batch_reduction=True, keep_dims=keep_dims) \
            if not per_step and self.batch_reduction else None

    @property
    def item_name(self):
        """
        Used in plot functions.
        """
        return "Output gradient"

    @property
    def item_type(self):
        """
        Used in plot functions.
        """
        return ItemType.LAYER

    @property
    def needs_output_gradients(self):
        return True

    @property
    def layer_shapes(self):
        """
        List of shapes of each layer, regardless of what's being collected.
        Provided for convenience when displaying results and doing other analysis,
        because this information is hard to obtain directly from the model.
        """
        return self._layer_shapes

    @property
    def gradients(self):
        """
        Raw captured gradients, if any.
        None if raw data is not being captured, and each layer entry is None if that
        layer is not captured.
        When collecting per-step, returned gradients include a batch-dimension which can vary on the last
        batch of each epoch. When collecting per-epoch, returned gradients include a batch-dimension of size 1,
        containing the mean across all samples in the epoch.
        Returns:
            list (by model.layers) of list (by step/epoch) of gradient tensors.
            None if variable capturing not enabled.
        """
        return self._gradient_values

    @property
    def collected_gradients(self):
        """
        Gets the list of collected gradients, containing only those that are collected.
        The indices of the returned gradients relative to the original model can only be
        determined via collected_gradient_indices().

        Returns:
            list (by captured gradient) of list (by step/epoch) of gradient tensors.
            None if variable capturing not enabled.
        """
        if self._gradient_values is not None:
            return [var_list for var_list in self._gradient_values if var_list is not None]
        else:
            return None

    @property
    def collected_gradient_indices(self):
        """
        Gets the indices of layers returned by collected_gradients() as they
        were on the original model and returned by model.layers.
        """
        if self._filtered_value_layer_indices is not None:
            return self._filtered_value_layer_indices
        else:
            return None

    def on_train_begin(self):
        """
        Validates configuration and partially expands it, now that we have access to the model.
        """
        # expand collection_sets and initialise gradient storages
        # (TODO also prepare slicing rules)
        if self.collection_sets:
            self.collection_sets = _normalize_collection_sets_for_layers(self.model, self.collection_sets)
            self._filtered_value_layer_indices = [index for collection_set in self.collection_sets
                                                  for index in collection_set['layer_indices']]
            self._gradient_values = [[] if l_idx in self._filtered_value_layer_indices else None
                                     for l_idx in range(len(self.model.layers))]

    def on_train_end(self):
        """
        Cleans up tracking, and converts some things to numpy arrays for easier consumption.
        High-level indices and stats are all converted to pandas dataframes.
        Raw values are retained as TF values.
        """
        if self.per_step:
            self.steps = np.array(self.steps)
        else:
            self.epochs = np.array(self.epochs)
        self._finalize_value_stats()
        self._finalize_activity_stats()

        # free memory
        del self._gradients_accumulator

    def on_epoch_begin(self, epoch):
        """
        Tracks the current epoch number and resets sums across each epoch
        """
        self._epoch = epoch
        if not self.per_step:
            self._reset_per_epoch_activity_stats()

    def on_train_batch_end(self, batch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Accumulates activations from each step. Also emits stats, if configured at per-step level.
        """
        # initialisation on first access to raw data
        if self._layer_shapes is None:
            self._layer_shapes = [output.shape if output is not None else None for output in output_gradients]

        # accumulate activation data
        # - always add each batch, regardless of emitting stats per-step or per-epoch
        # - accum when per-epoch, overwrite when per-step
        is_accum = (not self.per_step)
        self._accum_activity_stats(output_gradients, is_accum)

        # per-epoch mode only: accumulate gradients over course of epoch
        if not self.per_step:
            self._gradients_accumulator.accumulate(batch, output_gradients)

        # stats calculations for each step, if configured
        if self.per_step:
            if self.batch_reduction == 'sum':
                output_gradients = [tf.reduce_sum(t, keepdims=self.keep_dims) for t in output_gradients]
            elif self.batch_reduction == 'mean':
                output_gradients = [tf.reduce_mean(t, keepdims=self.keep_dims) for t in output_gradients]

            self.steps.append(self.params['steps'] * self._epoch + batch)
            self._collect_value_stats(output_gradients)
            self._collect_raw_values(output_gradients)

            # activity stats calculated based on accumulated partial stats
            self._collect_activity_stats(1)

    def on_epoch_end(self, epoch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Collects gradient stats and raw gradients after each epoch, if configured at per-epoch level.
        """
        if not self.per_step:
            # compute aggregated gradients, otherwise use last batch
            if self.batch_reduction == 'sum':
                output_gradients = self._gradients_accumulator.sum
            elif self.batch_reduction == 'mean':
                output_gradients = self._gradients_accumulator.mean

            self.epochs.append(epoch)
            self._collect_value_stats(output_gradients)
            self._collect_raw_values(output_gradients)

            # activity stats calculated based on accumulated partial stats
            self._collect_activity_stats(self.params['steps'])

    def _collect_raw_values(self, gradients):
        # TODO do slicing
        if self._gradient_values:
            for l_idx, val_list in enumerate(self._gradient_values):
                if val_list is not None:
                    if gradients[l_idx] is None:
                        # turns out that gradients aren't calculated for this layer
                        self._gradient_values[l_idx] = None
                    else:
                        val_list.append(gradients[l_idx])


class LearningRateHistoryCallback(BaseGradientCallback):
    """
    ADAM optimizers and others that use momentum have an effective dynamic learning rate that is computed
    per-element and adapts to changes in the loss landscape during training. This callback attempts
    to calculate that "implicit" per-element learning rate, and then to collect stats across the learning
    rates for reporting.

    When computing implicit learning rates on a per-epoch basis, computed values are an approximation. To get an
    accurate mean learning rate we'd need to compute the learning rate and norm every update step and calculate the
    norm at the end of the epoch, but this increases computation overhead by about 2.5x compared to the estimation
    approach used here.

    Multiple update steps affect each model variable as follows (where V = variable tensor, R = implicit learning
    rate tensor, and G = gradient tensor), and (*) = element-wise multiplication):
    > delta_V = sum{i=1..k}[R_i (*) G_i]

    We compute the gradient sum G_s over the whole epoch, and use the full epoch delta_V to estimate
    an average implicit learning rate R_mu as (where (/) is element-wise division). This is approximately a
    gradient-weighted average of the individual per-step learning rates, with the caveat that gradients can be
    negative. In practice the ILR norm trends measured this way tend to follow the same overall scale of a more
    accurate per-epoch ILR mean.
    >  delta_V = R_mu (*) G_s  => r_mu = delta_V (/) G_s

    If you need a more accurate result, it's better to just collect per_step data and do the averaging yourself after.
    """
    def __init__(self, per_step=False, stats=True):
        """
        Args:
            per_step: bool.
                Whether to collect per-step values, or per-epoch otherwise.
            stats: bool.
                Whether to collect percentile stats over per-element learning rates.
                Adds considerable time to training.
        """
        super().__init__()
        self.per_step = per_step
        self.stats = stats
        self.quantiles = [0., 12.5, 25., 37.5, 50., 62.5, 75., 87.5, 100.]

        # initially: list (by iteration) of list (by variable) of norm tensor
        # finally: list (by variable) of np-array with shape (iteration,)
        self._ilr_norms = []

        # initially: list (by iteration) of list (by variable) of percentiles tensor
        # finally: list (by variable) of pd-DataFrame with shape (iteration, percentile)
        self._ilr_stats = [] if stats else None

        # internal tracking
        self._variables_before = None
        self._gradients_accumulator = PerEpochAccumulatorStrategy() if not per_step else None

    @property
    def model_norm_stats(self):
        """
        Pandas DataFrame with shape (iteration, percentiles) of stats over the norms of implicit learning rates.
        """
        # gather into shape (iterations, variables)
        # then compute stats and return as (iterations, percentiles)
        q = [0, 25, 50, 75, 100]
        num_iterations = len(self._ilr_norms[0])
        data = np.stack([[norms[it] for norms in self._ilr_norms] for it in range(num_iterations)], axis=0)
        data = tfp.stats.percentile(data, q, axis=-1).numpy().T
        return pd.DataFrame(data, columns=q)

    @property
    def model_stats(self):
        """
        Pandas DataFrame with shape (iteration, percentiles) of stats over the medians of implicit learning rates.
        """
        q = [0, 25, 50, 75, 100]
        data = np.stack([stats[50] for stats in self._ilr_stats], axis=1)
        data = tfp.stats.percentile(data, q, axis=-1).numpy().T
        return pd.DataFrame(data, columns=q)

    @property
    def ilr_norms(self):
        """
        A list (by trainable variable) of 1D-array (iteration) containing the norms of the per-element
        "implicit learning rates". Norms are size-adjusted, making them mathematically equivalent to RMS of
        the per-element values.
        """
        return self._ilr_norms

    @property
    def ilr_stats(self):
        """
        A list (by trainable variable) of pandas DataFrame, with shape (iteration, percentile), containing percentile
        stats over the per-element "implicit learning rates".
        None if not enabled.
        """
        return self._ilr_stats

    def on_train_end(self):
        def gather(iteration_item_data, v_idx):
            item_data = [iteration_values[v_idx] for iteration_values in iteration_item_data]
            return np.stack(item_data, axis=0)  # shape: (iterations, ...)

        # convert to final format
        num_vars = len(self._ilr_stats[0])
        self._ilr_norms = [gather(self._ilr_norms, v_idx) for v_idx in range(num_vars)]
        if self._ilr_stats is not None:
            self._ilr_stats = [pd.DataFrame(gather(self._ilr_stats, v_idx), columns=self.quantiles)
                               for v_idx in range(num_vars)]

        # free memory
        del self._variables_before
        del self._gradients_accumulator
        
    def on_epoch_begin(self, epoch):
        self._variables_before = [tf.identity(v) for v in self.model.trainable_variables]

    def on_epoch_end(self, epoch, loss, gradients, trainable_variables, activations, output_gradients):
        # per-epoch
        if not self.per_step:
            self._do_collect(self._variables_before, self._gradients_accumulator.sum)

    def on_train_batch_end(self, batch, loss, gradients, trainable_variables, activations, output_gradients):
        if self.per_step:
            # per-step mode: collect and update 'before' for next step
            self._do_collect(self._variables_before, gradients)
            self._variables_before = [tf.identity(v) for v in self.model.trainable_variables]
        else:
            # per-epoch mode: accumulate gradients over course of epoch
            self._gradients_accumulator.accumulate(batch, gradients)

    def _do_collect(self, variables_before, gradients):
        # size-normalized norms (tf.norm(tensor) / tf.sqrt(tf.size(tensor))) are mathematically equivalent
        # to the root-mean-square of the tensor. In practice there are some numerical stability differences in
        # the implementation, and the RMS implementation is faster (~4x faster on GPU with tensors of size 2.5M).
        # Thus, we actually compute the RMS value rather than using tf.norm().
        deltas = [after - before for after, before in zip(self.model.trainable_variables, variables_before)]
        implicit_learning_rates = [-delta / g for delta, g in zip(deltas, gradients)]
        implicit_learning_rates = [tf.gather_nd(ilr, tf.where(tf.math.is_finite(ilr))) for ilr in
                                   implicit_learning_rates]
        ilr_norms = [tf.sqrt(tf.reduce_mean(tf.square(ilr))) for ilr in implicit_learning_rates]

        self._ilr_norms.append(ilr_norms)
        if self._ilr_stats is not None:
            ilr_percentiles = [tfp.stats.percentile(ilr, self.quantiles) for ilr in implicit_learning_rates]
            self._ilr_stats.append(ilr_percentiles)


class ItemType(Enum):
    """
    Broad categorisation of item data collected by a given callback.
    """
    LAYER = 1,
    VARIABLE = 2


def _log_normalize(arr, axis=None):
    """
    Normalises all values in the array or along the given axis so that they sum to 1.0,
    and so that large scale differences are reduced to linear differences in the same
    way that a log plot converts orders of magnitude differences to linear differences.

    Args:
        arr: numpy array or similar of values in range 0.0 .. +inf

    Returns:
        new array
    """
    # mask to work only on positive values
    if np.min(arr) < 0.0:
        print(f"WARN: negative values in _log_normalize are clipped at zero: {np.min(arr)}")
    mask = arr <= 0.0

    # convert to log scale
    # - result: orders-of-magnitude numbers in range -inf..+inf (eg: -4 to +4)
    # - avoid runtime warning about zeros
    arr = arr.copy()
    arr[mask] = 1.0  # dummy value to avoid warnings from np.log(), will be discarded
    scaled = np.log(arr)

    # move everything into positive
    # - shift such that the min value gets value 1.0, so it doesn't become zero.
    # - in simple terms we're doing:
    #     scaled = scaled - (np.min(scaled, axis=axis=, keepdims=True) - 1)
    dummy_max = np.max(scaled) + 1
    scaled[mask] = dummy_max  # so as not to affect np.min()
    offset = np.min(scaled, axis=axis, keepdims=True) - 1
    offset[offset == dummy_max] = 0.0  # remove dummy_max values
    scaled = scaled - offset
    scaled[mask] = 0.0  # final masked values

    # normalize
    tots = np.sum(scaled, axis=axis, keepdims=True)
    tots[tots == 0.0] = 1.0  # avoid divide-by-zero
    scaled = scaled / tots
    return scaled


def pos_neg_balance(stats, quantiles=None):
    """
    Calculates the "balance" of positive and negative values, based on value percentile information.
    Estimates the percentile of the zero line and returns values in range -1.0 to 1.0 indicating the relative
    fraction of values that fall on either side of the zero line. For example, -1.0 means that 100% of values
    are negative, 0.0 means that the values are evenly split between negative and positive, and +1.0 means that 100%
    of values are positive.
    Args:
        stats: pandas dataframe with shape (rows, percentiles) and column names identifying the percentile in
            range 0..100, or an array of the same where percentiles are evenly distributed from 0 to 100.
        quantiles: list of quantiles for each column.
            Usually the quantiles can be directly extracted from the column names of the dataframe, or
            inferred from the shape of the stats assuming evenly distributed percentiles from 0 to 100%.
            If not, quantiles can be explicitly provided. Must be values in range 0 to 100, but can be floats.
    Returns:
        np-array with shape (rows,) and values in range -1.0 to 1.0
    """
    # parse arguments
    if isinstance(stats, pd.DataFrame):
        values = stats.to_numpy()
        if quantiles is None:
            quantiles = pd.to_numeric(stats.columns, errors='raise').to_numpy()
    else:
        values = np.array(stats)
        if quantiles is None:
            quantiles = np.linspace(0, 100, num=values.shape[1], endpoint=True)

    balances = []
    for row_idx, row in enumerate(values):
        # trivial cases - all positive or all negative
        if np.all(row >= 0):
            balance = 1.0
        elif np.all(row <= 0):
            balance = -1.0
        else:
            # find point closest to zero
            mid_col_idx = np.argmin(np.abs(row))
            mid_q = quantiles[mid_col_idx]

            # fit curve to nearby points
            if mid_col_idx == 0:
                # left edge: fit line to first two
                p = Polynomial.fit(x=quantiles[:2], y=row[:2], deg=1)
            elif mid_col_idx == len(row) - 1:
                # right edge: fit line to last two
                p = Polynomial.fit(x=quantiles[-2:], y=row[-2:], deg=1)
            else:
                # in middle: fit curve to middle three
                p = Polynomial.fit(x=quantiles[mid_col_idx - 1:mid_col_idx + 2],
                                   y=row[mid_col_idx - 1:mid_col_idx + 2], deg=2)

            # extrapolate accurate zero-point quantile
            # - gives a floating point value in range 0 to 100
            roots = p.roots()
            root_idx = np.argmin(np.abs(roots - mid_q))  # pick root closest to mid-point
            zero_q = roots[root_idx]

            # convert to balance
            frac_pos = (100 - zero_q) / 100
            balance = 2 * frac_pos - 1
        balances.append(balance)
    return np.array(balances)


def _compute_common_stats_keys():
    """
    Gets the dictionary keys that are always returned by _compute_stats().
    Returns:
        list of string
    """
    return ['mean', 'min', 'max', 'std']


# Implementation note: optimised for efficient use within TF loops
# TODO may need to add some div-by-zero protection at some point
@tf.function
def _compute_common_stats(tensors: list, absolute: bool = False):
    """
    Computes common statistics across the provided values.

    Note: may issue warnings about retracing, but they can be ignored.
    This method must deal with the variations across each of the ways that it's called during a train step,
    (eg: whole model vs each layer), but after the first train step it will be fully optimised.

    Args:
        tensors: a list of tensors, all  assumed to be of type float, but may have different shapes
        absolute: whether to take `abs(tensors)` or to use their raw values otherwise
    Returns:
        dict of form: {
            'mean': float,
            'min': float,
            'max': float,
            'std': float   # standard deviation
        }
    """
    tot_n = tf.constant(0.0, dtype=tf.float32)
    tot_mean = tf.constant(0.0, dtype=tf.float32)
    tot_m2 = tf.constant(0.0, dtype=tf.float32)  # Sum of squared differences from the mean
    tot_min = tf.constant(float("inf"), dtype=tf.float32)
    tot_max = tf.constant(float("-inf"), dtype=tf.float32)

    for tensor in tensors:
        values = tf.abs(tensor) if absolute else tensor
        t_size = tf.size(tensor, out_type=tf.float32)
        t_min = tf.reduce_min(values)
        t_max = tf.reduce_max(values)
        t_sum = tf.reduce_sum(values)
        t_mean = t_sum / t_size
        t_var = tf.reduce_sum((values - t_mean) ** 2)  # variance

        tot_min = tf.minimum(tot_min, t_min)
        tot_max = tf.maximum(tot_max, t_max)

        # Welford's algorithm for computing running statistics
        delta = t_mean - tot_mean
        tot_n += t_size
        tot_mean += delta * (t_size / tot_n)
        tot_m2 += t_var + delta ** 2 * (t_size * (tot_n - t_size) / tot_n)

    return {
        'mean': tot_mean,
        'min': tot_min,
        'max': tot_max,
        'std': tf.sqrt(tot_m2 / tot_n)  # population std.dev
    }


@tf.function
def _compute_percentile_stats(tensor, quartiles, magnitudes=False):
    """
    Computes a set of percentiles across the given tensor.

    For tensors that commonly have values distributed either side of zero, the median will
    typically be around zero, and the 25th and 75th percentiles represent the respective medians
    in the positive and negative halves.

    Note: I haven't done experiments yet but in theory auto-graphing this function still adds some
    performance improvement even though it's only calling into a single standard TF function.

    Args:
        tensor: single tensor of values
        quartiles: list of quantiles to compute values for
        magnitudes: whether to calculate against the magnitudes of the tensor values, or the raw values otherwise
    Returns:
        single tensor of percentile values
    """
    if magnitudes:
        tensor = tf.abs(tensor)
    return tfp.stats.percentile(tensor, quartiles, interpolation='linear')


def variable_indices_by_layer(model: tf.keras.Model, include_trainable_only: bool = False):
    """
    Groups indices of model.variable by layer.

    Note: even with include_trainable_only==True, this produces different results from
    trainable_variable_indices_by_layer() which indexes variables by their position in model.trainable_variables.

    Params:
        model: a model
        include_trainable_only: bool
            Whether to only include the indices for trainable variables, or for all variables otherwise.
    Returns:
         list (by layer index) of list (by variable) of variable indices from original model.variable list.
    """
    if include_trainable_only:
        return [[_index_by_identity(model.variables, var) for var in layer.trainable_variables]
                for layer in model.layers]
    else:
        return [[_index_by_identity(model.variables, var) for var in layer.variables]
                for layer in model.layers]


def trainable_variable_indices_by_layer(model: tf.keras.Model):
    """
    Groups indices of model.trainable_variable by layer.
    This is different to variable_indices_by_layer() which indexes variables by their position
    in model.variables.
    Params:
        model: a model
    Returns:
         list (by layer index) of list (by variable) of variable indices from original model.trainable_variable list.
    """
    return [[_index_by_identity(model.trainable_variables, var) for var in layer.trainable_variables]
            for layer in model.layers]


def variable_indices_to_trainable_variable_indices(model, variable_indices, non_trainable_strategy='warn'):
    """
    Takes a list of variable indices relative to model.variables, and converts them to be relative
    to model.trainable_variables.
    Note that some variables are not listed within model.trainable_variables.
    Args:
        model: the model from which the variable indices came
        variable_indices: indices of existing variables on model.variables
        non_trainable_strategy: how to handle any non-trainable variable indices, valid options:
            'warn' - emit a warning
            'error' - fail
            'ignore' - silently ignore
    Returns:
        list of indices relative to model.trainable_variables
    """
    if non_trainable_strategy not in ['warn', 'error', 'ignore']:
        raise ValueError(f"non_trainable_strategy must be one of 'warn', 'error', 'ignore'. "
                         f"Got: {non_trainable_strategy}")

    # do mapping
    unfiltered = [_index_by_identity(model.trainable_variables, model.variables[var_idx])
                  for var_idx in variable_indices]
    filtered = [idx for idx in unfiltered if idx >= 0]

    # check for problems
    problems = [var_idx for inp_idx, var_idx in enumerate(variable_indices) if unfiltered[inp_idx] == -1]
    if problems:
        if non_trainable_strategy == 'error':
            raise ValueError(f"Cannot convert non-trainable variable indices {problems}")
        elif non_trainable_strategy == 'warn':
            print(
                f"WARN: variable_indices_to_trainable_variable_indices() dropped non-trainable variable indices {problems}")

    # return final result minus any problem entries
    return filtered


def trainable_variable_indices_to_variable_indices(model, trainable_variable_indices=None):
    """
    Takes a list of variable indices relative to model.trainable_variables, and converts them to be relative
    to model.variables.
    Args:
        model: the model from which the variable indices came
        trainable_variable_indices: Optional.
          Indices of existing variables on model.trainable_variables.
          If not provided, this function simply returns a lookup table for all trainable variable indices.
    Returns:
        list of indices relative to model.variables
    """
    if trainable_variable_indices is None:
        trainable_variable_indices = range(len(model.trainable_variables))
    return [_index_by_identity(model.variables, model.trainable_variables[var_idx])
            for var_idx in trainable_variable_indices]


def layer_indices_by_variable(model):
    """
    Reverses variable_indices_by_layer(), providing a lookup from variable index to layer index.
    Params:
        model: a model
    Returns:
        list (by variable index) of the layer to which the variable belongs relative to model.layers
    """
    lookup = [None] * len(model.variables)
    for l_idx, var_indices in enumerate(variable_indices_by_layer(model, include_trainable_only=False)):
        for v_idx in var_indices:
            lookup[v_idx] = l_idx
    return lookup


def layer_indices_by_trainable_variable(model):
    """
    Reverses trainable_variable_indices_by_layer(), providing a lookup from trainable variable index to layer index.
    Params:
        model: a model
    Returns:
        list (by trainable variable index) of the layer to which the variable belongs relative to model.layers
    """
    lookup = [None] * len(model.trainable_variables)
    for l_idx, var_indices in enumerate(trainable_variable_indices_by_layer(model)):
        for v_idx in var_indices:
            lookup[v_idx] = l_idx
    return lookup


def _index_by_identity(lst, target):
    """
    Some classes override the equals() method s.t. you can't simply
    do `lst.index(target)`. This function overcomes that problem.
    """
    return next((i for i, v in enumerate(lst) if id(v) == id(target)), -1)


# note: experiments have found that using a dict and iterating over it like this is not as performant
# as I'd like, but all attempts I've tried to do better ultimately fail.
# For example, using TF Variables to store lists goes considerably slower - on scale of 40ms vs 14ms per epoch.
def _append_dict_list(dic, addendum_dict):
    for key in addendum_dict.keys():
        dic[key].append(addendum_dict[key])


def _compute_model_summary_stats(iteration_item_values, quantiles=None):
    """
    Calculates meta-stats against a set of quantile stats.
    Assumes the use of dataframes where each column represents a quantile, represented as a number in range 0 to 100.
    Args:
        iteration_item_values: np array of shape (iterations, items) containing data to summarise
        quantiles: set of quantiles to return in final result.
    Returns:
        pandas dataframe with shape (iterations, quantiles)
    """
    quantiles = quantiles or [0, 25, 50, 75, 100]

    # calculate stats across the raw per-item values
    stats = tfp.stats.percentile(iteration_item_values, quantiles, axis=-1, interpolation='linear')
    stats = tf.transpose(stats)

    return pd.DataFrame(stats.numpy(), columns=quantiles)
    # return as dataframe with quantiles as columns


# TODO consider alternatively estimating the original mean as something like
#    sum([percentile * quantile for percentile,quantile in zip(percentiles, stats)])
#  more accurately:
#    sum([(percentile-prev_percentile) * (quantile+prev_quantile)/2 for ...
#         in zip(offset(percentiles), offset(stats), percentiles, stats])
def get_scales_across_stats_list(stats_dataframes, scale_quantile=50):
    """
    Extracts a set of "scale" heuristics from a set of quantile stats.
    Assumes the use of dataframes where each column represents a quantile, represented as a number in range 0 to 100.

    In the future, might change to defaulting to estimate the mean magnitude.
    Note that this works best if stats have been collected ocross magnitudes.

    Args:
        stats_dataframes: list of pandas dataframes of shape (iterations, quantiles)
        scale_quantile: a hint about the quantile to attempt to use as the proxy for the
            "scale" of typical values.
            If present, the both this quantile and its opposite (100 - scale_quantile)
            must be present within the stats.
    Returns:
        np-array with shape (iterations, variables)
    """
    # Implementation note:
    # - previously I'd done things like picking the 50th percentile, or assumed that the 25th and 75th
    #   were either side of zero. But they all run into problems.
    #   For example, taking the 50th even on magnitude data can be a problem if slightly > 50% of the
    #   values are zero. In the end, estimating the mean from the percentiles just seems like the best approach.
    #
    # TODO assume a normal distribution and use that to downplay the weight for the outer percentiles

    scales = [np.mean(np.abs(variable_stat.to_numpy()), axis=1)
              for variable_stat in stats_dataframes
              if variable_stat is not None]
    return np.stack(scales, axis=-1)  # shape: iterations x variables


def _normalize_collection_sets_for_layers(model: tf.keras.Model, collection_sets: list):
    """
    Handles the variations allowed in collection_sets used for selecting capture of
    per-layer-output data.
    Fully resolves all collection sets to: 'layer_indices' and 'slices'.

    Returns a copy of the provided collection_sets because the inputs may be re-used between
    multiple callbacks. As in this example:
    > collection_sets=[{layer_indices: [5]}]
    > variables = VariableHistoryCallback(collection_sets=collection_sets)
    > activity = ActivityHistoryCallback(collection_sets=collection_sets)
    > gradients = GradientHistoryCallback(collection_sets=collection_sets)
    > fit(model, train_data, callbacks=[variables, activity, gradients])

    Args:
        model: the model being examined
        collection_sets: list of dicts. Fine-grained control over how data is collected across the variables.
          If omitted, this callback collects nothing (in the future it may collect stats).
          Dicts of form (note: 'density' and 'slices' not yet supported):
            {
              # none or one of:
              'layers': [Layer]   # references to actual layers, to capture all variables in the given layers
              'layer_indices': [int]  # list of layer indices, to capture all variables in the given layers
              'layer_names': [string]  # list of layer names, to capture all variables in the given layers

              # applicable if none of above specified:
              'include_non_trainable': bool (default False)  # whether to include non-trainable layers

              # one of (NOT YET SUPPORTED):
              'density': float, default: 1.0  # fraction of units to collect outputs from, automatically sliced
              'max_units': int, default: None  # max number of units to collect outputs from, automatically sliced
              'slices': [slice]  # slices to use for each selected layer
            }
          A dict that omits layer or variable references applies its density/slicing rule to each trainable
          variable that hasn't otherwise been specified in any other collection sets.

    Returns:
        new collection_sets after modification
    """
    # precompute lookups
    layer_names = [layer.name for layer in model.layers]
    all_layer_indices = list(range(len(model.layers)))
    onlytrainable_layer_indices = [_index_by_identity(model.layers, layer) for layer in model.layers
                                   if layer.trainable_variables]

    tracked_layer_indices = set()  # flat set of variable indices

    # validate and standardise on closed layer indices
    # (lookups automatically throws ValueError if any references not present in model layers)
    collection_sets = _copy_collection_sets(collection_sets)
    for collection_set in collection_sets:
        _assert_at_most_one_property_of(collection_set, ['layers', 'layer_indices', 'layer_names'])

        # identify layers
        layer_indices = None
        if collection_set.get('layer_indices'):
            layer_indices = collection_set['layer_indices']
        elif collection_set.get('layers'):
            layer_indices = [_index_by_identity(model.layers, layer) for layer in collection_set['layers']]
        elif collection_set.get('layer_names'):
            layer_indices = [layer_names.index(name) for name in collection_set['layer_names']]

        # validate no duplicate indices
        if layer_indices is not None:
            duplicates = [index for index in layer_indices if index in tracked_layer_indices]
            if duplicates:
                raise ValueError(f"Duplicate references to layers not allowed. Duplicate indices found: {duplicates}")

        # commit
        collection_set['layer_indices'] = layer_indices
        if layer_indices is not None:
            tracked_layer_indices.update(layer_indices)

    # validate at-most one open set for layer indices
    open_collection_sets = [collection_set for collection_set in collection_sets
                            if collection_set['layer_indices'] is None]
    if len(open_collection_sets) > 1:
        raise ValueError(f"At most one collection set may be specified without any layer references. "
                         f"Found: {open_collection_sets}")

    # infer and standardise on open layer indices
    for collection_set in collection_sets:
        if collection_set['layer_indices'] is None:
            include_non_trainable = collection_set.get('include_non_trainable', False)
            indices_lookup = all_layer_indices if include_non_trainable else onlytrainable_layer_indices
            layer_indices = [index for index in indices_lookup if index not in tracked_layer_indices]
            collection_set['layer_indices'] = layer_indices
            tracked_layer_indices.update(layer_indices)

    # - validate and standardise on slicing
    for collection_set in collection_sets:
        _assert_at_most_one_property_of(collection_set, ['density', 'max_units', 'slices'])

        # TODO
        slices = None
        if collection_set.get('density'):
            if collection_set['density'] != 1.0:
                raise ValueError("Only density=1.0 currently supported")
        elif collection_set.get('max_units'):
            raise ValueError("Only density=1.0 currently supported")
        elif collection_set.get('slices'):
            raise ValueError("Only density=1.0 currently supported")
        else:
            # defaults to density = 1.0
            pass

    return collection_sets


def _normalize_collection_sets_for_variables(model: tf.keras.Model, collection_sets: list):
    """
    Handles the variations allowed in collection_sets used for selecting capture of model variables.
    Fully resolves all collection sets to: 'variable_indices' and 'slices'.

    Returns a copy of the provided collection_sets because the inputs may be re-used between
    multiple callbacks. As in this example:
    > collection_sets=[{layer_indices: [5]}]
    > variables = VariableHistoryCallback(collection_sets=collection_sets)
    > activity = ActivityHistoryCallback(collection_sets=collection_sets)
    > gradients = GradientHistoryCallback(collection_sets=collection_sets)
    > fit(model, train_data, callbacks=[variables, activity, gradients])

    Args:
        model: the model being examined
        collection_sets: list of dicts. Fine-grained control over how data is collected across the variables.
          If omitted, this callback collects nothing (in the future it may collect stats).
          Dicts of form (note: 'density' and 'slices' not yet supported):
            {
              # none or one of:
              'layers': [Layer]   # references to actual layers, to capture all variables in the given layers
              'layer_indices': [int]  # list of layer indices, to capture all variables in the given layers
              'layer_names': [string]  # list of layer names, to capture all variables in the given layers
              'variables': [Variable]  # references to actual variables
              'variable_indices': [int]  # list of variable indices according to model.variables
              'trainable_variable_indices': [int]  # list of variable indices according to model.trainable_variables

              # applicable if none of above specified, or if using 'layers', 'layer_indices', or 'layer_names':
              'include_non_trainable': bool (default False)  # whether to include non-trainable variables

              # one of (NOT YET SUPPORTED):
              'density': float, default: 1.0  # fraction of units to collect outputs from, automatically sliced
              'max_units': int, default: None  # max number of units to collect outputs from, automatically sliced
              'slices': [slice]  # slices to use for each selected variable
            }
          A dict that omits layer or variable references applies its density/slicing rule to each trainable
          variable that hasn't otherwise been specified in any other collection sets.

    Returns:
        new collection_sets after modification
    """
    # precompute lookups
    layer_names = [layer.name for layer in model.layers]
    all_variable_indices = list(range(len(model.variables)))
    onlytrainable_variable_indices = [_index_by_identity(model.variables, var) for var in model.trainable_variables]
    all_variable_indices_by_layer = variable_indices_by_layer(model, include_trainable_only=False)
    onlytrainable_variable_indices_by_layer = variable_indices_by_layer(model, include_trainable_only=True)

    tracked_variable_indices = set()  # flat set of variable indices

    # validate and standardise on closed variable indices
    # (lookups automatically throws ValueError if any references not present in model layers)
    collection_sets = _copy_collection_sets(collection_sets)
    for collection_set in collection_sets:
        _assert_at_most_one_property_of(collection_set, ['layers', 'layer_indices', 'layer_names',
                                                         'variable_indices', 'trainable_variable_indices'])

        include_non_trainable = collection_set.get('include_non_trainable', False)

        variable_indices = None
        if collection_set.get('variable_indices'):
            variable_indices = collection_set['variable_indices']
        elif collection_set.get('trainable_variable_indices'):
            trainable_indices = collection_set['trainable_variable_indices']
            variables = [model.trainable_variables[i] for i in trainable_indices]
            variable_indices = [_index_by_identity(model.variables, var) for var in variables]
        elif any([key for key in ['layers', 'layer_names', 'layer_indices'] if key in collection_set]):
            # identify layers
            if collection_set.get('layer_indices'):
                layer_indices = collection_set['layer_indices']
            elif collection_set.get('layers'):
                layer_indices = [_index_by_identity(model.layers, layer) for layer in collection_set['layers']]
            elif collection_set.get('layer_names'):
                layer_indices = [layer_names.index(name) for name in collection_set['layer_names']]
            else:
                raise AssertionError("Woops, unrecognised layer specifier type")

            # lookup variables from each selected layer
            indices_lookup_by_layer = all_variable_indices_by_layer if include_non_trainable \
                else onlytrainable_variable_indices_by_layer
            variable_indices = [item for l_idx in layer_indices for item in indices_lookup_by_layer[l_idx]]
        else:
            # leave expansion till later
            pass

        # validate no duplicate indices
        if variable_indices is not None:
            duplicates = [index for index in variable_indices if index in tracked_variable_indices]
            if duplicates:
                raise ValueError(f"Duplicate references to variables not allowed. "
                                 f"Duplicate indices found: {duplicates}")

        # commit
        collection_set['variable_indices'] = variable_indices
        if variable_indices is not None:
            tracked_variable_indices.update(variable_indices)

    # validate at-most one open set for variable indices
    # infer and standardise on open variable indices
    open_collection_sets = [collection_set for collection_set in collection_sets
                            if collection_set.get('variable_indices') is None]
    if len(open_collection_sets) > 1:
        raise ValueError(f"At most one collection set may be specified without any layer or variable references. "
                         f"Found: {open_collection_sets}")

    # infer and standardise on open variable indices
    for collection_set in collection_sets:
        if collection_set['variable_indices'] is None:
            include_non_trainable = collection_set.get('include_non_trainable', False)
            indices_lookup = all_variable_indices if include_non_trainable else onlytrainable_variable_indices
            variable_indices = [index for index in indices_lookup if index not in tracked_variable_indices]
            collection_set['variable_indices'] = variable_indices
            tracked_variable_indices.update(variable_indices)

    # - validate and standardise on slicing
    for collection_set in collection_sets:
        _assert_at_most_one_property_of(collection_set, ['density', 'max_units', 'slices'])

        # TODO
        slices = None
        if collection_set.get('density'):
            if collection_set['density'] != 1.0:
                raise ValueError("Only density=1.0 currently supported")
        elif collection_set.get('max_units'):
            raise ValueError("Only density=1.0 currently supported")
        elif collection_set.get('slices'):
            raise ValueError("Only density=1.0 currently supported")
        else:
            # defaults to density = 1.0
            pass

    return collection_sets


def _assert_at_most_one_property_of(source_list, allowed: list):
    """
    Internal method used for validating input arguments.
    Verifies that `source_list` contains at most one of the items in allowed, while ignoring everything else.
    """
    present = [key for key in allowed if key in source_list]
    if len(present) > 1:
        raise ValueError(f"At most one of {allowed} can be present. Found: {present}")


def _copy_collection_sets(collection_sets):
    """
    Internal method used by `_normalize_collection_sets_for_xxx()`.
    Performs an intermediate-depth clone:
    - clones the outer list and each dict,
    - but copies the dict values by reference.

    Copes with the possibility that the user has created a single collection_sets object and shared amongst multiple
    callbacks, while also coping with the possibility that some of the dict values are lists of layer or variable
    object references, which we don't want to try to clone.
    """
    result_sets = []
    for collection_set in collection_sets:
        copied = collection_set.copy()
        result_sets.append(copied)
    return result_sets


def measure_unit_activity(model, dataset, include_channel_activity=False, include_spatial_activity=False,
                          verbose=0, **kwargs):
    """
    Measures the rate of unit activations (having non-zero output) across all units in all layers, when
    predictions are made against the X values in the given dataset, and computes stats over the results.

    All layers are assumed to produce outputs with shapes of form: `(batch_size, ..spatial_dims.., channels)`.
    Unit activation rates are recorded per-channel, aggregated across batch and spatial dims, and then stats collected
    across the channels.

    In other words, each physical unit with its unique set of weights is treated as a single atomic component
    that is re-used multiple times across batch and spatial dims. Stats are then collected against that atomic
    component in terms of how often it is active (non-zero output) vs inactive (zero output).

    Args:
      model: model to examine
        For efficiency, the model can be pre-prepared with all layers set as model outputs,
        and setting 'extract_layers=False'
      dataset: assumed to be of form (X, Y), and must already be setup with appropriate batching
      include_channel_activity: bool
        Whether to additionally include per-layer raw activity rates across the channel dim.
      include_spatial_activity: bool
        Whether to additionally include per-layer raw activity rates across the spatial dims (eg: height, width).
        Layers without spatial dims just get scalar values for this.
      verbose: int, default: 0, extent of progress reporting
        0 = silent
        1 = show progress bar

    Keyword args:
      extract_layers: default=True
        Whether to extract layers from the model or to use the current model outputs as is.

    Returns:
      (model_stats, layer_stats, layer_channel_activity, layer_spatial_activity), where:
        model_stats = {
          'mean_dead_rate': mean dead rate across layers
          'min_dead_rate': min dead rate across layers
          'max_dead_rate': max dead rate across layers
          'mean_activation_rate': mean activate rate across layers
          'min_activation_rate': min activate rate across layers
          'max_activation_rate': max activate rate across layers
        }
        layer_stats = list with stats per layer: {
          'dead_rate': fraction of channels that always produce zero outputs regardless of input
          'activation_rate': mean fraction of channels that produce non-zero outputs for any given input
        }
        layer_channel_activity = list, with tensor for each layer, of shape `(channels,)`.
        layer_spatial_activity = list, with tensor for each layer, of shape `(..spatial_dims..)`
          or scaler if no spatial dims. Omitted unless include_spatial_activity is set True.

    """

    # prepare model
    extract_layers = kwargs.get('extract_layers', True)
    if extract_layers:
        inputs = _original_inputs(model)
        monitoring_model = tf.keras.Model(inputs=inputs, outputs=[layer.output for layer in model.layers])
    else:
        monitoring_model = model

    # init
    channel_sizes = [shape[-1] for shape in monitoring_model.output_shape]
    spatial_dims = [shape[1:-1] if len(shape) > 2 else () for shape in monitoring_model.output_shape]
    num_batches = tf.cast(dataset.cardinality(), dtype=tf.float32)
    layer_channel_activity_sums = [tf.Variable(tf.zeros(size, dtype=tf.float32)) for size in
                                   channel_sizes]  # by channel
    if include_spatial_activity:
        layer_spatial_activity_sums = [tf.Variable(tf.zeros(shape, dtype=tf.float32)) for shape in
                                       spatial_dims]  # by spatial dims
    else:
        layer_spatial_activity_sums = None

    # get raw active counts per layer across all batches in dataset
    # (we compute the mean for each batch, but sum across the batches and divide later)
    @tf.function
    def _collect_stats_outer(model_arg, dataset_arg, layer_channel_activity_sums_arg, layer_spatial_activity_sums_arg):
        for inputs, _ in dataset_arg:
            _collect_stats_inner(model_arg, inputs, layer_channel_activity_sums_arg, layer_spatial_activity_sums_arg)
        return layer_channel_activity_sums_arg, layer_spatial_activity_sums_arg

    @tf.function
    def _collect_stats_inner(model_arg, inputs_arg, layer_channel_activity_sums_arg, layer_spatial_activity_sums_arg):
        layer_outputs = model_arg(inputs=inputs_arg, training=False)
        for l_idx, layer_output in enumerate(layer_outputs):
            active_outputs = tf.cast(tf.not_equal(layer_output, 0.0), tf.float32)
            active_counts_by_channel = tf.reduce_mean(active_outputs, axis=tf.range(tf.rank(active_outputs) - 1))
            layer_channel_activity_sums_arg[l_idx].assign_add(active_counts_by_channel)
            if layer_spatial_activity_sums_arg is not None:
                active_counts_by_spatial = tf.reduce_mean(active_outputs, axis=(0, tf.rank(active_outputs) - 1))
                layer_spatial_activity_sums_arg[l_idx].assign_add(active_counts_by_spatial)
        return layer_channel_activity_sums_arg, layer_spatial_activity_sums_arg

    if verbose > 0:
        # note: can't use auto-graph for the outer loop so might be just a little bit slower (circa 13s vs 10s)
        for inputs, _ in tqdm.tqdm(dataset):
            _collect_stats_inner(monitoring_model, inputs, layer_channel_activity_sums, layer_spatial_activity_sums)
    else:
        # even dataset iteration loop is auto-graphed
        layer_channel_activity_sums, layer_spatial_activity_sums = _collect_stats_outer(
            monitoring_model, dataset, layer_channel_activity_sums, layer_spatial_activity_sums)

    # compute individual layer channel stats
    def _compute_channel_stats(channel_size, layer_active_sum):
        active_rates = layer_active_sum / num_batches
        dead_rate = tf.reduce_sum(tf.cast(tf.equal(layer_active_sum, 0.0), tf.int32)) / channel_size
        return {
            'dead_rate': dead_rate.numpy(),
            'activation_rate': tf.reduce_mean(active_rates).numpy()
        }

    layer_stats = [_compute_channel_stats(channel_size, layer_active_sum) for
                   channel_size, layer_active_sum in
                   zip(channel_sizes, layer_channel_activity_sums)]

    # collect raw layer activity rates
    layer_channel_activities = None
    if include_channel_activity:
        layer_channel_activities = [layer_active_sum / num_batches for layer_active_sum in layer_channel_activity_sums]

    layer_spatial_activities = None
    if layer_spatial_activity_sums is not None:
        layer_spatial_activities = [layer_active_sum / num_batches for layer_active_sum in layer_spatial_activity_sums]

    # compute aggregate stats across whole model
    def _compute_model_stats(layer_stats_list):
        dic = {}
        for key in ['dead_rate', 'activation_rate']:
            dic[f"min_{key}"] = min([stats[key] for stats in layer_stats_list])
            dic[f"max_{key}"] = max([stats[key] for stats in layer_stats_list])
            dic[f"mean_{key}"] = np.mean([stats[key] for stats in layer_stats_list])
        return dic

    model_stats = _compute_model_stats(layer_stats)

    # build result tuple
    res = (model_stats, layer_stats)
    if layer_channel_activities is not None:
        res += (layer_channel_activities,)
    if layer_spatial_activities is not None:
        res += (layer_spatial_activities,)
    return res


def plot_history_overview(callbacks: list, details=True, iterations=None):
    """
    Uber-plotting function that selecst the most salient attributes of the other
    plot_xxx_history() functions so that a single plot can highlight areas that need further
    investigation.

    Args:
        callbacks: list of callbacks with any or all of:
            - History
            - HistoryStats
            - VariableStatsCallback
            - GradientHistoryCallback
            - LayerOutputHistoryCallback
            - LayerOutputGradientHistoryCallback
        details: bool
            Whether to include rows of plots for each callback, or just the main overview row otherwise.
        iterations: slice, range, list, set, or other list-like
            Selection over iterations to be displayed, counted against epoch or steps, depending on what is being
            displayed. Selection method depends on type provided, which becomes important where history data
            doesn't start at iteration 0:
            - `slice(start, stop, step)` - selects by **index**, eg: slice(0, 50) for the first 50 iterations,
               whatever range they happen to be in.
            - `range(start, stop)` - selects by range of included **value**, eg: range(0, 50) for only those source
               iterations that fall within the range 0 .. 50.
            - any list-like object: Filters based on exact membership. Preserves selection order if available.
              It is an error to select iterations that are not present.
    """

    # parse arguments - extract individual callbacks by type
    history = None
    history_stats = None
    variables = None
    gradients = None
    activity = None
    output_gradients = None
    for cb in callbacks:
        if reload_safe_isinstance(cb, HistoryStats):
            history_stats = cb
        elif reload_safe_isinstance(cb, tf.keras.callbacks.History):
            history = cb
        elif reload_safe_isinstance(cb, VariableHistoryCallback):
            variables = cb
        elif reload_safe_isinstance(cb, LayerOutputHistoryCallback):
            activity = cb
        elif reload_safe_isinstance(cb, GradientHistoryCallback):
            gradients = cb
        elif reload_safe_isinstance(cb, LayerOutputGradientHistoryCallback):
            output_gradients = cb
    history_either = history_stats if history_stats is not None else history

    # sanity check
    # - note: don't care if history is collected at a different rate to the rest
    per_steps = [cb.per_step for cb in [variables, gradients, activity, output_gradients] if cb is not None]
    per_steps = set(per_steps)
    if len(per_steps) == 0:
        per_step = False
    elif len(per_steps) == 1:
        per_step = next(iter(per_steps))
    else:
        raise ValueError("Cannot plot a mixture of per-epoch and per-step data")

    # prepare - identify model
    model = None
    for cb in [variables, gradients, activity, output_gradients]:
        if not model and cb and cb.model:
            model = cb.model
    if model is None:
        raise ValueError("None of the callbacks have a model set")

    # prepare - iteration list
    # - must be right length for main callbacks
    # - but prefer list from history_stats if present and the right length
    # - then apply filter
    iteration_name = 'step' if per_step else 'epoch'
    src_it_len = None
    for cb in [variables, gradients, activity, output_gradients]:
        if cb and cb.collected_value_stats is not None:
            src_it_len = len(cb.collected_value_stats[0])
            break
        elif cb and cb.collected_activity_stats is not None:
            src_it_len = len(cb.collected_activity_stats[0])
            break
    if not len:
        raise ValueError("None of the callbacks seem to have iteration information")

    src_iterations = None
    if per_step and history_stats and history_stats.steps is not None:
        src_iterations = history_stats.steps
    elif not per_step and history_either is not None:
        src_iterations = history_either.epoch
    if src_iterations is None or len(src_iterations) != src_it_len:
        src_iterations = list(range(src_it_len))

    iterations, iteration_indices = _filter_iterations(src_iterations, iterations, return_indices=True)

    # determine plot size
    # - two plots across top, 3-grid-cols each
    # - three plots on each subsequent row, 2 grid-cols each
    # - total 6 grid-cols
    plot_rows = 1
    has_activity_stats = False
    for cb in [variables, gradients, activity, output_gradients]:
        if details and cb is not None:
            plot_rows += 1
        if cb is not None and cb.activity_stats is not None:
            has_activity_stats = True
    grid_width = 6
    grid_height = 2 + (plot_rows - 1)
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # Main plot - Loss
    # - uses per_step as a guide, but only plots what's available
    if history_stats or history:
        plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
        plt.title("Loss and Metrics")
        keys = history_stats.history.keys() if history_stats else history.history.keys()
        hist_per_step = per_step and history_stats is not None and history_stats.step_history is not None
        if hist_per_step == per_step:
            hist_iterations = iterations
            hist_iteration_indices = iteration_indices
        elif hist_per_step:
            hist_iterations = history_stats.steps
            hist_iteration_indices = list(range(len(hist_iterations)))
        else:
            hist_iterations = history_either.epoch
            hist_iteration_indices = list(range(len(hist_iterations)))
        for s_idx, key in enumerate(keys):
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][s_idx]
            if hist_per_step:
                plt.plot(
                    hist_iterations,
                    np.array(history_stats.step_history[key])[hist_iteration_indices],
                    label=key, color=color)
            elif key == 'loss' and history_stats:
                _plot_add_quantiles(
                    hist_iterations,
                    history_stats.epoch_stats[key].iloc[hist_iteration_indices],
                    label=key, show_percentile_labels=False, single_series=False, color=color)
            else:
                plt.plot(hist_iterations, np.array(history_either.history[key])[iteration_indices],
                         label=key, color=color)
        plt.margins(0)
        plt.yscale('log')
        plt.xlabel('step' if hist_per_step else 'epoch')
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # ensure integer tick marks
        plt.legend()

    # Main plot - Warnings
    if has_activity_stats:
        plt.subplot2grid((grid_height, grid_width), (0, grid_width // 2), colspan=grid_width // 2, rowspan=2)
        plt.title("Warnings")
        for cb in [variables, gradients, activity, output_gradients]:
            if cb is not None and cb.model_activity_stats is not None:
                item_name = cb.item_name
                plt.plot(iterations, np.array(cb.model_activity_stats['max_dead_rate'])[iteration_indices],
                         label=f"Worst {item_name.lower()} dead rate")
        plt.margins(0)
        plt.ylim([0.0, 1.1])
        plt.xlabel(iteration_name)
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.legend()

    # Per-callback plot rows
    cb_idx = -1
    for cb in [variables, gradients, activity, output_gradients]:
        if details and cb is not None:
            cb_idx += 1
            item_type = cb.item_type
            item_name = cb.item_name
            
            # Column plot - Value range
            if cb.model_norm_stats is not None:
                plt.subplot2grid((grid_height, grid_width), (2 + cb_idx, 0), colspan=2)
                plt.title(f"All model {item_name.lower()}s")
                _plot_add_quantiles(iterations, cb.model_norm_stats.iloc[iteration_indices])
                plt.margins(0)
                plt.yscale('log')
                plt.xlabel(iteration_name)
                plt.ylabel('norm')
                plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                plt.legend()

            # Column plot - Layer comparison
            if cb.value_norms is not None:
                plt.subplot2grid((grid_height, grid_width), (2 + cb_idx, 2), colspan=2)
                filtered_scales = [norms[iteration_indices] for norms in cb.collected_value_norms]
                _plot_layer_scale_comparison(
                    model, item_type, filtered_scales, cb.collected_value_stats_indices, iterations,
                    xlabel=iteration_name, ylabel="relative log-norm")

            # Column plot - Activity rates
            if cb.activity_stats is not None:
                plt.subplot2grid((grid_height, grid_width), (2 + cb_idx, 4), colspan=2)
                plt.title(f"{item_name} activation rates")
                plt.plot(iterations,
                         np.array(cb.model_activity_stats['mean_activation_rate'])[iteration_indices],
                         color='tab:blue', label='mean')
                plt.fill_between(iterations,
                                 np.array(cb.model_activity_stats['min_activation_rate'])[iteration_indices],
                                 np.array(cb.model_activity_stats['max_activation_rate'])[iteration_indices],
                                 color='tab:blue', alpha=0.2)
                plt.plot(iterations,
                         np.array(cb.model_activity_stats['max_dead_rate'])[iteration_indices],
                         color='tab:red', label='worst dead rate')
                plt.margins(0)
                plt.ylim([0.0, 1.1])
                plt.xlabel(iteration_name)
                plt.ylabel("fraction of units")
                plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                plt.legend()


def plot_train_history(callback: tf.keras.callbacks.History, per_step=False, show_loss_percentiles=True,
                       show_metric_percentiles=True):
    """
    Plots the loss and metric curves from a training run as collected by the standard `History` callback
    or the extended `HistoryStats` callback.
    Works with aggregate histories that have been built up over multiple model.fit() calls.

    Flexibly supports different losses and metrics.
    Applies some heuristics to group the losses together onto a log plot, and the other metrics
    on a linear plot.

    Args:
        callback: a history or history stats callback populated with data from training.
        per_step: whether to plot data from callback.step_history instead of callback.epoch_stats.
        show_loss_percentiles: whether to include full percentile information for losses and loss-like metrics,
          or just a single value otherwise. Disabling percentile plotting can be necessary if there are
          many loss-like metrics.
        show_metric_percentiles: whether to include full percentile information for metrics,
          or just a single value otherwise. Disabling percentile plotting can be necessary if there are
          many metrics.
    """
    # sanity checks
    if per_step and not hasattr(callback, 'step_history'):
        raise ValueError("History callback cannot be used to plot per_step data")
    if per_step and callback.step_history is None:
        raise ValueError("HistoryStats callback did not collect per_step data")

    # identify "losses" vs other metrics
    # - the main distinction needed here is between those that need to be on a log scale because they get progressively
    #   closer to zero, and those that tend to remain within the scale of 0.0 to 1.0.
    def is_loss(k):
        return 'loss' in k or 'entropy' in k
    loss_keys = [k for k in callback.history.keys() if is_loss(k)]
    metric_keys = [k for k in callback.history.keys() if not is_loss(k)]

    # prepare
    iterations = callback.steps if per_step else callback.epoch

    # do plots
    plt.figure(figsize=(11, 3))
    if len(loss_keys) > 0:
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        for s_idx, key in enumerate(loss_keys):
            if per_step:
                plt.plot(iterations, callback.step_history[key], label=key)
            elif show_loss_percentiles and hasattr(callback, 'epoch_stats'):
                _plot_add_quantiles(iterations, callback.epoch_stats[key],
                                    color=s_idx, label=key, show_percentile_labels=False, single_series=False)
            else:
                plt.plot(iterations, callback.history[key], label=key)
        plt.yscale('log')
        plt.xlabel('step' if per_step else 'epoch')
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # ensure integer x-axis ticks
        plt.legend()

    if len(metric_keys) > 0:
        plt.subplot(1, 2, 2)
        plt.title("Metrics")
        for s_idx, key in enumerate(metric_keys):
            if per_step:
                plt.plot(iterations, callback.step_history[key], label=key)
            elif show_metric_percentiles and hasattr(callback, 'epoch_stats'):
                _plot_add_quantiles(iterations, callback.epoch_stats[key],
                                    color=s_idx, label=key, show_percentile_labels=False, single_series=False)
            else:
                plt.plot(iterations, callback.history[key], label=key)
        plt.xlabel('step' if per_step else 'epoch')
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # ensure integer x-axis ticks
        plt.legend()

    plt.show()


def plot_value_history(callback: ValueStatsCollectingMixin, show='magnitudes', iterations=None):
    """
    Generates a figure containing a number of plots to visualise value or magnitude stats collected
    by one of the callbacks in this module.

    Uses a log plot for magnitudes, and a linear plot for raw values.

    Args:
        callback: any of "value stats collecting" callbacks in this module
        show: one of 'magnitudes' (default), 'values', 'norms'.
        iterations: slice, range, list, set, or other list-like
            Selection over iterations to be displayed, counted against epoch or steps, depending on what is being
            displayed. Selection method depends on type provided, which becomes important where history data
            doesn't start at iteration 0:
            - `slice(start, stop, step)` - selects by **index**, eg: slice(0, 50) for the first 50 iterations,
               whatever range they happen to be in.
            - `range(start, stop)` - selects by range of included **value**, eg: range(0, 50) for only those source
               iterations that fall within the range 0 .. 50.
            - any list-like object: Filters based on exact membership. Preserves selection order if available.
              It is an error to select iterations that are not present.
    """
    # sanity checks
    item_type = callback.item_type if hasattr(callback, 'item_type') else None
    if show not in ('magnitudes', 'values', 'norms'):
        raise ValueError(f"Invalid value for show: '{show}'")
    if item_type is None or item_type.value not in (ItemType.VARIABLE.value, ItemType.LAYER.value):  # reload-safe
        raise ValueError(f"Callback collects unsupported item type: {item_type}")
    if callback.value_norms is None:
        raise ValueError(f"{type(callback).__name__} did not collect value norms")
    if callback.magnitude_stats is None:
        raise ValueError(f"{type(callback).__name__} did not collect value magnitude stats")
    if callback.value_stats is None:
        raise ValueError(f"{type(callback).__name__} did not collect value stats")

    # collect data
    model = callback.model
    model_stats = callback.model_norm_stats
    item_type = callback.item_type
    collected_item_indices = callback.collected_value_stats_indices
    num_items = len(collected_item_indices)

    # Deal with callback differences
    item_name = callback.item_name.lower()
    title = f"All model {item_name}s"
    if hasattr(callback, 'trainable_only'):
        title = f"All model trainable {item_name}s" if callback.trainable_only else \
            f"All model {item_name}s (incl. non-trainable)"
    if hasattr(callback, 'before_updates'):
        title += ("\n(before updates)" if callback.before_updates else "\n(after updates)")

    # Prepare x-axis iterations
    # - and apply filtering
    iteration_name = 'epoch' if hasattr(callback, 'epochs') else 'step'
    src_iterations = callback.epochs if hasattr(callback, 'epochs') else callback.steps
    iterations, iteration_indices = _filter_iterations(src_iterations, iterations, return_indices=True)
    model_stats = model_stats.iloc[iteration_indices]
    collected_item_value_norms = [norms[iteration_indices] for norms in callback.collected_value_norms]
    collected_item_value_stats = [stat.iloc[iteration_indices] for stat in callback.collected_value_stats]
    collected_item_magnitude_stats = [stat.iloc[iteration_indices] for stat in callback.collected_magnitude_stats]

    # Prepare for layer mode
    item_display_names = []
    if item_type.value == ItemType.LAYER.value:  # reload-safe
        item_shapes = [callback.layer_shapes[i_idx] for i_idx in collected_item_indices]
        for l_idx in collected_item_indices:
            layer_name = model.layers[l_idx].name
            item_display_names.append(f"layer {l_idx}:\n{layer_name}")
    else:
        item_shapes = [model.variables[v_idx].shape for v_idx in collected_item_indices]
        layer_id_lookup = layer_indices_by_variable(model)
        for v_idx in collected_item_indices:
            l_idx = layer_id_lookup[v_idx]
            layer_name = model.layers[l_idx].name
            variable_name = model.variables[v_idx].name
            item_display_names.append(f"{layer_name}(#{l_idx})/{variable_name}")

    # start figure
    # - at least 4 layer plots wide
    # - otherwise target a square grid of layer plots
    grid_width = min(6, max(4, round(math.sqrt(num_items) / 2) * 2))  # nearest even number >= 4 and <= 6
    grid_height = 2 + math.ceil(num_items / grid_width)
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # all-model high-level summary
    plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
    _plot_add_quantiles(iterations, model_stats)
    plt.margins(0)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(iteration_name)
    plt.ylabel(f"norm (size-normalized)")
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # ensure integer x-axis ticks
    plt.legend()

    # layer contributions - high-level summary
    plt.subplot2grid((grid_height, grid_width), (0, grid_width // 2), colspan=grid_width // 2, rowspan=2)
    _plot_layer_scale_comparison(model, item_type, collected_item_value_norms, collected_item_indices, iterations)

    # individual layers or variables
    for i_idx in range(num_items):
        r = 2 + i_idx // grid_width
        c = i_idx % grid_width
        plt.subplot2grid((grid_height, grid_width), (r, c))
        plt.title(item_display_names[i_idx])
        plt.margins(0)
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if show == 'magnitudes':
            _plot_add_quantiles(iterations, collected_item_magnitude_stats[i_idx])
            yscale = 'log'
            ylabel = 'magnitude'
        elif show == 'values':
            _plot_add_quantiles(iterations, collected_item_value_stats[i_idx])
            yscale = 'linear'
            ylabel = 'value'
        else:
            plt.plot(collected_item_value_norms[i_idx])
            yscale = 'log'
            ylabel = 'norm'
        plt.yscale(yscale)
        if c == 0:
            plt.ylabel(ylabel)

        # add "pos/neg balance" information
        ax1 = plt.gca()
        if show == 'magnitudes':
            balances = pos_neg_balance(collected_item_value_stats[i_idx])
            ax2 = ax1.twinx()
            ax2.set_ylim([-1.0, +1.0])
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax2.plot(balances, color="tab:orange", linewidth=1, alpha=0.3)
            # cool but too distracting:
            #  - ax2.fill_between(iterations, balances, 0, where=(balances > 0), color="tab:orange", alpha=0.1)
            #  - ax2.fill_between(iterations, balances, 0, where=(balances < 0), color="tab:orange", alpha=0.1)
            if c == (grid_width-1):
                ax2.set_ylabel('pos/neg balance', color="tab:orange")
            else:
                ax2.yaxis.set_visible(False)

        # text overlay
        plot_width = np.max(iterations)
        plot_range = np.array(ax1.get_ylim())
        plot_mid = np.exp(np.mean(np.log(plot_range))) if yscale == 'log' else np.mean(plot_range)
        if item_shapes:
            ax1.text(plot_width * 0.5, plot_mid,
                     f"{item_shapes[i_idx]}",
                     horizontalalignment='center', verticalalignment='center')

    plt.show()


def plot_activity_history(callback: ActivityStatsCollectingMixin, iterations=None):
    """
    Plots a high-level summary of unit activity rates across the entire model
    and across each layer.

    Args:
        callback: any of "activity stats collecting" callbacks in this module
        iterations: slice, range, list, set, or other list-like
            Selection over iterations to be displayed, counted against epoch or steps, depending on what is being
            displayed. Selection method depends on type provided, which becomes important where history data
            doesn't start at iteration 0:
            - `slice(start, stop, step)` - selects by **index**, eg: slice(0, 50) for the first 50 iterations,
               whatever range they happen to be in.
            - `range(start, stop)` - selects by range of included **value**, eg: range(0, 50) for only those source
               iterations that fall within the range 0 .. 50.
            - any list-like object: Filters based on exact membership. Preserves selection order if available.
              It is an error to select iterations that are not present.

    Generated figure is of form:
    - top row (two columns):
        - "Unit activation rates across layers"
        - "Dead unit rates across layers"
    - remaining rows (arranged approximately as a square grid)
        - per-layer unit activation rates
    """
    # sanity checks
    item_type = callback.item_type if hasattr(callback, 'item_type') else None
    if item_type is None or item_type.value not in (ItemType.VARIABLE.value, ItemType.LAYER.value):  # reload-safe
        raise ValueError(f"Callback collects unsupported item type: {item_type}")
    if callback.activity_stats is None:
        raise ValueError(f"{type(callback).__name__} did not collect activity stats")

    # collect data
    model = callback.model
    model_stats = callback.model_activity_stats
    collected_activity_stats = callback.collected_activity_stats
    collected_item_indices = callback.collected_activity_stats_indices
    num_items = len(collected_activity_stats)

    # Deal with callback differences
    item_name_upper = callback.item_name
    item_type_name = 'layers' if item_type.value == ItemType.LAYER.value else 'variables'

    # Prepare x-axis iterations
    # - and apply filtering
    iteration_name = 'epoch' if hasattr(callback, 'epochs') else 'step'
    src_iterations = callback.epochs if hasattr(callback, 'epochs') else callback.steps
    iterations, iteration_indices = _filter_iterations(src_iterations, iterations, return_indices=True)
    model_stats = model_stats.iloc[iteration_indices]
    collected_activity_stats = [stat.iloc[iteration_indices] for stat in collected_activity_stats]

    # Prepare for layer mode
    item_display_names = []
    if item_type.value == ItemType.LAYER.value:  # reload-safe
        item_shapes = [callback.layer_shapes[i_idx] for i_idx in collected_item_indices]
        spatial_shapes = [shape[1:-1] for shape in item_shapes]  # assume: (batch, ..spatial_dims.., channels)
        for l_idx in collected_item_indices:
            layer_name = model.layers[l_idx].name
            item_display_names.append(f"layer {l_idx}:\n{layer_name}")
    else:
        item_shapes = [model.variables[v_idx].shape for v_idx in collected_item_indices]
        spatial_shapes = [shape[0:-1] for shape in item_shapes]  # assume: (..spatial_dims.., channels)
        layer_id_lookup = layer_indices_by_variable(model)
        for v_idx in collected_item_indices:
            l_idx = layer_id_lookup[v_idx]
            layer_name = model.layers[l_idx].name
            variable_name = model.variables[v_idx].name
            item_display_names.append(f"{layer_name}(#{l_idx})/{variable_name}")
    has_spatial_shapes = any([len(shape) > 0 for shape in spatial_shapes])

    # start figure
    # - at least 4 layer plots wide
    # - otherwise target a square grid of layer plots
    grid_width = min(6, max(4, round(math.sqrt(num_items) / 2) * 2))  # nearest even number >= 4 and <= 6
    grid_height = 2 + math.ceil(num_items / grid_width)
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # all-model high-level summary
    plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
    plt.title(f"{item_name_upper} unit activation rates over all {item_type_name}")
    plt.plot(iterations, model_stats['mean_activation_rate'], label='mean activation rate',
             color='tab:blue')
    plt.fill_between(iterations, model_stats['min_activation_rate'],
                     model_stats['max_activation_rate'], color='tab:blue', alpha=0.2,
                     label='min/max range')
    plt.ylim([0.0, 1.1])
    plt.xlabel(iteration_name)
    plt.ylabel('fraction of units')
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # ensure integer x-axis ticks
    plt.legend()

    # Death rate plot
    plt.subplot2grid((grid_height, grid_width), (0, grid_width // 2), colspan=grid_width // 2, rowspan=2)
    plt.title(f"Dead unit rates over all {item_type_name}")
    plt.xlabel(iteration_name)
    plt.ylabel('fraction of units')
    plt.ylim([0.0, 1.1])
    plt.plot(iterations, model_stats['mean_dead_rate'], label='mean dead rate', color='tab:red')
    plt.fill_between(iterations, model_stats['min_dead_rate'], model_stats['max_dead_rate'],
                     color='tab:red', alpha=0.2, label='min/max range')
    if has_spatial_shapes:
        plt.plot(iterations, model_stats['mean_spatial_dead_rate'], label='mean spatial dead rate', color='tab:orange')
        plt.fill_between(iterations, model_stats['min_spatial_dead_rate'], model_stats['max_spatial_dead_rate'],
                         color='tab:orange', alpha=0.2, label='min/max dead rate')
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.legend()

    # individual items
    for i_idx in range(num_items):
        activation_rates = collected_activity_stats[i_idx]['activation_rate'].to_numpy()
        dead_rates = collected_activity_stats[i_idx]['dead_rate'].to_numpy()
        spatial_dead_rates = collected_activity_stats[i_idx]['spatial_dead_rate'].to_numpy()
        final_dead_rate = dead_rates[-1]
        final_spatial_dead_rate = spatial_dead_rates[-1]

        r = 2 + i_idx // grid_width
        c = i_idx % grid_width
        plt.subplot2grid((grid_height, grid_width), (r, c))
        plt.title(item_display_names[i_idx])
        plt.plot(iterations, activation_rates, label='activation rates', color='tab:blue')
        plt.fill_between(iterations, 0, activation_rates, color='tab:blue', alpha=0.2)
        plt.plot(iterations, dead_rates, label='dead units', color='tab:red')
        plt.fill_between(iterations, 0, dead_rates, color='tab:red', alpha=0.2)
        if len(spatial_shapes[i_idx]) > 0 or (i_idx == 0 and has_spatial_shapes):
            # only include if the layer has spatial dims,
            # and include in first plot because it's the only one with a legend
            plt.plot(iterations, spatial_dead_rates, label='spatial dead units', color='tab:orange', alpha=0.8)
        plt.ylim([0.0, 1.0])
        plt.margins(0)
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # text overlay
        plot_width = np.max(iterations)
        if len(spatial_shapes[i_idx]) > 0:
            plt.text(plot_width * 0.5, 0.5,
                     f"{item_shapes[i_idx]}\n"
                     f"{final_dead_rate * 100:.1f}% dead channels\n"
                     f"{final_spatial_dead_rate * 100:.1f}% dead spatial\n",
                     horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(plot_width * 0.5, 0.5,
                     f"{item_shapes[i_idx]}\n"
                     f"{final_dead_rate * 100:.1f}% dead",
                     horizontalalignment='center', verticalalignment='center')

    plt.show()


def plot_lr_history(callback: LearningRateHistoryCallback, show='auto', iterations=None):
    """
    Generates a figure containing a number of plots to visualise the collected learning rate history
    information during training.

    Args:
        callback: callback populated with data from training
        show: one of 'auto' (default), 'values', 'norms'. Where:
            'auto' - plots value stats if available, otherwise norms.
            'values' - implicit LR stats
            'norms' - implicit LR norms
        iterations: slice, range, list, set, or other list-like
            Selection over iterations to be displayed, counted against epoch or steps, depending on what is being
            displayed. Selection method depends on type provided, which becomes important where history data
            doesn't start at iteration 0:
            - `slice(start, stop, step)` - selects by **index**, eg: slice(0, 50) for the first 50 iterations,
               whatever range they happen to be in.
            - `range(start, stop)` - selects by range of included **value**, eg: range(0, 50) for only those source
               iterations that fall within the range 0 .. 50.
            - any list-like object: Filters based on exact membership. Preserves selection order if available.
              It is an error to select iterations that are not present.
    """
    # sanity checks
    if show not in ('auto', 'values', 'norms'):
        raise ValueError(f"Invalid value for show: '{show}'")
    if show == 'auto':
        show = 'values' if callback.ilr_stats is not None else 'norms'
    if show == 'values' and callback.ilr_stats is None:
        raise ValueError(f"Callback did not collect implicit learning rate stats")

    # collect data
    model = callback.model
    num_items = len(callback.ilr_norms)

    # Prepare x-axis iterations
    # - and apply filtering
    iteration_name = 'epoch' if hasattr(callback, 'epochs') else 'step'
    src_iterations = list(range(len(callback.ilr_norms[0])))
    iterations, iteration_indices = _filter_iterations(src_iterations, iterations, return_indices=True)
    if show == 'values':
        model_stats = callback.model_stats.iloc[iteration_indices]
    else:
        model_stats = callback.model_norm_stats.iloc[iteration_indices]
    collected_item_value_norms = [norms[iteration_indices] for norms in callback.ilr_norms]
    collected_item_value_stats = [stat.iloc[iteration_indices] for stat in callback.ilr_stats]
    collected_item_indices = trainable_variable_indices_to_variable_indices(model)

    # Prepare display info
    item_display_names = []
    item_shapes = [model.variables[v_idx].shape for v_idx in collected_item_indices]
    layer_id_lookup = layer_indices_by_variable(model)
    for v_idx in collected_item_indices:
        l_idx = layer_id_lookup[v_idx]
        layer_name = model.layers[l_idx].name
        variable_name = model.variables[v_idx].name
        item_display_names.append(f"{layer_name}(#{l_idx})/{variable_name}")

    # start figure
    # - at least 4 layer plots wide
    # - otherwise target a square grid of layer plots
    grid_width = min(6, max(4, round(math.sqrt(num_items) / 2) * 2))  # nearest even number >= 4 and <= 6
    grid_height = 2 + math.ceil(num_items / grid_width)
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # all-model high-level summary
    plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
    _plot_add_quantiles(iterations, model_stats)
    plt.title("All model implicit learning rates")
    plt.margins(0)
    plt.yscale('log')
    plt.xlabel(iteration_name)
    plt.ylabel("median" if show == 'values' else "norm (size-normalized)")
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # ensure integer x-axis ticks
    plt.legend()

    # individual layers or variables
    for i_idx in range(num_items):
        r = 2 + i_idx // grid_width
        c = i_idx % grid_width
        plt.subplot2grid((grid_height, grid_width), (r, c))
        plt.title(item_display_names[i_idx])
        plt.margins(0)
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if show == 'values':
            _plot_add_quantiles(iterations, collected_item_value_stats[i_idx])
            yscale = 'log'
            ylabel = 'value'
        else:
            plt.plot(collected_item_value_norms[i_idx])
            yscale = 'log'
            ylabel = 'norm'
        plt.yscale(yscale)
        if c == 0:
            plt.ylabel(ylabel)

        # text overlay
        plot_width = np.max(iterations)
        plot_range = np.array(plt.gca().get_ylim())
        plot_mid = np.exp(np.mean(np.log(plot_range))) if yscale == 'log' else np.mean(plot_range)
        if item_shapes:
            plt.text(plot_width * 0.5, plot_mid,
                     f"{item_shapes[i_idx]}",
                     horizontalalignment='center', verticalalignment='center')

    plt.show()


def plot_channel_stats(layer_channel_activity, model=None):
    """
    Simple grid plot of per-channel unit activitation rates across
    the different layers.

    Args:
        layer_channel_activity:
            as collected from measure_unit_activity()
        model:
            Optionally pass this to add layer names.

    Example:
    >>> _, _, layer_channel_activity = measure_unit_activity(model, dataset, include_channel_activity=True)
    >>> plot_channel_stats(layer_channel_activity, model)
    """
    num_layers = len(layer_channel_activity)

    # start figure
    # - at least 4 plots wide
    # - each layer has two plots, arranged vertically
    # - otherwise target a square grid of layer plots
    grid_width = min(6, max(4, round(math.sqrt(num_layers))))  # 4 <= width <= 6
    grid_height = math.ceil(num_layers / grid_width)
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # two plots for each layer
    for l_idx, activation_rates in enumerate(layer_channel_activity):
        r = (l_idx // grid_width)
        c = l_idx % grid_width

        # flatten to 1D if necessary and collect some stats
        activation_rates = activation_rates.numpy().flatten()
        len_active_rate = np.size(activation_rates)
        mean_active_rate = np.mean(activation_rates)
        min_active_rate = np.min(activation_rates)
        max_active_rate = np.max(activation_rates)
        dead_rate = np.mean(tf.cast(tf.equal(activation_rates, 0.0), tf.float32))

        plt.subplot2grid((grid_height, grid_width), (r, c))
        plt.title(f"layer {l_idx}:\n{model.layers[l_idx].name}" if model is not None else f"layer {l_idx}")
        plt.xlim([0.0, 1.0])
        plt.yticks([])
        plt.xticks([0.0, 0.5, 1.0])
        if r == 0:
            plt.xlabel('activation rate')
        if c == 0:
            plt.ylabel('histogram')
        hist_vals, _, _ = plt.hist(activation_rates, bins=np.arange(0, 1.1, 0.1))

        # text overlay
        plot_height = np.max(hist_vals)
        text_col = 'black'
        if 0.0 < dead_rate < 1.0:
            text_col = 'tab:orange'
        elif dead_rate == 1.0:
            text_col = 'tab:red'
        plt.text(0.5, plot_height * 0.5,
                 f"{len_active_rate} channels\n"
                 f"mean {mean_active_rate * 100:.1f}%\n"
                 f"min {min_active_rate * 100:.1f}%\n"
                 f"max {max_active_rate * 100:.1f}%\n"
                 f"dead {dead_rate * 100:.1f}%",
                 color=text_col, horizontalalignment='center', verticalalignment='center')
    plt.show()


def plot_spatial_stats(layer_spatial_activity, model=None):
    """
    Simple grid plot of spatially-arrange unit activitation rates across
    the different layers.

    Args:
        layer_spatial_activity:
            as collected from measure_unit_activity()
        model:
            Optionally pass this to add layer names.

    Example:
    >>> _, _, layer_spatial_activity = measure_unit_activity(model, dataset, include_spatial_activity=True)
    >>> plot_spatial_stats(layer_spatial_activity, model)
    """
    num_layers = len(layer_spatial_activity)

    # start figure
    # - at least 4 plots wide
    # - each layer has two plots, arranged vertically
    # - otherwise target a square grid of layer plots
    grid_width = max(9, max(4, round(math.sqrt(num_layers * 2))))  # 4 <= width <= 9
    grid_height = math.ceil(num_layers / grid_width) * 2
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # two plots for each layer
    for l_idx, activation_rates in enumerate(layer_spatial_activity):
        r = (l_idx // grid_width) * 2
        c = l_idx % grid_width

        # flatten to 2D if needed and collect stats
        if tf.rank(activation_rates) >= 2:
            activation_rates = tf.reduce_mean(activation_rates, axis=range(2, tf.rank(activation_rates)))
        alive_units = tf.cast(tf.not_equal(activation_rates, 0.0), tf.float32)
        dead_rate = np.mean(tf.cast(tf.equal(activation_rates, 0.0), tf.float32))
        mean_active_rate = np.mean(activation_rates)
        min_active_rate = np.min(activation_rates)
        max_active_rate = np.max(activation_rates)
        plot_shape = (activation_rates.shape[0] - 1, activation_rates.shape[1] - 1) if tf.rank(
            activation_rates) >= 2 else (1, 1)

        # top plot
        plt.subplot2grid((grid_height, grid_width), (r, c))
        plt.title(f"layer {l_idx}:\n{model.layers[l_idx].name}" if model is not None else f"layer {l_idx}")
        plt.xticks([])
        plt.yticks([])
        if c == 0:
            plt.ylabel('activations')
        if tf.rank(activation_rates) >= 2:
            plt.imshow(activation_rates, vmin=0.0)
        plt.text(plot_shape[1] * 0.5, plot_shape[0] * 0.5,
                 f"mean {mean_active_rate * 100:.1f}%\n"
                 f"min {min_active_rate * 100:.1f}%\n"
                 f"max {max_active_rate * 100:.1f}%",
                 horizontalalignment='center', verticalalignment='center')

        # bottom plot
        plt.subplot2grid((grid_height, grid_width), (r + 1, c))
        plt.xticks([])
        plt.yticks([])
        if c == 0:
            plt.ylabel('alive outputs')
        if tf.rank(activation_rates) >= 2:
            plt.imshow(alive_units, cmap='gray', vmin=0.0, vmax=1.0)
        text_col = 'black'
        if 0.0 < dead_rate < 1.0:
            text_col = 'tab:orange'
        elif dead_rate == 1.0:
            text_col = 'tab:red'
        plt.text(plot_shape[1] * 0.5, plot_shape[0] * 0.5,
                 f"dead rate\n{dead_rate * 100:.1f}%",
                 color=text_col, horizontalalignment='center', verticalalignment='center')
    plt.show()


def _plot_add_quantiles(x, data, label=None, color="tab:blue", show_percentile_labels=True, single_series=True):
    """
    Adds multi-quantile data to an existing plot.
    Useful for displaying stats returned by the history callbacks.
    Args:
        x: list-like. X-axis values.
        data: pandas Dataframe with columns corresponding to quantiles, labeled in range 0 to 100.
        color: int or string.
          Series color. Provide a normal color string. Or, provide an int and it'll pick that indexed color from the normal
          color list.
        label: data series name, or None to show only percentile labels (if any)
        show_percentile_labels: whether to show percentile information in labels
        single_series: whether to enable visual optimisations that only work for single series. Including:
            - min/max are shown in grey instead of a faint version of the colour
    """
    # handle arguments
    label_prefix = f"{label} " if label else ""
    if isinstance(color, int):
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][color]

    def _label(q1, q2):
        if q2 is None:
            if show_percentile_labels:
                return f"{label_prefix}median" if q1 == 50 else f"{label_prefix}{q1}%"
            else:
                return label
        elif q1 == 0 and q2 == 100:
            return f"{label_prefix}min/max" if show_percentile_labels else None
        elif 100 - q1 == q2:
            return f"{label_prefix}±{q1}%" if show_percentile_labels else None
        else:
            return f"{label_prefix}{q1}% to {q2}%" if show_percentile_labels else None

    quantiles = data.columns
    quantile_len = len(quantiles)
    bot, top = 0, quantile_len - 1
    while bot < top:
        display_color = 'tab:grey' if single_series and quantiles[bot] == 0 and quantiles[top] == 100 else color
        plt.fill_between(x, data[quantiles[bot]], data[quantiles[top]],
                         alpha=0.2, color=display_color, linewidth=0,
                         label=_label(quantiles[bot], quantiles[top]))
        bot += 1
        top -= 1
    if bot == top:
        plt.plot(x, data[quantiles[bot]],
                 color=color,
                 label=_label(quantiles[bot], None))


def _plot_layer_scale_comparison(model, item_type, item_scales, item_indices,
                                 iterations,
                                 title="Layer comparison",
                                 xlabel="Iteration",
                                 ylabel="relative log-norm"):
    """
    Adds a "layer comparison" chart to an existing figure.
    For easier visual display in variables mode, uses only the largest variable from each layer.
    Args:
        model: the model
        item_type: layer or variable
        item_scales: list (by item) of 1D-arrays (by iteration) indicating the "scale" of each collected
            item across iterations. Typically individual values are some sort of norm of the original tensor.
        item_indices: layer or variable indices of collected items
        iterations: x-axis values
    """
    if item_type.value == ItemType.LAYER.value:  # reload-safe
        filtered_layer_names = [model.layers[l_idx].name for l_idx in item_indices]
        filtered_scales = item_scales
    else:
        filtered_scales, filtered_layers, _ = _pick_layer_data_from_variables(
            item_scales, item_indices, model)
        filtered_layer_names = [layer.name for layer in filtered_layers]
    filtered_scales = np.stack(filtered_scales, axis=-1)
    band_log_scales = _log_normalize(filtered_scales, axis=-1)

    plt.margins(0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().set_yticks([])  # don't show y-axis tick marks
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # ensure x-axis integer ticks
    plt.stackplot(iterations, band_log_scales.T, colors=['lightsteelblue', 'royalblue'], linewidth=0)
    # layer labels placed on mid-height of layer band on left-hand side
    sample_len = max(1, math.ceil(band_log_scales.shape[0] / 5))
    band_mid_points = np.cumsum(band_log_scales, axis=1) - band_log_scales*0.5
    placement = np.mean(band_mid_points[:sample_len, :], axis=0)
    for f_idx, layer_name in enumerate(filtered_layer_names):
        plt.text(len(iterations) / 100, placement[f_idx], layer_name, ha="left")


def _pick_layer_data_from_variables(variable_data, variable_indices, model):
    """
    Used to simplify display of many variables by focusing on the single
    biggest variable for each layer.
    Args:
        variable_data: data of any type associated with a collection of model variables
        variable_indices: indices of the variables relative to model.variables
        model: the model from which the variables and layers are collected
    Returns:
        - filtered_data - variable_data filtered to one per layer
        - filtered_layers - the model layer for each filtered data
        - filtered_layer_indices - indices relative to model.layers of layers represented
    """
    # tuples of: (i_idx of biggest variable, size of biggest variable)
    # where: i_idx is "item index" that indexes into variable_indices
    layer_metas = [(None, None)] * len(model.layers)
    layer_id_lookup = layer_indices_by_variable(model)

    # pick the largest variable for each layer
    for i_idx, v_idx in enumerate(variable_indices):
        l_idx = layer_id_lookup[v_idx]
        variable = model.variables[v_idx]
        biggest_v_idx, biggest_size = layer_metas[l_idx]
        if biggest_size is None or tf.size(variable) > biggest_size:
            layer_metas[l_idx] = i_idx, tf.size(variable)

    filtered_stats = [variable_data[i_idx] for i_idx, _ in layer_metas if i_idx is not None]
    filtered_layers = [model.layers[l_idx] for l_idx, (i_idx, _) in enumerate(layer_metas) if i_idx is not None]
    filtered_layer_indices = [l_idx for l_idx, (i_idx, _) in enumerate(layer_metas) if i_idx is not None]
    return filtered_stats, filtered_layers, filtered_layer_indices


def _filter_iterations(src_iterations, selection=None, return_indices=False):
    """
    Filters the source iterations list
    Args:
      src_iterations: list of recorded iteration numbers, which may not start at zero.
      selection (slice, range, or iterable):
        - `slice(start, stop, step)`: selects by **index**, eg: slice(0, 50) for the first 50 iterations,
           whatever range they happen to be in.
        - `range(start, stop, step)`: selects by **value**, eg: range(0, 50) for only those source iterations that
           fall within the range 0 .. 50.
        - Any iterable (list, set, generator, numpy array, etc.): Filters based on membership.
          Throws an error if any value in the selection is not found in the source.
          Preserves selection order if providing an ordered iterable, or otherwise preserves the source iterations order.
      return_indices: bool.
        Whether to additionally return the indices of the selected iterations, w.r.t src_iterations
    Returns:
      A list of the filtered iterations.
      OR
      (filtered_iterations, filtered_indices)
    """

    if selection is None:
        filtered_iterations = src_iterations
        filtered_indices = list(range(len(src_iterations)))
    elif isinstance(selection, slice):
        # select by index
        filtered_iterations = src_iterations[selection]
        filtered_indices = list(range(*selection.indices(len(src_iterations))))
    elif isinstance(selection, range):
        # select by allowed range, ignoring any selection values that aren't present
        filtered_iterations = [v for v in src_iterations if v in selection]
        filtered_indices = [idx for idx, v in enumerate(src_iterations) if v in selection]
    elif isinstance(selection, (list, tuple, set, np.ndarray)):
        # sanity check
        src_lookup = set(src_iterations)  # for speed efficiency
        unmatched = [v for v in selection if v not in src_lookup]
        if len(unmatched) > 0:
            raise ValueError(f"Iteration values not found in source: {unmatched}")

        # use as direct matching-filter int src_iterations (select by value membership)
        if isinstance(selection, set):
            # selection has no order (eg: set type) so preserve original order instead
            selection = set(selection)  # for speed efficiency
            filtered_iterations = [v for v in src_iterations if v in selection]
            filtered_indices = [idx for idx, v in enumerate(filtered_iterations) if v in selection]
        else:
            # selection has order so preserve that
            filtered_iterations = list(selection)
            filtered_indices = [src_iterations.index(v) for v in filtered_iterations]
    else:
        raise TypeError("selection must be a slice, range, or iterable")

    if return_indices:
        return filtered_iterations, filtered_indices
    else:
        return filtered_iterations


def reload_safe_isinstance(obj, cls):
    """
    Because I do a lot of `reload(module)` during development, `isinstance` tests become unreliable.
    """
    def class_isinstance(obj_cls, ref_cls):
        obj_cls_fqn = f"{obj_cls.__module__}.{obj_cls.__name__}"
        cls_fqn = f"{ref_cls.__module__}.{ref_cls.__name__}"
        if obj_cls_fqn == cls_fqn:
            return True

        # recurse through inheritance hierarchies
        for parent in obj_cls.__bases__:
            if class_isinstance(parent, ref_cls):
                return True
        return False

    if isinstance(obj, cls):
        return True
    return class_isinstance(type(obj), cls)
