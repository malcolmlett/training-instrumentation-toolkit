import tensorflow as tf
import tensorflow_probability as tfp
import keras
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

# tip: to get output shape of a layer:
#  model.layers[l].compute_output_shape(model.layers[l].input.shape)


class LessVerboseProgressLogger(tf.keras.callbacks.Callback):
    """
    Progress logger for when running models that train via thousands of very short epochs.
    By default, automatically logs progress 10 times during training.

    Use as:
    >>> model.fit(...., verbose=0, callbacks=[LessVerboseProgressLogger()])
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


# Tries to replicate keras.backend.tensorflow.TensorFlowTrainer.fit() (trainer.py, keras 3.5.0)
# as much as possible.
def fit(model, dataset, epochs=1, verbose=1, callbacks=None, initial_epoch=0):
    """
    A custom training loop mimicking model.fit() that makes raw gradient and layer output
    information available for tracking.

    Honours the state of `tf.config.run_functions_eagerly(bool)`.

    Args:
        model: usual meaning
        dataset: usual meaning, except only copes with fixed-length datasets
        epochs: usual meaning
        verbose: usual meaning
        callbacks: usual meaning plus can take instances of BaseGradientCallback
        initial_epoch: usual meaning
    Returns:
         HistoryCallback
    """
    # prepare epochs
    num_batches = len(dataset)

    # prepare callbacks tracking
    gradient_callbacks = []
    if callbacks is not None and not isinstance(callbacks, tf.keras.callbacks.CallbackList):
        gradient_callbacks = [callback for callback in callbacks if isinstance(callback, BaseGradientCallback)]
        callbacks = [callback for callback in callbacks if not isinstance(callback, BaseGradientCallback)]
    if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
        callbacks = tf.keras.callbacks.CallbackList(callbacks, add_history=True, add_progbar=verbose != 0,
                                                    verbose=verbose, epochs=epochs, steps=num_batches, model=model)
    for gradient_callback in gradient_callbacks:
        gradient_callback.set_params({'epochs': epochs, 'steps': len(dataset)})
        gradient_callback.set_model(model)

    needs_output_gradients = any(cb.needs_output_gradients for cb in gradient_callbacks)

    # prepare model for layer output collection
    # (original model output(s) will be first entry of new outputs array, it will have single tensor or list
    # accordingly)
    # FIXME this is now triggering the following warning when the model gets called during the train step.
    #  It appears to be because model.inputs returns some sort of post-processed KerasTensor
    #  but Model(inputs=...) is usually passed a less-compiled version of KerasTensor.
    #  Note: tf.keras.Input() returns KerasTensor.
    #  > /usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py:237: UserWarning: The structure of
    #  >   `inputs` doesn't match the expected structure.
    #  > Expected: ['keras_tensor_66']
    #  > Received: inputs=Tensor(shape=(32, 2))
    #  >   warnings.warn(msg)
    monitoring_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.outputs] + [layer.output for layer in model.layers])

    # prepare train function
    if tf.config.functions_run_eagerly():
        train_step_fn = _gradient_returning_train_step
    else:
        train_step_fn = tf.function(_gradient_returning_train_step)

    # latest values
    # - these are computed at each step, but we need to provide values at the end of each epoch,
    #   so we'll just hold onto the last value computed at each step
    logs = {}
    loss = None
    trainable_gradients = None
    output_gradients = None
    activations = None

    # train
    print(f"Training via custom fit() function. Will produce a few warnings; you can usually ignore these.")
    callbacks.on_train_begin()
    for gradient_callback in gradient_callbacks:
        gradient_callback.on_train_begin()
    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        for gradient_callback in gradient_callbacks:
            gradient_callback.on_epoch_begin(epoch)

        for step, data in enumerate(dataset):
            x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
            callbacks.on_train_batch_begin(step)
            for gradient_callback in gradient_callbacks:
                gradient_callback.on_train_batch_begin(step)

            loss, metrics, trainable_gradients, output_gradients, activations = train_step_fn(
                model, monitoring_model, x, y, sample_weight, needs_output_gradients)

            logs = metrics
            logs['loss'] = loss.numpy()
            callbacks.on_train_batch_end(step, logs)
            for gradient_callback in gradient_callbacks:
                gradient_callback.on_train_batch_end(
                    batch=step,
                    loss=loss,
                    gradients=trainable_gradients,
                    trainable_variables=model.trainable_variables,
                    activations=activations,
                    output_gradients=output_gradients if gradient_callback.needs_output_gradients else None)

        # end of epoch
        callbacks.on_epoch_end(epoch, logs)  # should be passing loss and mse
        for gradient_callback in gradient_callbacks:
            gradient_callback.on_epoch_end(
                epoch=epoch,
                loss=loss,
                gradients=trainable_gradients,
                trainable_variables=model.trainable_variables,
                activations=activations,
                output_gradients=output_gradients if gradient_callback.needs_output_gradients else None)
    callbacks.on_train_end(logs)
    for gradient_callback in gradient_callbacks:
        gradient_callback.on_train_end()

    return model.history


# Tries to replicate keras.backend.tensorflow.TensorFlowTrainer.train_step() (trainer.py, keras 3.5.0)
# as much as possible.
def _gradient_returning_train_step(model, monitoring_model, x, y, sample_weight, compute_output_gradients):
    """
    This method is programmatically converted via auto-graph.

    Returns:
        - loss - float. Loss returned by loss function (before optimizer scaling).
        - metrics - dict. Metrics returned by model (note: also includes a 'loss' value but it's always zero)
        - trainable_variable_gradients - list. Gradients tensor for each trainable variable.
        - output_gradients - list. Gradients tensor for each layer output, or None if not requested.
            Note that some layers will have None as the gradients tensor. eg: the last layer and layers
            that aren't involved in the final output.
        - layer_outputs - list. Raw outputs from each layer.
    """

    # Forward pass
    with tf.GradientTape() as tape:
        monitoring_outputs = monitoring_model(x)
        y_pred = monitoring_outputs[0]
        layer_outputs = monitoring_outputs[1:]

        loss = model.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=True)
        reported_loss = loss  # tracking before scaling
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

    return reported_loss, metrics, trainable_grads, output_grads, layer_outputs


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
        training doesn't use batches, or when you only want to sample the occasional update.

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

    def __init__(self, value_stats=True, value_stats_quantiles=None, *args, **kwargs):
        """
        Args:
            value_stats: bool.
                Whether to enable collection of value stats.

            value_stats_quantiles: list of percentiles in range 0 .. 100.
                Default: [0., 12.5, 25., 37.5, 50., 62.5, 75., 87.5, 100.]
        """
        super().__init__(*args, **kwargs)
        self.value_stats_enabled = value_stats
        self.value_stats_quantiles = value_stats_quantiles or [0., 12.5, 25., 37.5, 50., 62.5, 75., 87.5, 100.]

        # shape of data structure is optimised for speed of accumulation during training
        # so needs to be converted to a more usable data structure after training
        # - initially: list (by item) of list (by iteration) of tensors (by quantile)
        # - finally:   list (by item) of pandas dataframes with shape (iterations, stats)
        self._value_stats = None
        self._magnitude_stats = None

    @property
    def model_magnitude_stats(self):
        """
        Stats across the general magnitudes of values across the variables.
        Pandas data-frame with shape (iterations, percentiles).
        """
        if self._magnitude_stats is not None:
            return self._compute_scale_distribution_across_stats_list(self._magnitude_stats)
        else:
            return None

    def _init_value_stats(self, values):
        """
        Initialisation of tracking once we have a first example of what the values look like.
        Automatically called on first data collection.
        Safe to call this every iteration. Only takes effect on the first call.
        """
        if self.value_stats_enabled and self._value_stats is None:
            self._value_stats = [[] if value is not None else None for value in values]
            self._magnitude_stats = [[] if value is not None else None for value in values]

    def _finalize_value_stats(self):
        # convert data structures
        # - from: list (by item) of list (by iteration) of tensor (by stat)
        # - to:   list (by item) of pd-dataframe: iterations x percentiles
        if self._value_stats is not None:
            self._value_stats = self._stats_tensor_list_to_dataframes(
                self._value_stats, self.value_stats_quantiles)
            self._magnitude_stats = self._stats_tensor_list_to_dataframes(
                self._magnitude_stats, self.value_stats_quantiles)

    def _collect_value_stats(self, values):
        self._init_value_stats(values)
        if self._value_stats is not None:
            # compute value and magnitued percentile stats for each individual variable
            stat_pairs = self._compute_percentile_stats(values, self.value_stats_quantiles)

            # append to stats list
            # (performance note: this loop doesn't seem to cost much)
            for item_value_stats, item_magnitude_stats, (value_percentiles, magnitude_percentiles) \
                    in zip(self._value_stats, self._magnitude_stats, stat_pairs):
                if item_value_stats is not None:
                    item_value_stats.append(value_percentiles)
                if item_magnitude_stats is not None:
                    item_magnitude_stats.append(magnitude_percentiles)

    # This is fairly expensive. It could be worth optionally calculating percentiles, but by default
    # calculating simpler stats. Just need to find a way to cope with magnitude data that doesn't
    # work well for std.dev calcualtions.
    @tf.function
    def _compute_percentile_stats(self, tensors, quantiles):
        """
        Computes a set of percentiles across the raw values and magnitudes of each
        provided tensor.
        For tensors that commonly have values distributed either side of zero, the median will
        typically be around zero, and the 25th and 75th percentiles represent the respective medians
        in the positive and negative halves.
        Args:
            tensors: list of tensors for which percentiles should be calculated, some of which may be None.
            quantiles: list of quantiles to compute values for
        Returns:
            - value_percentiles - tensor of percentile values across the tensor values, None for None tensors
            - magnitude_percentile - tensor of percentile values across the tensor magnitudes, None for None tensors
        """
        def computation(tensor):
            if tensor is not None:
                value_percentiles = tfp.stats.percentile(tensor, quantiles, interpolation='linear')
                magnitude_percentiles = tfp.stats.percentile(tf.abs(tensor), quantiles, interpolation='linear')
            else:
                value_percentiles, magnitude_percentiles = None, None
            return value_percentiles, magnitude_percentiles
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

    @staticmethod
    def _compute_scale_distribution_across_stats_list(magnitude_stats_dataframes, quantiles=None):
        """
        Calculates meta-stats against a set of quantile stats.
        Assumes the use of dataframes where each column represents a quantile, represented as a number in range 0 to 100.
        Args:
            magnitude_stats_dataframes: list (by item) of pandas dataframes of shape (iterations, quantiles)
            quantiles: set of quantiles to return in final result.
        Returns:
            pandas dataframe with shape (iterations, quantiles)
        """
        quantiles = quantiles or [0, 25, 50, 75, 100]

        # collect a table of all scales across all stats
        scales = get_scales_across_stats_list(magnitude_stats_dataframes, 50)

        # calculate stats across the table
        stats = tfp.stats.percentile(scales, quantiles, axis=-1, interpolation='linear')
        stats = tf.transpose(stats)

        # return as dataframe with quantiles as columns
        return pd.DataFrame(stats.numpy(), columns=quantiles)


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
            - min_activation_rate - min value of activation_rate across all layers
            - mean_activation_rate - mean value of activation_rate across all layers
            - max_activation_rate - max value of activation_rate across all layers
            - min_dead_rate - min value of dead_rate across all layers
            - mean_dead_rate - min value of dead_rate across all layers
            - max_dead_rate - min value of dead_rate across all layers
            - min_spatial_dead_rate - min value of spatial_dead_rate across all layers
            - mean_spatial_dead_rate - min value of spatial_dead_rate across all layers
            - max_spatial_dead_rate - max value of spatial_dead_rate across all layers
        """
        return self._model_activity_stats

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
            def convert(i_idx):
                if self._activity_stats[0][i_idx] is not None:
                    item_data = [np.stack(iteration_stats[i_idx], axis=-1) for iteration_stats in self._activity_stats]
                    return pd.DataFrame(item_data, columns=self._activity_stat_keys())
                return None
            num_items = len(self._activity_stats[0])
            self._activity_stats = [convert(i_idx) for i_idx in range(num_items)]

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
                activity_sum.assign(tf.zeros_like(activity_sum))
            for activity_sum in self._spatial_activity_sums:
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
        # Callback doesn't honour python 3's MRO, so init mixins directly
        ValueStatsCollectingMixin.__init__(self, value_stats=value_stats, value_stats_quantiles=value_stats_quantiles)
        ActivityStatsCollectingMixin.__init__(self, data_format='SC', activity_stats=activity_stats)
        super().__init__()
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
        self._collected_stats_indices = None
        self._variable_stats_mask = None
        self._filtered_value_variable_indices = None

    @property
    def variable_value_stats(self):
        """
        List (by variable) of dataframes containing value stats, or None if not enabled.
        Each variable's list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that variable.
        """
        return self._value_stats

    @property
    def variable_magnitude_stats(self):
        """
        List (by variable) of dataframes containing value magnitude stats, or None if not enabled.
        Each variable's list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that variable.
        """
        return self._magnitude_stats

    @property
    def variable_activity_stats(self):
        """
        List (by variable) of dataframes containing activity stats, or None if not enabled.
        Each variable's list entry is either as pandas dataframe of shape (iterations, stats), with the following
        stats, or None if stats are not collected for that variable:
            - activation_rate - fraction of non-zero values
            - dead_rate - fraction of output channels that are all zero across all other dims
            - spatial_dead_rate - fraction across other dims where all channels are zero
        """
        return self._activity_stats

    @property
    def collected_variable_value_stats(self):
        """
        Variable value stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_variable_stats_indices() to identify which model variables are included.
        """
        if self._value_stats is not None:
            return [stats for stats in self._value_stats if stats is not None]
        else:
            return None

    @property
    def collected_variable_magnitude_stats(self):
        """
        Variable magnitude stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_variable_stats_indices() to identify which model variables are included.
        """
        if self._magnitude_stats is not None:
            return [stats for stats in self._magnitude_stats if stats is not None]
        else:
            return None

    @property
    def collected_variable_activity_stats(self):
        """
        Variable activity stats filtered only on those that have been collected.
        None if activity collection is disabled.
        Use collected_variable_stats_indices() to identify which model variables are included.
        """
        if self._activity_stats is not None:
            return [stats for stats in self._activity_stats if stats is not None]
        else:
            return None

    @property
    def collected_variable_stats_indices(self):
        """
        Indices of variables for which value, magnitude, and activity stats are returned.
        Indices are as per the variable's position in model.variables.
        """
        return self._collected_stats_indices

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
        self._collected_stats_indices = [idx for idx, value in enumerate(values) if value is not None]
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
        self._collect_raw_values(values)

    def _collect_raw_values(self, values):
        # TODO do slicing
        if self._variable_values:
            for var_idx, (val_list, value) in enumerate(zip(self._variable_values, values)):
                if value is not None:
                    val_list.append(tf.identity(value))  # take copy of current state


class GradientHistoryCallback(BaseGradientCallback, ValueStatsCollectingMixin, ActivityStatsCollectingMixin):
    """
    Custom tot.fit() gradient callback that collects various statistics and/or raw values of the gradients during
    training.
    Only works for normal gradients of trainable variables.
    See `LayerOutputGradientHistoryCallback` for gradients w.r.t layer outputs.

    Other properties:
        model: the model captured
        epochs: list of int. Epoch numbers correlated to captured gradients/gradient-stats
            (only available if per_step is False)
        steps: list of int. Step numbers correlated to captured gradients/gradient-stats
            (only available if per_step is True)
    """

    def __init__(self, per_step=False, value_stats_quantiles=None, value_stats=True, activity_stats=True,
                 collection_sets=None):
        """
        Args:
            per_step: bool. Whether to collect per-step stats and raw values, or per-epoch otherwise.
                By default, data is accumulated per-epoch, and an 'epochs' list is available
                that tracks the display indices of each sample.
                If per-step is set, then a `steps` list is available instead, and activity
                is collected on each update step.
                The same applies to layer output capture if enabled.
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
        ValueStatsCollectingMixin.__init__(self, value_stats=value_stats, value_stats_quantiles=value_stats_quantiles)
        ActivityStatsCollectingMixin.__init__(self, data_format='SC', activity_stats=activity_stats)
        super().__init__()
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
        self._collected_stats_indices = None
        self._collected_stats_indices_transpose = None  # from model.variable to model.trainable_variable
        self._filtered_value_variable_indices = None

    @property
    def gradient_value_stats(self):
        """
        List (by variable) of dataframes containing gradient value stats, or None if not enabled.
        Each variable's list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that variable.
        """
        return self._value_stats

    @property
    def gradient_magnitude_stats(self):
        """
        List (by variable) of dataframes containing gradient magnitude stats, or None if not enabled.
        Each variable's list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that variable.
        """
        return self._magnitude_stats

    @property
    def gradient_activity_stats(self):
        """
        List (by variable) of dataframes containing gradient activity stats, or None if not enabled.
        Each variable's list entry is either as pandas dataframe of shape (iterations, stats), with the following
        stats, or None if stats are not collected for that variable:
            - activation_rate - fraction of non-zero values
            - dead_rate - fraction of output channels that are all zero across all other dims
            - spatial_dead_rate - fraction across other dims where all channels are zero
        """
        return self._activity_stats

    @property
    def collected_gradient_value_stats(self):
        """
        Gradient value stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_gradient_stats_indices() to identify which model variables have gradients included.
        """
        if self._value_stats is not None:
            return [stats for stats in self._value_stats if stats is not None]
        else:
            return None

    @property
    def collected_gradient_magnitude_stats(self):
        """
        Gradient magnitude stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_gradient_stats_indices() to identify which model variables have gradients included.
        """
        if self._magnitude_stats is not None:
            return [stats for stats in self._magnitude_stats if stats is not None]
        else:
            return None

    @property
    def collected_gradient_activity_stats(self):
        """
        Gradient activity stats filtered only on those that have been collected.
        None if activity collection is disabled.
        Use collected_gradient_stats_indices() to identify which model variables have gradients included.
        """
        if self._activity_stats is not None:
            return [stats for stats in self._activity_stats if stats is not None]
        else:
            return None

    @property
    def collected_gradient_stats_indices(self):
        """
        Indices of variables for which value, magnitude, and activity stats are returned.
        Indices are as per the variable's position in model.variables.
        """
        return self._collected_stats_indices

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
        self._collected_stats_indices =\
            [_index_by_identity(self.model.variables, var) for var in self.model.trainable_variables]

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
            self._do_collection(gradients)

    def on_train_batch_end(self, batch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Collects gradient stats and raw gradients after each update step, if configured.
        """
        # collect stats at each training step
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
                if value is not None:
                    val_list.append(value)


class LayerOutputHistoryCallback(BaseGradientCallback, ValueStatsCollectingMixin,
                                 ActivityStatsCollectingMixin):
    """
    Custom tot.fit() gradient callback function that collects various statistics and/or raw values of
    layer outputs during training.

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
            per_step: bool. Whether to collect per-step stats, or per-epoch otherwise.
                By default, activity is accumulated per-epoch, and an 'epochs' list is available
                that tracks the display indices of each sample.
                If per-step is set, then a `steps` list is available instead, and activity
                is collected on each update step.
                The same applies to layer output capture if enabled.
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
        self.per_step = per_step
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
        self._collected_stats_indices = None
        self._filtered_value_layer_indices = None

    @property
    def layer_shapes(self):
        """
        List of shapes of each layer, regardless of what's being collected.
        Provided for convenience when displaying results and doing other analysis,
        because this information is hard to obtain directly from the model.
        """
        return self._layer_shapes

    @property
    def layer_value_stats(self):
        """
        List (by layer) of dataframes containing value stats, or None if not enabled.
        Each layer's list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that layer.
        """
        return self._value_stats

    @property
    def layer_magnitude_stats(self):
        """
        List (by layer) of dataframes containing value magnitude stats, or None if not enabled.
        Each layer's list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that layer.
        """
        return self._magnitude_stats

    @property
    def layer_activity_stats(self):
        """
        List (by layer) of dataframes containing activity stats, or None if not enabled.
        Each layer's list entry is either as pandas dataframe of shape (iterations, stats), with the following
        stats, or None if stats are not collected for that layer:
            - activation_rate - mean fraction of units active at any given time
            - dead_rate - fraction of units/channels that are always zero across batch and spatial dims
            - spatial_dead_rate - fraction of spatial positions that are always zero across batch and channel dims
        """
        return self._activity_stats

    @property
    def collected_layer_value_stats(self):
        """
        Layer output value stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_layer_stats_indices() to identify which model layers are included.

        Note: currently, all layers are always included. This flexibility is included for consistency with the other
        callbacks in this suite, and for the flexibility to pick and choose the layers in the future
        """
        if self._value_stats is not None:
            return [stats for stats in self._value_stats if stats is not None]
        else:
            return None

    @property
    def collected_variable_magnitude_stats(self):
        """
        Layer output magnitude stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_layer_stats_indices() to identify which model layers are included.
        """
        if self._magnitude_stats is not None:
            return [stats for stats in self._magnitude_stats if stats is not None]
        else:
            return None

    @property
    def collected_layer_activity_stats(self):
        """
        Layer activity stats filtered only on those that have been collected.
        None if activity collection is disabled.
        """
        if self._activity_stats is not None:
            return [stats for stats in self._activity_stats if stats is not None]
        else:
            return None

    @property
    def collected_layer_stats_indices(self):
        """
        Indices of layers for which value, magnitude, and activity stats are returned.
        Indices are as per the layer's position in model.layers.
        """
        return self._collected_stats_indices

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
        # precompute _collected_stats_indices independent of data collection mixins because they might be disabled
        self._collected_stats_indices = list(range(len(self.model.layers)))

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

        # stats calculations for each step, if configured
        if self.per_step:
            self.steps.append(self.params['steps'] * self._epoch + batch)
            self._collect_value_stats(activations)
            self._collect_activity_stats(1)
            self._collect_raw_values(activations)

    def on_epoch_end(self, epoch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Collects gradient stats and raw gradients after each epoch, if configured at per-epoch level.
        """
        # stats calculation across entire epoch, if configured
        # (uses partial stats that were accumulated across the steps in the batch)
        if not self.per_step:
            self.epochs.append(epoch)
            self._collect_value_stats(activations)
            self._collect_activity_stats(self.params['steps'])
            self._collect_raw_values(activations)

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
        self.per_step = per_step
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
        self._collected_stats_indices = None
        self._filtered_value_layer_indices = None

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
    def layer_value_stats(self):
        """
        List (by layer) of dataframes containing gradient value stats, or None if not enabled.
        Each layer's list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that layer.
        """
        return self._value_stats

    @property
    def layer_magnitude_stats(self):
        """
        List (by layer) of dataframes containing gradient magnitude stats, or None if not enabled.
        Each layer's list entry is either as pandas dataframe of shape (iterations, percentiles),
        or None if stats are not collected for that layer.
        """
        return self._magnitude_stats

    @property
    def layer_activity_stats(self):
        """
        List (by layer) of dataframes containing gradient activity stats, or None if not enabled.
        Each layer's list entry is either as pandas dataframe of shape (iterations, stats), with the following
        stats, or None if stats are not collected for that layer:
            - activation_rate - mean fraction of units that have non-zero gradients at any given time
            - dead_rate - fraction of units/channels that always have zero gradients across batch and spatial dims
            - spatial_dead_rate - fraction of spatial positions that always have zero-gradients across batch and
              channel dims
        """
        return self._activity_stats

    @property
    def collected_layer_value_stats(self):
        """
        Layer gradient stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_layer_stats_indices() to identify which model layers have gradients included.

        Note: no gradients are collected for the very last layer, nor for any layers which do not participate
        in the final loss.
        """
        if self._value_stats is not None:
            return [stats for stats in self._value_stats if stats is not None]
        else:
            return None

    @property
    def collected_variable_magnitude_stats(self):
        """
        Layer gradient magnitude stats filtered only on those that have been collected.
        None if value stats collection is disabled.
        Use collected_layer_stats_indices() to identify which model layers have gradients included.
        """
        if self._magnitude_stats is not None:
            return [stats for stats in self._magnitude_stats if stats is not None]
        else:
            return None

    @property
    def collected_layer_activity_stats(self):
        """
        Layer gradient activity stats filtered only on those that have been collected.
        None if activity collection is disabled.
        """
        if self._activity_stats is not None:
            return [stats for stats in self._activity_stats if stats is not None]
        else:
            return None

    @property
    def collected_layer_stats_indices(self):
        """
        Indices of layers for which gradient value, magnitude, and activity stats are returned.
        Indices are as per the layer's position in model.layers.
        """
        return self._collected_stats_indices

    @property
    def gradients(self):
        """
        Raw captured gradients, if any.
        None if raw data is not being captured, and each layer entry is None if that
        layer is not captured.
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
            self._collected_stats_indices = [idx for idx, value in enumerate(output_gradients) if value is not None]

        # accumulate activation data
        # - always add each batch, regardless of emitting stats per-step or per-epoch
        # - accum when per-epoch, overwrite when per-step
        is_accum = (not self.per_step)
        self._accum_activity_stats(output_gradients, is_accum)

        # stats calculations for each step, if configured
        if self.per_step:
            self.steps.append(self.params['steps'] * self._epoch + batch)
            self._collect_value_stats(output_gradients)
            self._collect_activity_stats(1)
            self._collect_raw_values(output_gradients)

    def on_epoch_end(self, epoch, loss, gradients, trainable_variables, activations, output_gradients):
        """
        Collects gradient stats and raw gradients after each epoch, if configured at per-epoch level.
        """
        # stats calculation across entire epoch, if configured
        # (uses partial stats that were accumulated across the steps in the batch)
        if not self.per_step:
            self.epochs.append(epoch)
            self._collect_value_stats(output_gradients)
            self._collect_activity_stats(self.params['steps'])
            self._collect_raw_values(output_gradients)

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


class ActivityRateMeasuringCallback(tf.keras.callbacks.Callback):
    """
    Standard model.fit() callback that intermittently computes unit activation rates during training,
    via call to measure_unit_activity().

    More costly to run than ActivityHistoryCallback, but can be used with the standard training function.
    However, hasn't been maintained and probably doesn't work properly anymore.

    Uses measure_unit_activity() with the following definition:
    > All layers are assumed to produce outputs with shapes of form: `(batch_size, ..spatial_dims.., channels)`.
    > Unit activation rates are recorded per-channel, aggregated across batch and spatial dims, and then stats collected
    > across the channels.
    > In other words, each physical unit with its unique set of weights is treated as a single atomic component
    > that is re-used multiple times across batch and spatial dims. Stats are then collected against that atomic
    > component in terms of how often it is active (non-zero output) vs inactive (zero output).
    """

    def __init__(self, x, interval=1, batch_size=None, **kwargs):
        """
        Args:
            x: Input data or TF Dataset
                Ideally batched, but will try to automatically batch otherwise.
            interval: int, default: every epoch.
                Sample every N epochs.
            batch_size: int
                Must be supplied in order to apply to dataset if not already batched.
        """
        super().__init__(**kwargs)
        self.x = x
        self.interval = interval
        self.batch_size = batch_size

        # collected stats
        self.epochs = None
        self.model_stats = None
        self.layer_stats = None

        # internal tracking
        self._dataset = None
        self._monitoring_model = None
        self._layer_names = None

    @property
    def layer_names(self):
        return self._layer_names

    @property
    def layer_shapes(self):
        return self._monitoring_model.output_shape

    def on_train_begin(self, logs=None):
        """
        Initialises tracking, now that we know the model and number of epochs
        """
        # handle variations in how data is supplied
        if isinstance(self.x, tf.data.Dataset):
            self._dataset = self.x
            # ensure dataset has been batched
            if not self._is_batched(self._dataset):
                steps_per_epoch = self.params['steps']
                if self.batch_size is None and steps_per_epoch is not None:
                    self.batch_size = int(math.ceil(self._dataset.cardinality().numpy() / steps_per_epoch))
                if self.batch_size is not None:
                    print(f"Batching dataset with batch size {self.batch_size}")
                    self._dataset = self._dataset.batch(self.batch_size)
                else:
                    raise ValueError("dataset not batched and unable to infer batch size.")
        else:
            steps_per_epoch = self.params['steps']
            if self.batch_size is None and steps_per_epoch is not None:
                self.batch_size = int(math.ceil(len(self.x) / steps_per_epoch))
            if self.batch_size is None:
                raise ValueError("one of batch_size or steps_per_epoch must be provided when x list/array given.")
            y = tf.zeros((len(self.x),))  # fake y values
            self._dataset = tf.data.Dataset.from_tensor_slices((self.x, y)).batch(self.batch_size)

        # prepare access to each layer of model
        self._monitoring_model = tf.keras.Model(inputs=self.model.inputs,
                                                outputs=[layer.output for layer in self.model.layers])
        self._layer_names = [layer.name for layer in self.model.layers]

        # init stats
        self.epochs = []
        self.model_stats = {}
        self.layer_stats = [{key: [] for key in self._stat_keys()} for _ in self.model.layers]

    def on_train_end(self, logs=None):
        """
        Cleans up tracking, and converts everything to numpy arrays for easier consumption.
        """
        self.epochs = np.array(self.epochs)
        for key in self.model_stats.keys():
            self.model_stats[key] = np.array(self.model_stats[key])
        for l_idx in range(len(self.layer_stats)):
            for key in self.layer_stats[l_idx].keys():
                self.layer_stats[l_idx][key] = np.array(self.layer_stats[l_idx][key])

    def on_epoch_end(self, epoch, logs=None):
        # every interval and definitely on last epoch
        if epoch % self.interval == 0 or epoch == (self.params['epochs'] - 1):
            # compute for this epoch
            epoch_model_stats, epoch_layer_stats = measure_unit_activity(self._monitoring_model, self._dataset,
                                                                         extract_layers=False)

            # accumulate over time
            self.epochs.append(epoch)
            self._append_dict_list(self.model_stats, epoch_model_stats)
            for l_idx, stats in enumerate(epoch_layer_stats):
                self._append_dict_list(self.layer_stats[l_idx], stats)

    @staticmethod
    def _append_dict_list(dic, addendum_dict):
        for key in addendum_dict.keys():
            if key not in dic:
                dic[key] = []
            dic[key].append(addendum_dict[key])

    @staticmethod
    def _stat_keys():
        """
        Gets the list of stats that will be computed.
        Currently static but may be computed based on configuration in the future.
        """
        return ['dead_rate', 'activation_rate']

    @staticmethod
    def _is_batched(dataset):
        def _is_batched_spec(obj):
            return isinstance(obj, tf.TensorSpec) and isinstance(obj.shape, tf.TensorShape) and obj.shape.rank > 0 and \
                obj.shape[0] is None

        try:
            for spec in dataset.element_spec:
                if isinstance(spec, tuple):
                    for item in spec:
                        if _is_batched_spec(item):
                            return True
                elif _is_batched_spec(spec):
                    return True
            print(f"not batched")
            return False
        except AttributeError:
            print(f"error")
            return False

    def plot(self):
        """
        Alias for plot_summary().
        Determines the default once there are multiple plot methods.
        """
        self.plot_summary()

    def plot_summary(self):
        """
        Alias for calling tot.plot_activity_rate_history(activity_callback).
        """
        plot_activity_rate_history(self)


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
    scales = []
    for variable_stat in stats_dataframes:
        if variable_stat is not None:
            # estimate the overall magnitude scale at the target quantile
            # - assumes that the original dataset was either raw values centered around
            #   zero, or was magnitude values.
            pos_mean = variable_stat[scale_quantile].to_numpy()
            neg_mean = variable_stat[100-scale_quantile].to_numpy()
            if np.any(neg_mean < 0):
                # assume source dataset falls either side of zero
                # - scale is average of the positive and negative mean magnitudes
                scale = (pos_mean + abs(neg_mean)) * 0.5
            else:
                # assume source dataset is positive-only
                # - scale is just the target quantile alone
                scale = pos_mean
            scales.append(scale)  # shape: variables x iterations
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

              # one of:
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

              # one of:
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
      (model_stats, layer_stats, layer_spatial_activity_rates), where:
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
        monitoring_model = tf.keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
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


def plot_value_history(callback: ValueStatsCollectingMixin, magnitudes=True):
    """
    Generates a figure containing a number of plots to visualise value or magnitude stats collected
    by one of the callbacks in this module.

    Uses a log plot for magnitudes, and a linear plot for raw values.

    Args:
        callback: any of "value stats collecting" callbacks in this module
        magnitudes: whether to plot stats for value magnitudes (default),
            or for raw values otherwise.
    """

    # collect data
    iterations = callback.epochs if hasattr(callback, 'epochs') else callback.steps
    iteration_name = 'epoch' if hasattr(callback, 'epochs') else 'step'
    model = callback.model
    model_stats = callback.model_magnitude_stats
    if callback._magnitude_stats is None:
        raise ValueError(f"{type(callback).__name__} did not collect value magnitude stats")
    collected_item_magnitude_stats = [item_stats for item_stats in callback._magnitude_stats if item_stats is not None]
    if magnitudes:
        collected_item_stats = [item_stats for item_stats in callback._magnitude_stats if item_stats is not None]
        collected_item_indices = [i_idx for i_idx, item_stats in enumerate(callback._magnitude_stats) if item_stats is not None]
    else:
        if callback._value_stats is None:
            raise ValueError(f"{type(callback).__name__} did not collect value stats")
        collected_item_stats = [item_stats for item_stats in callback._value_stats if item_stats is not None]
        collected_item_indices = [i_idx for i_idx, item_stats in enumerate(callback._value_stats) if item_stats is not None]
    num_item_stats = len(collected_item_stats)

    # Deal with callback differences
    title = None
    item_mode = None
    item_name = None
    item_name_upper = None
    if isinstance(callback, VariableHistoryCallback):
        item_mode = 'variables'
        item_name = "variable"
        item_name_upper = "Variable"
        trainable_only = callback.trainable_only
        before_updates = callback.before_updates
        title = f"All model trainable variables" if trainable_only else "All model variables (incl. non-trainable)"
        title += ("\n(before updates)" if before_updates else "\n(after updates)")
    elif isinstance(callback, GradientHistoryCallback):
        item_mode = 'variables'
        item_name = "gradient"
        item_name_upper = "Gradient"
        title = "All model gradients"
    elif isinstance(callback, LayerOutputHistoryCallback):
        item_mode = 'layers'
        item_name = "output"
        item_name_upper = "Output"
        title = "All model outputs"
    elif isinstance(callback, LayerOutputGradientHistoryCallback):
        item_mode = 'layers'
        item_name = "output gradient"
        item_name_upper = "Output gradient"
        title = "Al model output gradients"

    # Prepare for layer mode
    item_display_names = []
    if item_mode == 'layers':
        layer_shapes = callback.layer_shapes
        item_shapes = [layer_shapes[i_idx] for i_idx in collected_item_indices]
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
    grid_width = max(4, round(math.sqrt(num_item_stats) / 2) * 2)  # nearest even number >= 4
    grid_height = 2 + math.ceil(num_item_stats / grid_width)
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # all-model high-level summary
    plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
    _plot_add_quantiles(iterations, model_stats)
    plt.margins(0)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(iteration_name)
    plt.ylabel(f"mean scale of {item_name} magnitudes")
    plt.legend()

    # layer contributions - high-level summary
    # - for easier visual display in variables mode, uses only the largest variable from each layer
    if item_mode == 'layers':
        filtered_layer_names = [model.layers[l_idx].name for l_idx in collected_item_indices]
        filtered_stats = collected_item_magnitude_stats
    else:
        # tuples of: (v_idx of biggest variable, size of biggest variable)
        layer_metas = [(None, None)] * len(model.layers)
        layer_id_lookup = layer_indices_by_variable(model)
        for v_idx in collected_item_indices:
            l_idx = layer_id_lookup[v_idx]
            variable = model.variables[v_idx]
            biggest_v_idx, biggest_size = layer_metas[l_idx]
            if biggest_size is None or tf.size(variable) > biggest_size:
                layer_metas[l_idx] = v_idx, tf.size(variable)
        filtered_layer_names =\
            [model.layers[l_idx].name for l_idx, (v_idx, _) in enumerate(layer_metas) if v_idx is not None]
        filtered_stats = [callback._magnitude_stats[v_idx] for v_idx, _ in layer_metas if v_idx is not None]
    scales = get_scales_across_stats_list(filtered_stats, scale_quantile=50)
    band_log_scales = _log_normalize(scales, axis=-1)

    plt.subplot2grid((grid_height, grid_width), (0, grid_width // 2), colspan=grid_width // 2, rowspan=2)
    plt.stackplot(iterations, band_log_scales.T, colors=['lightsteelblue', 'royalblue'], linewidth=0)
    plt.margins(0)
    plt.title('Layer comparison')
    plt.xlabel(iteration_name)
    plt.ylabel(f"{item_name_upper} scale log-proportion")
    # layer labels placed on mid-height of layer band on left-hand side
    x_loc = round(band_log_scales.shape[0] / 100)
    placement = band_log_scales[x_loc, :] * 0.5
    placement[1:] += np.cumsum(band_log_scales[0, :])[0:-1]
    for f_idx, layer_name in enumerate(filtered_layer_names):
        plt.text(len(iterations) / 100, placement[f_idx], layer_name, ha="left")

    # individual layers or variables
    for i_idx in range(num_item_stats):
        r = 2 + i_idx // grid_width
        c = i_idx % grid_width
        plt.subplot2grid((grid_height, grid_width), (r, c))
        data = collected_item_stats[i_idx]
        _plot_add_quantiles(iterations, data)
        plt.margins(0)
        plt.yscale('log' if magnitudes else 'linear')
        if c == 0:
            plt.ylabel('log-magnitude' if magnitudes else 'value')
        plt.title(item_display_names[i_idx])

        # text overlay
        plot_min = np.min(data.to_numpy())
        plot_max = np.max(data.to_numpy())
        plot_width = np.max(iterations)
        plot_mid = (plot_min + plot_max) * 0.5
        if item_shapes:
            plt.text(plot_width * 0.5, plot_mid,
                     f"{item_shapes[i_idx]}",
                     horizontalalignment='center', verticalalignment='center')

    plt.show()


def plot_gradient_history(gradient_callback: GradientHistoryCallback, magnitudes=False):
    """
    Generates a figure containing a number of plots to visualise gradient stats
    from a GradientHistoryCallback object.

    Args:
        gradient_callback: callback populated with gradient stats from training.
        magnitudes: whether to show raw values or estimate stats for magnitudes.
            When showing magnitudes, automatically switches to a log plot for easier
            comparison across differing scales.
            Forced when the original callback collected magnitude data.
    """
    # collect data
    iterations = gradient_callback.epochs if hasattr(gradient_callback, 'epochs') else gradient_callback.steps
    iteration_name = 'epoch' if hasattr(gradient_callback, 'epochs') else 'step'
    needs_conversion_to_magnitudes = magnitudes
    if gradient_callback.magnitudes:
        magnitudes = True  # forced
        needs_conversion_to_magnitudes = False
    model_stats = gradient_callback.model_stats

    model = gradient_callback.model
    gradient_stats = gradient_callback.collected_gradient_stats
    variable_ids = gradient_callback.collected_gradient_stats_indices
    num_gradient_stats = len(gradient_stats)
    layer_id_lookup = layer_indices_by_variable(model)
    variable_shapes = [model.variables[v_idx].shape for v_idx in variable_ids]

    variable_display_names = []
    for v_idx in variable_ids:
        layer_id = layer_id_lookup[v_idx]
        layer_name = model.layers[layer_id].name
        variable_name = model.variables[v_idx].name
        variable_display_names.append(f"{layer_name}(#{layer_id})/{variable_name}")

    # start figure
    # - at least 4 layer plots wide
    # - otherwise target a square grid of layer plots
    grid_width = max(4, round(math.sqrt(num_gradient_stats) / 2) * 2)  # nearest even number >= 4
    grid_height = 2 + math.ceil(num_gradient_stats / grid_width)
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # all-model high-level summary
    plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
    _plot_add_quantiles(iterations, model_stats)
    plt.margins(0)
    plt.yscale('log')
    plt.title('All model gradients')
    plt.xlabel(iteration_name)
    plt.ylabel('mean scale of gradient magnitudes')
    plt.legend()

    # layer contributions - high-level summary
    # - for easier visual display uses only the largest variable from each layer
    # - also only includes trainable layers
    filtered_layer_metas = []  # list of tuples: (l_idx, layer, var_idx)
    for l_idx, layer in enumerate(model.layers):
        biggest_var = None
        if layer.trainable:
            for var in layer.variables:
                if biggest_var is None or tf.size(var) > tf.size(biggest_var):
                    biggest_var = var
        if biggest_var is not None:
            var_idx = _index_by_identity(model.trainable_variables, biggest_var)
            filtered_layer_metas.append((l_idx, layer, var_idx))
    filtered_gradients = [gradient_stats[v_idx] for l_idx, layer, v_idx in filtered_layer_metas]
    scales = get_scales_across_stats_list(filtered_gradients, scale_quantile=75)
    band_log_scales = _log_normalize(scales, axis=-1)

    plt.subplot2grid((grid_height, grid_width), (0, grid_width // 2), colspan=grid_width // 2, rowspan=2)
    plt.stackplot(iterations, band_log_scales.T, colors=['lightsteelblue', 'royalblue'], linewidth=0)
    plt.margins(0)
    plt.title('Layer comparison')
    plt.xlabel(iteration_name)
    plt.ylabel('Gradient scale log-proportion')
    # layer labels placed on centre of layer band on left-hand side
    x_loc = round(band_log_scales.shape[0] / 100)
    placement = band_log_scales[x_loc, :] * 0.5
    placement[1:] += np.cumsum(band_log_scales[0, :])[0:-1]
    for f_idx in range(len(filtered_layer_metas)):
        l_idx, layer, v_idx = filtered_layer_metas[f_idx]
        plt.text(len(iterations) / 100, placement[f_idx], layer.name, ha="left")

    # individual layers
    for v_idx in range(num_gradient_stats):
        r = 2 + v_idx // grid_width
        c = v_idx % grid_width
        plt.subplot2grid((grid_height, grid_width), (r, c))
        data = gradient_stats[v_idx]
        if needs_conversion_to_magnitudes:
            # approximate stats over magnitudes
            data = np.abs(data.to_numpy())
            data = np.sort(data, axis=-1)
            data = pd.DataFrame(data, columns=gradient_stats[v_idx].columns)
        _plot_add_quantiles(iterations, data)
        plt.margins(0)
        plt.yscale('log' if magnitudes else 'linear')
        if c == 0:
            plt.ylabel('log-magnitude' if magnitudes else 'value')
        plt.title(variable_display_names[v_idx])

        # text overlay
        plot_min = np.min(data.to_numpy())
        plot_max = np.max(data.to_numpy())
        plot_width = np.max(iterations)
        plot_mid = (plot_min + plot_max) * 0.5
        if variable_shapes:
            plt.text(plot_width * 0.5, plot_mid,
                     f"{variable_shapes[v_idx]}",
                     horizontalalignment='center', verticalalignment='center')

    plt.show()


def plot_output_gradient_history(gradient_callback: LayerOutputGradientHistoryCallback, magnitudes=False):
    """
    Generates a figure containing a number of plots to visualise layer output gradient stats
    from a LayerOutputGradientHistoryCallback object.

    Args:
        gradient_callback: callback populated with gradient stats from training.
        magnitudes: whether to show raw values or estimate stats for magnitudes.
            When showing magnitudes, automatically switches to a log plot for easier
            comparison across differing scales.
            Forced when the original callback collected magnitude data.
    """
    # collect data
    iterations = gradient_callback.epochs if hasattr(gradient_callback, 'epochs') else gradient_callback.steps
    iteration_name = 'epoch' if hasattr(gradient_callback, 'epochs') else 'step'
    needs_conversion_to_magnitudes = magnitudes
    if gradient_callback.magnitudes:
        magnitudes = True  # forced
        needs_conversion_to_magnitudes = False
    model = gradient_callback.model
    model_stats = gradient_callback.model_stats
    layer_stats = gradient_callback.collected_layer_stats
    layer_ids = gradient_callback.collected_layer_stats_indices
    layer_names = [model.layers[l_idx].name for l_idx in layer_ids]
    layer_shapes = gradient_callback.layer_shapes

    num_layer_stats = len(layer_stats)

    # start figure
    # - at least 4 layer plots wide
    # - otherwise target a square grid of layer plots
    grid_width = max(4, round(math.sqrt(num_layer_stats) / 2) * 2)  # nearest even number >= 4
    grid_height = 2 + math.ceil(num_layer_stats / grid_width)
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # all-model high-level summary
    plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
    _plot_add_quantiles(iterations, model_stats)
    plt.margins(0)
    plt.yscale('log')
    plt.title('All model output gradients')
    plt.xlabel(iteration_name)
    plt.ylabel('mean scale of gradient magnitudes')
    plt.legend()

    # layer contributions - high-level summary
    scales = get_scales_across_stats_list(layer_stats, scale_quantile=75)
    band_log_scales = _log_normalize(scales, axis=-1)

    plt.subplot2grid((grid_height, grid_width), (0, grid_width // 2), colspan=grid_width // 2, rowspan=2)
    plt.stackplot(iterations, band_log_scales.T, colors=['lightsteelblue', 'royalblue'], linewidth=0)
    plt.margins(0)
    plt.title('Layer comparison')
    plt.xlabel(iteration_name)
    plt.ylabel('Gradient scale log-proportion')
    # layer labels placed on centre of layer band on left-hand side
    x_loc = round(band_log_scales.shape[0] / 100)
    placement = band_log_scales[x_loc, :] * 0.5
    placement[1:] += np.cumsum(band_log_scales[0, :])[0:-1]
    for s_idx in range(len(layer_stats)):
        l_idx = layer_ids[s_idx]
        layer = model.layers[l_idx]
        plt.text(len(iterations) / 100, placement[s_idx], layer.name, ha="left")

    # individual layers
    for s_idx in range(num_layer_stats):
        r = 2 + s_idx // grid_width
        c = s_idx % grid_width
        plt.subplot2grid((grid_height, grid_width), (r, c))
        data = layer_stats[s_idx]
        if needs_conversion_to_magnitudes:
            # approximate stats over magnitudes
            data = np.abs(data.to_numpy())
            data = np.sort(data, axis=-1)
            data = pd.DataFrame(data, columns=layer_stats[s_idx].columns)
        _plot_add_quantiles(iterations, data)
        plt.margins(0)
        plt.yscale('log' if magnitudes else 'linear')
        if c == 0:
            plt.ylabel('log-magnitude' if magnitudes else 'value')
        plt.title(layer_names[s_idx])
        plt.title(f"layer {layer_ids[s_idx]}:\n{layer_names[s_idx]}")

        # text overlay
        plot_min = np.min(data.to_numpy())
        plot_max = np.max(data.to_numpy())
        plot_width = np.max(iterations)
        plot_mid = (plot_min + plot_max) * 0.5
        if layer_shapes:
            plt.text(plot_width * 0.5, plot_mid,
                     f"{layer_shapes[s_idx]}",
                     horizontalalignment='center', verticalalignment='center')

    plt.show()


# TODO change to be agnostic to the callback, so long as it uses the _ActivityStatsCollectingCapability
def plot_activity_rate_history(activity_callback):
    """
    Plots a high-level summary of unit activity rates across the entire model
    and across each layer.

    Args:
        activity_callback: instance of ActivityHistoryCollback or ActivityRateMeasuringCallback after training.

    Generated figure is of form:
    - top row (two columns):
        - "Unit activation rates across layers"
        - "Dead unit rates across layers"
    - remaining rows (arranged approximately as a square grid)
        - per-layer unit activation rates
    """
    # collect data
    iterations = activity_callback.epochs if hasattr(activity_callback, 'epochs') else activity_callback.steps
    iteration_name = 'epoch' if hasattr(activity_callback, 'epochs') else 'step'
    num_layers = len(activity_callback.layer_stats)
    model = activity_callback.model
    model_stats = activity_callback.model_stats
    layer_stats = activity_callback.layer_stats
    layer_names = [layer.name for layer in model.layers]
    channel_sizes = [shape[-1] for shape in activity_callback.layer_shapes]

    # start figure
    # - at least 4 layer plots wide
    # - otherwise target a square grid of layer plots
    grid_width = max(4, round(math.sqrt(num_layers) / 2) * 2)  # nearest even number >= 4
    grid_height = 2 + math.ceil(num_layers / grid_width)
    plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

    # all-model high-level summary
    plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
    plt.plot(iterations, model_stats['mean_activation_rate'], label='mean activation rate',
             color='tab:blue')
    plt.fill_between(iterations, model_stats['min_activation_rate'],
                     model_stats['max_activation_rate'], color='tab:blue', alpha=0.2,
                     label='min/max range')
    plt.ylim([0.0, 1.1])
    plt.title("Unit activation rates across layers")
    plt.xlabel(iteration_name)
    plt.ylabel('fraction of units')
    plt.legend()

    plt.subplot2grid((grid_height, grid_width), (0, grid_width // 2), colspan=grid_width // 2, rowspan=2)
    plt.plot(iterations, model_stats['mean_dead_rate'], label='mean dead rate', color='tab:red')
    plt.fill_between(iterations, model_stats['min_dead_rate'], model_stats['max_dead_rate'],
                     color='tab:red', alpha=0.2, label='min/max range')
    plt.ylim([0.0, 1.1])
    plt.title("Dead unit rates across layers")
    plt.xlabel(iteration_name)
    plt.ylabel('fraction of units')
    plt.legend()

    # individual layers
    for l_idx in range(num_layers):
        r = 2 + l_idx // grid_width
        c = l_idx % grid_width
        plt.subplot2grid((grid_height, grid_width), (r, c))
        dead_rates = layer_stats[l_idx]['dead_rate']
        activation_rates = layer_stats[l_idx]['activation_rate']
        plt.plot(iterations, activation_rates, label='activation rates', color='tab:blue')
        plt.fill_between(iterations, 0, activation_rates, color='tab:blue', alpha=0.2)
        plt.plot(iterations, dead_rates, label='dead units', color='tab:red')
        plt.fill_between(iterations, 0, dead_rates, color='tab:red', alpha=0.2)
        plt.ylim([0.0, 1.0])
        plt.margins(0)
        plt.title(f"layer {l_idx}:\n{layer_names[l_idx]}" if model is not None else f"layer {l_idx}")

        # text overlay
        plot_width = np.max(iterations)
        final_dead_rate = dead_rates[-1]
        plt.text(plot_width * 0.5, 0.5,
                 f"{channel_sizes[l_idx]} units\n"
                 f"{final_dead_rate * 100:.1f}% dead",
                 horizontalalignment='center', verticalalignment='center')

        if l_idx == 0:
            plt.legend()

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
    # - each layer has two plots, arranged virtically
    # - otherwise target a square grid of layer plots
    grid_width = max(4, round(math.sqrt(num_layers)))
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
    # - each layer has two plots, arranged virtically
    # - otherwise target a square grid of layer plots
    grid_width = max(4, round(math.sqrt(num_layers * 2)))
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


def _plot_add_quantiles(x, data):
    """
    Adds multi-quantile data to an existing plot.
    Useful for displaying stats returned by the history callbacks.
    Args:
        x: list-like. X-axis values.
        data: pandas Dataframe with columns corresponding to quantiles, labeled in range 0 to 100.
    """
    def _label(q1, q2):
        if q2 is None:
            return "median" if q1 == 50 else f"{q1}%"
        elif q1 == 0 and q2 == 100:
            return "min/max"
        elif 100 - q1 == q2:
            return f"±{q1}%"
        else:
            return f"{q1}% to {q2}%"

    quantiles = data.columns
    quantile_len = len(quantiles)
    bot, top = 0, quantile_len - 1
    while bot < top:
        color = 'tab:grey' if quantiles[bot] == 0 and quantiles[top] == 100 else 'tab:blue'
        plt.fill_between(x, data[quantiles[bot]], data[quantiles[top]],
                         alpha=0.2, color=color, linewidth=0,
                         label=_label(quantiles[bot], quantiles[top]))
        bot += 1
        top -= 1
    if bot == top:
        plt.plot(x, data[quantiles[bot]],
                 color='tab:blue',
                 label=_label(quantiles[bot], None))
