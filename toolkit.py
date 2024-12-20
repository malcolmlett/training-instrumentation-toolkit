import tensorflow as tf
import keras
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm


class LessVerboseProgressLogger(tf.keras.callbacks.Callback):
    """
    Progress logger for when running models that train via thousands of very short epochs.
    By default, automatically logs progress 10 times during training.

    Use as:
    >>> model.fit(...., verbose=0, callbacks=[LessVerboseProgressLogger()])
    """
    def __init__(self, display_interval=None, display_total=10):
        super(LessVerboseProgressLogger, self).__init__()
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
        if self.display_interval == 0 or ((epoch + 1) % self.display_interval == 0) or epoch == self.epoch_count-1:
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
def fit(model, dataset, epochs=1, verbose=1, callbacks=None, initial_epoch=0, gradient_callback=None):
    """
    A custom training loop mimicking model.fit() that makes gradient information available for tracking.

    Honours the state of `tf.config.run_functions_eagerly(bool)`.

    Args:
        model: usual meaning
        dataset: usual meaning
        epochs: usual meaning
        verbose: usual meaning
        callbacks: usual meaning
        initial_epoch: usual meaning
        gradient_callback: Supply a subclass instance of BaseGradientCallback in order to collect gradient
            information during training.
    Returns:
         HistoryCallback
    """
    # prepare epochs
    num_batches = len(dataset)

    # prepare callbacks tracking
    if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
        callbacks = tf.keras.callbacks.CallbackList(callbacks, add_history=True, add_progbar=verbose != 0,
                                                    verbose=verbose, epochs=epochs, steps=num_batches, model=model)
    if gradient_callback is None:
        gradient_callback = BaseGradientCallback()
    gradient_callback.set_params({'epochs': epochs, 'steps': len(dataset)})
    gradient_callback.set_model(model)

    # prepare train function
    if tf.config.functions_run_eagerly():
        train_step_fn = _gradient_returning_train_step
    else:
        train_step_fn = tf.function(_gradient_returning_train_step)

    # train
    logs = {}  # holds latest value at any given moment in time
    loss = None
    gradients = None
    activations = None
    callbacks.on_train_begin()
    gradient_callback.on_train_begin()
    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        start = tf.timestamp()
        callbacks.on_epoch_begin(epoch)
        gradient_callback.on_epoch_begin(epoch)

        for step, data in enumerate(dataset):
            x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
            callbacks.on_train_batch_begin(step)
            gradient_callback.on_train_batch_begin(step)

            loss, metrics, gradients = train_step_fn(model, x, y, sample_weight)

            logs = metrics
            logs['loss'] = loss.numpy()
            callbacks.on_train_batch_end(step, logs)
            gradient_callback.on_train_batch_end(step, loss, gradients, model.trainable_variables, activations)

        # end of epoch
        dur = (tf.timestamp() - start).numpy()
        callbacks.on_epoch_end(epoch, logs)  # should be passing loss and mse
        gradient_callback.on_epoch_end(epoch, loss, gradients, model.trainable_variables, activations)
        metric_str = ''
        for k in logs.keys():
            metric_str += f" - {k}: {logs[k]:.3f}"
    callbacks.on_train_end(logs)
    gradient_callback.on_train_end()

    return model.history


# Tries to replicate keras.backend.tensorflow.TensorFlowTrainer.train_step() (trainer.py, keras 3.5.0)
# as much as possible.
def _gradient_returning_train_step(model, x, y, sample_weight):
    """
    This method is programmatically converted via auto-graph.

    Returns:
        loss: float. Loss returned by loss function (before optimizer scaling).
        metrics: dict. Metrics returned by model (note: also includes a 'loss' value but it's always zero)
        gradients: list. Gradients tensor for each trainable variable
    """

    # Forward pass
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = model.compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=True)
        reported_loss = loss  # tracking before scaling
        loss = model.optimizer.scale_loss(loss)

    # Backward pass
    if model.trainable_weights:
        gradients = tape.gradient(loss, model.trainable_variables)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
        raise ValueError('No trainable weights to update.')

    # Metrics
    metrics = model.compute_metrics(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight)

    return reported_loss, metrics, gradients


class BaseGradientCallback:
    """
    Supply a subclass instance to the custom fit() method in order to collect gradient information.
    This implementation does nothing, and is suitable for use as a no-op.
    """
    def __init__(self):
        self.params = None
        self._model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

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

    def on_epoch_end(self, epoch, loss, gradients, trainable_variables, activations=None):
        """Called at the end of an epoch during training.
        Subclasses should override for any actions to run.

        Supplied with training parameters from the last batch of the epoch.
        Can be a convenient alternative where training doesn't use batches,
        or when you only want to sample the occasional update.

        Args:
            epoch: Integer, index of epoch.
            loss: float. The loss value of the batch.
            gradients: the list of gradients for each trainable variable.
            trainable_variables: the list of trainable variables.
            activations: the list of activations for each layer. Currently
              never populated, but might be in the future.
        """

    def on_train_batch_begin(self, batch):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
        """

    def on_train_batch_end(self, batch, loss, gradients, trainable_variables, activations=None):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            loss: float. The loss value of the batch.
            gradients: the list of gradients for each trainable variable.
            trainable_variables: the list of trainable variables.
            activations: the list of activations for each layer. Currently
              never populated, but might be in the future.
        """


# TODO add explicit properties for epochs, steps, model_stats, layer_stats
class GradientHistoryCallback(BaseGradientCallback):
    """
    Properties:
        epochs: list of int. Epoch numbers correlated to captured gradients/gradient-stats
            (only populated if verbose == 0)
        steps: list of int. Step numbers correlated to captured gradients/gradient-stats
            (only populated if verbose > 0)
        model_stats: dict of np-arrays. Keys list out different stats, eg: mean, min, max, std
        layer_stats: list of dict of np-arrays. One list item for each layer, with each
            list item containing either a dict of stat np-arrays like for model_stats,
            or `None` if it doesn't have any trainable variable.
        trainable_layer_stats: filtered version of layer_stats that omits layers with no trainable variables
        trainable_layer_indices: indices of layers in `trainable_layer_stats`
        trainable_layer_names: names of layers in `trainable_layer_stats`
    """

    def __init__(self, verbose=1):
        """
        Args:
            verbose: int. To be refactored later:
              1: collects stats on gradients at each step or epoch, but does
                 not keep raw gradients.
              2: keeps a sampling of raw gradients
              3: keeps all raw gradients
        """
        super(GradientHistoryCallback, self).__init__()
        self.verbose = verbose

        # results variable creation
        if verbose == 0:
            self.epochs = []
        else:
            self.steps = []  # maybe rename as 'iterations'
        self.model_stats = {}  # dict (by stat) of lists (by iteration)
        self.layer_stats = []  # list (by layer) of dicts (by stat) of lists (by iteration)
        if verbose > 1:
            self.gradients_list = None

        # internal tracking
        self._epoch = 0
        self._variable_indices_by_layer = None
        self._layer_names = None

    @property
    def trainable_layer_stats(self):
        return [stats for stats in self.layer_stats if stats is not None]

    @property
    def trainable_layer_indices(self):
        return [idx for idx, stats in enumerate(self.layer_stats) if stats is not None]

    @property
    def trainable_layer_names(self):
        return [self._layer_names[idx] for idx in self.trainable_layer_indices]

    def on_train_begin(self):
        """
        Initialises tracking, now that we know the model and number of epochs and steps per epoch.
        """
        # init stats
        self.model_stats = {key: [] for key in self._stat_keys()}
        for layer in self.model.layers:
            if layer.trainable_variables:
                self.layer_stats.append({key: [] for key in self._stat_keys()})
            else:
                self.layer_stats.append(None)

        # pre-compute lookups
        self._layer_names = [layer.name for layer in self.model.layers]
        self._variable_indices_by_layer = [[]] * len(self.model.layers)
        for l, layer in enumerate(self.model.layers):
            if layer.trainable_variables:
                indices = [self._index_by_identity(self.model.trainable_variables, var) for var in
                           layer.trainable_variables]
                self._variable_indices_by_layer[l] = indices

        if self.verbose > 1:
            self.gradients_list = []

    def on_train_end(self):
        """
        Cleans up tracking, and converts everything to numpy arrays for easier consumption.
        """
        if self.verbose == 0:
            self.epochs = np.array(self.epochs)
        else:
            self.steps = np.array(self.steps)
        for key in self.model_stats.keys():
            lst = [v.numpy() for v in self.model_stats[key]]
            self.model_stats[key] = np.array(lst)
        for l in range(len(self.layer_stats)):
            if self.layer_stats[l] is not None:
                for key in self.layer_stats[l].keys():
                    lst = [v.numpy() for v in self.layer_stats[l][key]]
                    self.layer_stats[l][key] = np.array(lst)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Just tracks the current epoch number
        """
        self._epoch = epoch

    def on_epoch_end(self, epoch, loss, gradients, trainable_variables, activations=None):
        """
        Collects gradient stats and raw gradients after each epoch, if configured.
        """
        if self.verbose == 0:
            self.epochs.append(self._epoch)
            self._collect_stats(loss, gradients, trainable_variables, activations)

    def on_train_batch_end(self, batch, loss, gradients, trainable_variables, activations=None):
        """
        Collects gradient stats and raw gradients after each update step, if configured.
        """
        # collect stats at each training step
        if self.verbose >= 1:
            step = self.params['steps'] * self._epoch + batch
            self.steps.append(step)
            self._collect_stats(loss, gradients, trainable_variables, activations)

        # collect raw gradients
        if self.verbose >= 3:
            self.gradients_list.append(gradients)

    def _collect_stats(self, loss, gradients, trainable_variables, activations):
        # stats across all gradients as one big bucket (not weight adjusted for different sized layers)
        self._append_dict_list(self.model_stats, self._compute_stats(gradients))

        # compute stats across all gradients (weights + biases) for each layer
        for l_idx, layer in enumerate(self.model.layers):
            grad_indices = self._variable_indices_by_layer[l_idx]
            layer_grads = [gradients[i] for i in grad_indices]
            if layer_grads:
                self._append_dict_list(self.layer_stats[l_idx], self._compute_stats(layer_grads))

    @staticmethod
    def _stat_keys():
        """
        Gets the list of stats that will be computed.
        Currently static but may be computed based on configuration in the future.
        """
        return ['mean', 'min', 'max', 'std']

    # Note: may issue warnings about retracing, but they can be ignored.
    # This method must deal with the variations across each of the ways that it's called during a train step,
    # (eg: whole model vs each layer), but after the first train step it will be fully optimised.
    @tf.function
    def _compute_stats(self, gradients):
        tot_n = tf.constant(0.0, dtype=tf.float32)
        tot_mean = tf.constant(0.0, dtype=tf.float32)
        tot_m2 = tf.constant(0.0, dtype=tf.float32)  # Sum of squared differences from the mean
        tot_min = tf.constant(float("inf"), dtype=tf.float32)
        tot_max = tf.constant(float("-inf"), dtype=tf.float32)

        for g in gradients:
            g_mags = tf.abs(g)
            g_size = tf.size(g, out_type=tf.float32)
            g_min = tf.reduce_min(g_mags)
            g_max = tf.reduce_max(g_mags)
            g_sum = tf.reduce_sum(g_mags)
            g_mean = g_sum / g_size
            g_var = tf.reduce_sum((g_mags - g_mean) ** 2)

            tot_min = tf.minimum(tot_min, g_min)
            tot_max = tf.maximum(tot_max, g_max)

            # Welford's algorithm for computing running statistics
            delta = g_mean - tot_mean
            tot_n += g_size
            tot_mean += delta * (g_size / tot_n)
            tot_m2 += g_var + delta ** 2 * (g_size * (tot_n - g_size) / tot_n)

        return {
            'mean': tot_mean,
            'min': tot_min,
            'max': tot_max,
            'std': tf.sqrt(tot_m2 / tot_n)  # population std.dev
        }

    @staticmethod
    def _index_by_identity(lst, target):
        return next((i for i, v in enumerate(lst) if id(v) == id(target)), -1)

    # note: experiments have found that this is faster than trying to optimise it
    # (for example, trying to use TF Variables to store the lists goes considerably
    #  slower - on scale of 40ms vs 14ms per epoch)
    @staticmethod
    def _append_dict_list(dic, addendum_dict):
        for key in addendum_dict.keys():
            dic[key].append(addendum_dict[key])

    def plot(self):
        """
        Alias for plot_summary().
        Sets the default plot method, once I have multiple.
        """

    def plot_summary(self):
        """
        Generates a figure containing a number of plots to visualise gradient stats
        from a GradientHistoryCallback object.

        Args:
            self: gradients collected during training.
        """
        steps = self.steps

        # get filtered set of layer stats (dropping those with no stats)
        layer_ids = self.trainable_layer_indices
        layer_names = self.trainable_layer_names
        layer_stats = self.trainable_layer_stats
        num_trainable_layers = len(layer_stats)

        layer_log_means = np.column_stack([stats['mean'] for stats in layer_stats])
        layer_log_means = _log_normalize(layer_log_means, axis=1)

        # start figure
        # - at least 4 layer plots wide
        # - otherwise target a square grid of layer plots
        grid_width = max(4, round(math.sqrt(num_trainable_layers) / 2) * 2)  # nearest even number >= 4
        grid_height = 2 + math.ceil(num_trainable_layers / grid_width)
        plt.figure(figsize=(13, 4 * grid_height/2), layout='constrained')

        # all-model high-level summary
        plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
        means = self.model_stats['mean']
        stds = self.model_stats['std']
        mins = self.model_stats['min']
        maxs = self.model_stats['max']
        plt.plot(steps, means, label='mean', color='royalblue')
        plt.fill_between(steps, means - stds, means + stds, color='blue', alpha=0.2, linewidth=0, label='+/- sd')
        plt.fill_between(steps, mins, maxs, color='lightgray', linewidth=0, alpha=0.2, label='min/max range')
        plt.margins(0)
        plt.yscale('log')
        plt.xlabel('step')
        plt.ylabel('gradient magnitude')
        plt.title('All model gradients')
        plt.legend()

        # layer contributions - high-level summary
        plt.subplot2grid((grid_height, grid_width), (0, grid_width // 2), colspan=grid_width // 2, rowspan=2)
        plt.stackplot(steps, layer_log_means.T, colors=['lightsteelblue', 'royalblue'], linewidth=0)
        plt.margins(0)
        plt.xlabel('step')
        plt.ylabel('Log-proportion contribution')
        plt.title('Layer contributions')
        # layer labels placed on centre of layer band on left-hand side
        placement = layer_log_means[0, :] * 0.5
        placement[1:] += np.cumsum(layer_log_means[0, :])[0:-1]
        for l_idx in range(num_trainable_layers):
            plt.text(len(steps) / 100, placement[l_idx], f"{layer_names[l_idx]} (#{layer_ids[l_idx]})", ha="left")

        # individual layers
        for l_idx in range(num_trainable_layers):
            r = 2 + l_idx // grid_width
            c = l_idx % grid_width
            plt.subplot2grid((grid_height, grid_width), (r, c))
            means = layer_stats[l_idx]['mean']
            stds = layer_stats[l_idx]['std']
            mins = layer_stats[l_idx]['min']
            maxs = layer_stats[l_idx]['max']
            plt.plot(steps, means, label='mean', color='royalblue')
            plt.fill_between(steps, means - stds, means + stds, color='blue', alpha=0.2, linewidth=0, label='+/- sd')
            plt.fill_between(steps, mins, maxs, color='lightgray', linewidth=0, alpha=0.2, label='min/max range')
            plt.margins(0)
            plt.yscale('log')
            plt.title(f"{layer_names[l_idx]} (#{layer_ids[l_idx]})")

        plt.show()


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
    arr[mask] = 1.0   # dummy value to avoid warnings from np.log(), will be discarded
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


def measure_unit_activity(model, dataset, include_channel_activity=False, include_spatial_activity=False,
                          verbose=0, **kwargs):
    """
    Measures the rate of unit activations (having non-zero output) across all units in all layers, when
    predictions are made against the X values in the given dataset, and computes stats over the results.

    All layers are assumed to produce outputs with shapes of form: `(batch_size, ..spatial_dims.., channels)`.
    Unit activation rates are recorded per-channel, aggregated across batch and spatial dims, and then stats collected
    across the channels.

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
    layer_channel_activity_sums = [tf.Variable(tf.zeros(size, dtype=tf.float32)) for size in channel_sizes]  # by channel
    if include_spatial_activity:
        layer_spatial_activity_sums = [tf.Variable(tf.zeros(shape, dtype=tf.float32)) for shape in spatial_dims]  # by spatial dims
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
        plt.title(f"{model.layers[l_idx].name} (#{l_idx})" if model is not None else f"layer {l_idx}")
        plt.xlim([0.0, 1.0])
        plt.yticks([])
        plt.xticks([0.0, 0.5, 1.0])
        if r == 0:
            plt.xlabel('activation rate')
        if c == 0:
            plt.ylabel('histogram')
        hist_vals, _, _ = plt.hist(activation_rates, bins=np.arange(0, 1.1, 0.1))
        plot_height = np.max(hist_vals)
        text_col = 'black'
        if 0.0 < dead_rate < 1.0:
            text_col = 'tab:orange'
        elif dead_rate == 1.0:
            text_col = 'tab:red'
        plt.text(0.5, plot_height*0.5,
                 f"len {len_active_rate}\n"
                 f"mean {mean_active_rate:.3f}\nmin {min_active_rate:.3f}\nmax {max_active_rate:.3f}\n"
                 f"dead {dead_rate:.3f}", color=text_col,
                 horizontalalignment='center', verticalalignment='center')
    plt.show()


def plot_spatial_stats(layer_spatial_activity, model=None):
    """
    Simple grid plot of spatially-arrange unit activitation rates across
    the different layers.

    Args:
        layer_spatial_activity:
            as collecte from measure_unit_activity()
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
        plot_shape = (activation_rates.shape[0]-1,activation_rates.shape[1]-1) if tf.rank(activation_rates) >= 2 else (1, 1)

        plt.subplot2grid((grid_height, grid_width), (r, c))
        plt.title(f"{model.layers[l_idx].name} (#{l_idx})" if model is not None else f"layer {l_idx}")
        plt.xticks([])
        plt.yticks([])
        if c == 0:
            plt.ylabel('activations')
        if tf.rank(activation_rates) >= 2:
            plt.imshow(activation_rates, vmin=0.0)
        plt.text(plot_shape[1]*0.5, plot_shape[0]*0.5,
                 f"mean {mean_active_rate:.3f}\nmin {min_active_rate:.3f}\nmax {max_active_rate:.3f}",
                 horizontalalignment='center', verticalalignment='center')

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
        plt.text(plot_shape[1]*0.5, plot_shape[0]*0.5,
                 f"dead rate\n{dead_rate:.3f}", color=text_col,
                 horizontalalignment='center', verticalalignment='center')
    plt.show()


class ActivityRateCallback(tf.keras.callbacks.Callback):
    """
    Model training callback that collects unit activation rates during training.
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
        super(ActivityRateCallback, self).__init__(**kwargs)
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
        if epoch % self.interval == 0 or epoch == (self.params['epochs']-1):
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
        epochs = self.epochs
        num_layers = len(self.layer_stats)
        layer_names = self.layer_names

        # start figure
        # - at least 4 layer plots wide
        # - otherwise target a square grid of layer plots
        grid_width = max(4, round(math.sqrt(num_layers) / 2) * 2)  # nearest even number >= 4
        grid_height = 2 + math.ceil(num_layers / grid_width)
        plt.figure(figsize=(13, 4 * grid_height / 2), layout='constrained')

        # all-model high-level summary
        plt.subplot2grid((grid_height, grid_width), (0, 0), colspan=grid_width // 2, rowspan=2)
        plt.plot(epochs, self.model_stats['mean_activation_rate'], label='mean activation rate',
                 color='tab:blue')
        plt.fill_between(epochs, self.model_stats['min_activation_rate'],
                         self.model_stats['max_activation_rate'], color='tab:blue', alpha=0.2,
                         label='min/max range')
        plt.ylim([0.0, 1.1])
        plt.title("Unit activation rates across layers")
        plt.xlabel('step')
        plt.ylabel('fraction of units')
        plt.legend()

        plt.subplot2grid((grid_height, grid_width), (0, grid_width // 2), colspan=grid_width // 2, rowspan=2)
        plt.plot(epochs, self.model_stats['mean_dead_rate'], label='mean dead rate', color='tab:red')
        plt.fill_between(epochs, self.model_stats['min_dead_rate'], self.model_stats['max_dead_rate'],
                         color='tab:red', alpha=0.2, label='min/max range')
        plt.ylim([0.0, 1.1])
        plt.title("Dead unit rates across layers")
        plt.xlabel('step')
        plt.ylabel('fraction of units')
        plt.legend()

        # individual layers
        for l_idx in range(num_layers):
            r = 2 + l_idx // grid_width
            c = l_idx % grid_width
            plt.subplot2grid((grid_height, grid_width), (r, c))
            dead_rates = self.layer_stats[l_idx]['dead_rate']
            activation_rates = self.layer_stats[l_idx]['activation_rate']
            plt.plot(epochs, activation_rates, label='activation rates', color='tab:blue')
            plt.fill_between(epochs, 0, activation_rates, color='tab:blue', alpha=0.2)
            plt.plot(epochs, dead_rates, label='dead units', color='tab:red')
            plt.fill_between(epochs, 0, dead_rates, color='tab:red', alpha=0.2)
            plt.ylim([0.0, 1.0])
            plt.margins(0)
            plt.title(f"{layer_names[l_idx]} (#{l_idx})")
            if l_idx == 0:
                plt.legend()

        plt.show()
