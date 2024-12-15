import tensorflow as tf
import math
import numpy as np
import matplotlib as plt


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
    callbacks.on_train_begin()
    gradient_callback.on_train_begin()
    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        start = tf.timestamp()
        callbacks.on_epoch_begin(epoch)
        gradient_callback.on_epoch_begin(epoch)

        for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            sample_weight = None  # TODO use: x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
            # print(f"  Step {step+1}: x_batch_train: {x_batch_train.shape}, y_batch_train: {y_batch_train.shape}")
            callbacks.on_train_batch_begin(step)
            gradient_callback.on_train_batch_begin(step)

            loss, metrics, gradients = train_step_fn(model, x_batch_train, y_batch_train, sample_weight)

            logs = metrics
            logs['loss'] = loss.numpy()
            callbacks.on_train_batch_end(step, logs)
            gradient_callback.on_train_batch_end(step, loss, gradients, model.trainable_variables, None)

        # end of epoch
        dur = (tf.timestamp() - start).numpy()
        callbacks.on_epoch_end(epoch, logs)  # should be passing loss and mse
        gradient_callback.on_epoch_end(epoch, loss, gradients, model.trainable_variables, None)
        metric_str = ''
        for k in logs.keys():
            metric_str += f" - {k}: {logs[k]:.3f}"
    callbacks.on_train_end(logs)
    gradient_callback.on_train_end()

    return model.history


# Tries to replicate keras.backend.tensorflow.TensorFlowTrainer.train_step() (trainer.py, keras 3.5.0)
# as much as possible.
def _gradient_returning_train_step(model, x_batch_train, y_batch_train, sample_weight):
    """
    This method is programmatically converted via auto-graph.

    Returns:
        loss: float. Loss returned by loss function (before optimizer scaling).
        metrics: dict. Metrics returned by model (note: also includes a 'loss' value but it's always zero)
        gradients: list. Gradients tensor for each trainable variable
    """

    # Forward pass
    with tf.GradientTape() as tape:
        y_batch_pred = model(x_batch_train)
        loss = model.compute_loss(x=x_batch_train,
                                  y=y_batch_train,
                                  y_pred=y_batch_pred,
                                  sample_weight=sample_weight,
                                  training=True)
        reported_loss = loss  # tracking before scaling
        loss = model.optimizer.scale_loss(loss)

    # Backward pass
    if model.trainable_weights:
        gradients = tape.gradient(loss, model.trainable_variables)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
        raise ValueError('No trainable weights to update.')

    # Metrics
    metrics = model.compute_metrics(x=x_batch_train, y=y_batch_train, y_pred=y_batch_pred, sample_weight=sample_weight)

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


class GradientHistoryCallback(BaseGradientCallback):
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


def show_gradient_stats(gradients_cb: GradientHistoryCallback):
    """
    Generates a figure containing a number of plots to visualise gradient stats
    from a GradientHistoryCallback object.

    Args:
        gradients_cb: gradients collected during training.
    """
    steps = gradients_cb.steps

    plt.figure(figsize=(15, 4))

    # all-model high-level summary
    plt.subplot(1, 2, 1)
    mean = gradients_cb.model_stats['mean']
    std = gradients_cb.model_stats['std']
    plt.plot(steps, gradients_cb.model_stats['mean'], label='mean', color='royalblue')
    plt.fill_between(steps, mean - std, mean + std, color='blue', alpha=0.2, linewidth=0, label='+/- sd')
    plt.fill_between(steps, gradients_cb.model_stats['min'], gradients_cb.model_stats['max'], color='lightgray',
                     linewidth=0, alpha=0.2, label='min/max range')
    plt.margins(0)
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('gradient magnitude')
    plt.title('All model gradients')
    plt.legend()

    # all-layer high-level summary
    plt.subplot(1, 2, 2)
    layer_log_means = np.column_stack([gradients_cb.layer_stats[l]['mean'] for l in range(len(gradients_cb.layer_stats))])
    layer_log_means = _log_normalize(layer_log_means, axis=1)
    plt.stackplot(steps, layer_log_means.T, colors=['lightsteelblue', 'royalblue'], linewidth=0)
    plt.margins(0)
    plt.xlabel('step')
    plt.ylabel('Log-proportion contribution')
    plt.title('Layer contributions')
    # layer labels placed on centre of layer band on left-hand side
    placement = layer_log_means[0, :] * 0.5
    placement[1:] += np.cumsum(layer_log_means[0, :])[0:-1]
    for l in range(layer_log_means.shape[1]):
        plt.text(len(steps) / 100, placement[l], f"layer {l}", ha="left")

    plt.show()


def _log_normalize(arr, axis=None):
    """
    Normalises all values in the array or along the given axis so that they sum to 1.0,
    and so that large scale differences are reduced to linear differences in the same
    way that a log plot converts orders of magnitude differences to linear differences.

    Args:
        arr: numpy array or similar

    Returns:
        new array
    """
    # convert to log scale
    # - result: orders-of-magnitude numbers in range -inf..+inf (eg: -4 to +4)
    scaled = np.log(arr)

    # move everything into positive
    # - shift such that the min value gets value 1.0, so it doesn't become zero.
    scaled = scaled - (np.min(scaled, axis=axis, keepdims=True) - 1)

    # normalize
    return scaled / np.sum(scaled, axis=axis, keepdims=True)
