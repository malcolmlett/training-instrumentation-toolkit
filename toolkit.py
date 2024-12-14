import tensorflow as tf
import math


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
