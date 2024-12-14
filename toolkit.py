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
        self.group_start_epoch = 0
        self.group_start_time = tf.timestamp()
        if self.display_interval is None:
            self.display_interval = math.floor(self.epoch_count / self.display_total)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        if ((epoch + 1) % self.display_interval == 0) or epoch == self.epoch_count-1:
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



def train_step(model, x_batch_train, y_batch_train, sample_weight):
    reported_loss = None

    # Forward pass
    with tf.GradientTape() as tape:
        y_batch_pred = model(x_batch_train)
        loss = model.compute_loss(x=x_batch_train, y=y_batch_train, y_pred=y_batch_pred, sample_weight=sample_weight,
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


# Honours tf.config.run_functions_eagerly(bool)
# See tensorflow trainer.Trainer.train_step() for reference
def fit(model, dataset, epochs=1, verbose=1, callbacks=None, initial_epoch=0):
    # prepare epochs
    num_batches = len(dataset)

    # prepare callbacks tracking
    # if verbose >= 1:
    #  history = tf.keras.callbacks.History()
    #  callbacks.append(history)
    if not isinstance(callbacks, tf.keras.callbacks.CallbackList):
        callbacks = tf.keras.callbacks.CallbackList(callbacks, add_history=True, add_progbar=verbose != 0,
                                                    verbose=verbose, epochs=epochs, steps=num_batches, model=model)

    # prepare train function
    if tf.config.functions_run_eagerly():
        print(f"Execution mode: eager")
        train_step_fn = train_step
    else:
        print(f"Execution mode: autograph")
        train_step_fn = tf.function(train_step)

    # start
    callbacks.set_params({'verbose': 1, 'epochs': epochs, 'steps': len(dataset)})

    # train
    # gradients_list = []
    logs = {}  # holds latest value at any given moment in time
    callbacks.on_train_begin()
    for epoch in range(initial_epoch, epochs):
        model.reset_metrics()
        start = tf.timestamp()
        callbacks.on_epoch_begin(epoch)

        for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            sample_weight = None  # TODO use: x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
            # print(f"  Step {step+1}: x_batch_train: {x_batch_train.shape}, y_batch_train: {y_batch_train.shape}")
            callbacks.on_train_batch_begin(step)

            loss, metrics, gradients = train_step_fn(model, x_batch_train, y_batch_train, sample_weight)

            # gradients_list.append(gradients)
            logs = metrics
            logs['loss'] = loss.numpy()
            callbacks.on_train_batch_end(step, logs)

        # end of epoch
        dur = (tf.timestamp() - start).numpy()
        callbacks.on_epoch_end(epoch, logs)  # should be passing loss and mse
        metric_str = ''
        for k in logs.keys():
            metric_str += f" - {k}: {logs[k]:.3f}"
        # print(f"Epoch {epoch+1} - {dur:.1f}s - loss: {logs['loss']:.3f}{metric_str}")
    callbacks.on_train_end(logs)

    # return history, gradients_list
    return model.history