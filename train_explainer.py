# Includes some special purpose functions for "explaining" how various forms of matrix multiplication
# produce the results that they do.
# For example, you can obtain zero or near-zero results in a matmul operation for a number of reasons:
#  - everything on both inputs could be near-zero
#  - everything in just one input could be near-zero while the other had only large positive values
#  - both inputs could be a mixture of near-zeros and positive values that overlap in such a way that
#    you never get a combination of positive * positive.
#  - both inputs could contain large values, but some are negative, and they cancel out to produce a near-zero value.
#
# The functions here group the computations in such a way that makes it possibly to easily classify the results
# of a given computation to explain its output.

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import train_observability_toolkit as tot


def classify_terms():
    return ['PP', 'PZ', 'PN', 'ZP', 'ZZ', 'ZN', 'NP', 'NZ', 'NN']


# change threshold args to 'thresholds', taking a single value or a list, eg: 0.35; [None, 0.75]; [0.45, 0.75]
def matmul_classify(x1, x2, confidence: float = 0.95, threshold1: float = None, threshold2: float = None):
    """
    Calculates how the dot-product of x1 . x2 comes to have the range of values that it does.

    For each computation of an output cell, both sets of inputs are independently grouped
    into those that are near-zero (Z), positive at or above threshold (P), and negative at or below threshold (N).
    The output cell value is then defined as the sum of the combinations of those groups, with 9 combinations
    in total, in the following fixed order: PP, PZ, PN, ZP, ZZ, ZN, NP, NZ, NN.

    The counts and sums of each component are returned as separate tensors.
    The original matmul result for any given output cell is given by simply
    summing the sums of each category:
      sum(PP) + sum(PZ) + sum(PN) + ... sum(NN)
    For the entire result this can be computed via:
      output = np.sum(sums, axis=-1)

    "Near zero" is defined as any value having an absolute value less than a threshold.
    By default, the threshold is determined by taking the (1 - zero_confidence) percentile level across each
    of the separate inputs. Alternatively, an absolute magnitude threshold may be specified for
    any of the inputs.

    Example usage:
    >>> a = np.tile(np.arange(0.0, 1.0, 0.1), (10,1))
    >>> counts, sums = matmul_classify(a, a, confidence=0.75)
    >>> for i, name in enumerate(classify_terms()):
    >>>   if np.sum(counts[:,:,i]) > 0:
    >>>     print(f"Counts({name}): {counts[:,:,i]}")
    >>> for i, name in enumerate(classify_terms()):
    >>>   if np.sum(counts[:,:,i]) > 0:
    >>>     print(f"Sums({name}): {sums[:,:,i]}")
    >>> print(f"True matmul: {np.matmul(a, a)}")
    >>> print(f"Derived matmul: {np.sum(sums, axis=-1)}")

    Args:
      x1: np-array or tensor with shape (n, k)
      x2: np-array or tensor with shape (k, m)
      confidence: statistical confidence (0.0 to 1.0) that you wish to meet
        that a value is accurately placed within the P, Z, or N categories.
        Higher values lead to more strict requirements for "near zero".
        1.0 only considers exactly 0.0 as "near zero".
      threshold1: abs(X1) values less than this are considered near-zero,
        otherwise inferred from confidence
      threshold2: abs(X2) values less than this are considered near-zero,
        otherwise inferred from confidence

    Returns:
      (counts, sums) where each is an np-array with shape (n, m, 9)
    """

    # standardise on data format
    x1 = np.array(x1)
    x2 = np.array(x2)

    # determine thresholds
    # (note: on small matrices with few discrete numbers, np.percentile will find a value on either side
    #  of the percentage threshold, thus we should apply the threshold rule as "zero if value < threshold"
    if threshold1 is None:
        threshold1 = np.percentile(np.abs(x1), 100 * (1 - confidence), method='midpoint')
    if threshold2 is None:
        threshold2 = np.percentile(np.abs(x2), 100 * (1 - confidence), method='midpoint')

    # create masks that classify each input individually
    x1_p = x1 >= threshold1
    x1_z = np.abs(x1) < threshold1
    x1_n = x1 <= -threshold1

    x2_p = x2 >= threshold2
    x2_z = np.abs(x2) < threshold2
    x2_n = x2 <= -threshold2

    # compute counts
    counts = np.zeros((x1.shape[0], x2.shape[1], 9), dtype=int)
    sums = np.zeros((x1.shape[0], x2.shape[1], 9), dtype=np.result_type(x1.dtype, x2.dtype))

    counts[:, :, 0] = np.matmul(x1_p.astype(int), x2_p.astype(int))
    counts[:, :, 1] = np.matmul(x1_p.astype(int), x2_z.astype(int))
    counts[:, :, 2] = np.matmul(x1_p.astype(int), x2_n.astype(int))
    counts[:, :, 3] = np.matmul(x1_z.astype(int), x2_p.astype(int))
    counts[:, :, 4] = np.matmul(x1_z.astype(int), x2_z.astype(int))
    counts[:, :, 5] = np.matmul(x1_z.astype(int), x2_n.astype(int))
    counts[:, :, 6] = np.matmul(x1_n.astype(int), x2_p.astype(int))
    counts[:, :, 7] = np.matmul(x1_n.astype(int), x2_z.astype(int))
    counts[:, :, 8] = np.matmul(x1_n.astype(int), x2_n.astype(int))

    sums[:, :, 0] = np.matmul(x1 * x1_p, x2 * x2_p)
    sums[:, :, 1] = np.matmul(x1 * x1_p, x2 * x2_z)
    sums[:, :, 2] = np.matmul(x1 * x1_p, x2 * x2_n)
    sums[:, :, 3] = np.matmul(x1 * x1_z, x2 * x2_p)
    sums[:, :, 4] = np.matmul(x1 * x1_z, x2 * x2_z)
    sums[:, :, 5] = np.matmul(x1 * x1_z, x2 * x2_n)
    sums[:, :, 6] = np.matmul(x1 * x1_n, x2 * x2_p)
    sums[:, :, 7] = np.matmul(x1 * x1_n, x2 * x2_z)
    sums[:, :, 8] = np.matmul(x1 * x1_n, x2 * x2_n)

    return counts, sums


# change threshold args to 'thresholds', taking a single value or a list, eg: 0.35; [None, 0.75]; [0.45, 0.75]
def multiply_classify(x, y, confidence: float = 0.95, x_threshold: float = None, y_threshold: float = None):
    """
    Like matmul_classify but for elementwise multiplication.

    Args:
        x: np-array or tensor
        y: np-array or tensor, must have the same type and shape as x

        confidence: statistical confidence (0.0 to 1.0) that you wish to meet
          that a value is accurately placed within the P, Z, or N categories.
          Higher values lead to more strict requirements for "near zero".
          1.0 only considers exactly 0.0 as "near zero".
        x_threshold: abs(X1) values less than this are considered near-zero,
          otherwise inferred from confidence
        y_threshold: abs(X2) values less than this are considered near-zero,
          otherwise inferred from confidence

    Returns:
        (counts, sums) containing the counts and sums of each component, respectively.
        Each a tensor with shape `x_shape + (9,)`.
    """
    # standardise on data format
    x = tf.constant(x)
    y = tf.constant(y)

    # determine thresholds
    # (note: on small matrices with few discrete numbers, percentile() will find a value on either side
    #  of the percentage threshold, thus we should apply the threshold rule as "zero if value < threshold"
    if x_threshold is None:
        x_threshold = tfp.stats.percentile(tf.abs(x), 100 * (1 - confidence), interpolation='midpoint')
    if y_threshold is None:
        y_threshold = tfp.stats.percentile(tf.abs(y), 100 * (1 - confidence), interpolation='midpoint')

    # create masks that classify each input individually
    x_p = x >= x_threshold
    x_z = np.abs(x) < x_threshold
    x_n = x <= -x_threshold

    y_p = y >= y_threshold
    y_z = np.abs(y) < y_threshold
    y_n = y <= -y_threshold

    # compute counts and sums for each classification
    counts = []
    x_pc = tf.cast(x_p, tf.float32)
    x_zc = tf.cast(x_z, tf.float32)
    x_nc = tf.cast(x_n, tf.float32)
    y_pc = tf.cast(x_p, tf.float32)
    y_zc = tf.cast(x_z, tf.float32)
    y_nc = tf.cast(x_n, tf.float32)
    counts.append(tf.math.multiply(x_pc, y_pc))
    counts.append(tf.math.multiply(x_pc, y_zc))
    counts.append(tf.math.multiply(x_pc, y_nc))
    counts.append(tf.math.multiply(x_zc, y_pc))
    counts.append(tf.math.multiply(x_zc, y_zc))
    counts.append(tf.math.multiply(x_zc, y_nc))
    counts.append(tf.math.multiply(x_nc, y_pc))
    counts.append(tf.math.multiply(x_nc, y_zc))
    counts.append(tf.math.multiply(x_nc, y_nc))

    sums = []
    x_pv = tf.where(x_p, x, tf.zeros_like(x))
    x_zv = tf.where(x_z, x, tf.zeros_like(x))
    x_nv = tf.where(x_n, x, tf.zeros_like(x))
    y_pv = tf.where(x_p, y, tf.zeros_like(y))
    y_zv = tf.where(x_z, y, tf.zeros_like(y))
    y_nv = tf.where(x_n, y, tf.zeros_like(y))
    sums.append(tf.math.multiply(x_pv, y_pv))
    sums.append(tf.math.multiply(x_pv, y_zv))
    sums.append(tf.math.multiply(x_pv, y_nv))
    sums.append(tf.math.multiply(x_zv, y_pv))
    sums.append(tf.math.multiply(x_zv, y_zv))
    sums.append(tf.math.multiply(x_zv, y_nv))
    sums.append(tf.math.multiply(x_nv, y_pv))
    sums.append(tf.math.multiply(x_nv, y_zv))
    sums.append(tf.math.multiply(x_nv, y_nv))

    # format into final output
    return tf.stack(counts, axis=-1), tf.stack(sums, axis=-1)


# change threshold args to 'thresholds', taking a single value or a list, eg: 0.35; [None, 0.75]; [0.45, 0.75]
def conv_classify(inputs, kernel, strides=1, padding="VALID", confidence: float = 0.95,
                  inputs_threshold: float = None, kernel_threshold: float = None):
    """
    Like matmul_classify but for convolutions.
    Supports 1D, 2D and 3D convolution.

    Args:
        inputs: Tensor of rank N+2. `inputs` has shape
            `(batch_size,) + inputs_spatial_shape + (num_channels,)`
        kernel: Tensor of rank N+2. `kernel` has shape
            `(kernel_spatial_shape, num_input_channels, num_output_channels)`.
            `num_input_channels` should match the number of channels in
            `inputs`.
        strides: int or int tuple/list of `len(inputs_spatial_shape)`,
            specifying the strides of the convolution along each spatial
            dimension. If `strides` is int, then every spatial dimension shares
            the same `strides`.
        padding: string, either `"valid"` or `"same"`. `"valid"` means no
            padding is applied, and `"same"` results in padding evenly to the
            left/right or up/down of the input such that output has the
            same height/width dimension as the input when `strides=1`.
        confidence: statistical confidence (0.0 to 1.0) that you wish to meet
          that a value is accurately placed within the P, Z, or N categories.
          Higher values lead to more strict requirements for "near zero".
          1.0 only considers exactly 0.0 as "near zero".
        inputs_threshold: abs(X1) values less than this are considered near-zero,
          otherwise inferred from confidence
        kernel_threshold: abs(X2) values less than this are considered near-zero,
          otherwise inferred from confidence

    Returns:
        (counts, sums) containing the counts and sums of each component, respectively.
        Each a tensor with shape `(batch_size,) + inputs_spatial_shape + (num_channels,9)`.
    """
    # standardise on data format
    inputs = tf.constant(inputs)
    kernel = tf.constant(kernel)

    # determine thresholds
    # (note: on small matrices with few discrete numbers, percentile() will find a value on either side
    #  of the percentage threshold, thus we should apply the threshold rule as "zero if value < threshold"
    if inputs_threshold is None:
        inputs_threshold = tfp.stats.percentile(tf.abs(inputs), 100 * (1 - confidence), interpolation='midpoint')
    if kernel_threshold is None:
        kernel_threshold = tfp.stats.percentile(tf.abs(kernel), 100 * (1 - confidence), interpolation='midpoint')

    # create masks that classify each input individually
    inputs_p = inputs >= inputs_threshold
    inputs_z = np.abs(inputs) < inputs_threshold
    inputs_n = inputs <= -inputs_threshold

    kernel_p = kernel >= kernel_threshold
    kernel_z = np.abs(kernel) < kernel_threshold
    kernel_n = kernel <= -kernel_threshold

    # compute counts and sums for each classification
    counts = []
    inputs_pc = tf.cast(inputs_p, tf.float32)
    inputs_zc = tf.cast(inputs_z, tf.float32)
    inputs_nc = tf.cast(inputs_n, tf.float32)
    kernel_pc = tf.cast(kernel_p, tf.float32)
    kernel_zc = tf.cast(kernel_z, tf.float32)
    kernel_nc = tf.cast(kernel_n, tf.float32)
    counts.append(tf.nn.convolution(input=inputs_pc, filters=kernel_pc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_pc, filters=kernel_zc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_pc, filters=kernel_nc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_zc, filters=kernel_pc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_zc, filters=kernel_zc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_zc, filters=kernel_nc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_nc, filters=kernel_pc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_nc, filters=kernel_zc, strides=strides, padding=padding))
    counts.append(tf.nn.convolution(input=inputs_nc, filters=kernel_nc, strides=strides, padding=padding))

    sums = []
    inputs_pv = tf.where(inputs_p, inputs, tf.zeros_like(inputs))
    inputs_zv = tf.where(inputs_z, inputs, tf.zeros_like(inputs))
    inputs_nv = tf.where(inputs_n, inputs, tf.zeros_like(inputs))
    kernel_pv = tf.where(kernel_p, kernel, tf.zeros_like(kernel))
    kernel_zv = tf.where(kernel_z, kernel, tf.zeros_like(kernel))
    kernel_nv = tf.where(kernel_n, kernel, tf.zeros_like(kernel))
    sums.append(tf.nn.convolution(input=inputs_pv, filters=kernel_pv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_pv, filters=kernel_zv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_pv, filters=kernel_nv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_zv, filters=kernel_pv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_zv, filters=kernel_zv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_zv, filters=kernel_nv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_nv, filters=kernel_pv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_nv, filters=kernel_zv, strides=strides, padding=padding))
    sums.append(tf.nn.convolution(input=inputs_nv, filters=kernel_nv, strides=strides, padding=padding))

    # format into final output
    return tf.stack(counts, axis=-1), tf.stack(sums, axis=-1)


def _find_inbound_layers(model, layer, return_type='layer'):
    """
    :return_type: one of 'layer', 'index'
    """
    if return_type not in ['layer', 'index']:
        raise ValueError(f"return_type must be one of 'layer', 'index', got: {return_type}")
    layers = [_find_layer_by_node(model, node, return_type) for inbound in layer._inbound_nodes for node in
              inbound.parent_nodes]
    return [node for node in layers if node is not None]


def _find_outbound_layers(model, layer, return_type='layer'):
    """
    :return_type: one of 'layer', 'index'
    """
    if return_type not in ['layer', 'index']:
        raise ValueError(f"return_type must be one of 'layer', 'index', got: {return_type}")
    layers = [_find_layer_by_node(model, node, return_type) for node in layer._outbound_nodes]
    return [node for node in layers if node is not None]


def _find_layer_by_node(model, node, return_type='layer'):
    """
    :return_type: one of 'layer', 'index'
    """
    for l_idx, layer in enumerate(model.layers):
        if node in layer._inbound_nodes:
            return layer if return_type == 'layer' else l_idx
    return None


def _split_by_largest(tensors):
    """
    Splits the given list of tensors into the single largest tensor, and the rest.
    Useful as a heuristic for identifying the main weights tensor in a layer and the rest, without assuming
    a particular order.
    If there are multiple largest tensors, the first one is returned.
    Returns:
      largest_tensor, [rest]
    """
    biggest_t, biggest_idx = None, None
    for t_idx, t in enumerate(tensors):
        if t is not None:
            if biggest_t is None or tf.size(t) > tf.size(biggest_t):
                biggest_t = t
                biggest_idx = t_idx
    rest = [t for t_idx, t in enumerate(tensors) if t_idx != biggest_idx]
    return biggest_t, rest


def explain_near_zero_gradients(layer_index: int,
                                gradients: tot.GradientHistoryCallback,
                                activity: tot.ActivityHistoryCallback,
                                variables: tot.VariableHistoryCallback,
                                epoch: None, step: None,
                                threshold: float = None, threshold_percentile: float = 0.01):
    """
    Attempts to identify the explanation for zero and near-zero gradients in a given layer.

    Args:
        layer_index: layer to examine
        gradients:
            callback populated with raw gradients for at least the layer in question
            and any AFTER it that take it as input
        activity:
            callback populated with raw layer outputs for at least the layer in question
            and any BEFORE it that it takes as input
        variables:
            callback populated with raw weights, biases, and other variables for at least the layer in question
            and any AFTER it that take it as input
        epoch: epoch number selection from callback histories, if callbacks captured against epochs
        step: step number selection from callback histories, if callbacks captured against update steps
        threshold: gradients equal or below this magnitude are considered "near-zero"
        threshold_percentile: threshold inferred at this percentile magnitude of the range of gradients.
            Ignored for this layer gradient if 'threshold' is set, but still used for other thresholds.

    Usage:
    > l_idx = 35
    > gradients = GradientHistoryCallback(collection_sets=[{layer_indices: [l_idx, l_idx+1]})
    > activity = ActivityHistoryCallback(collection_sets=[{layer_indices: [l_idx-1, l_idx]})
    > variables = VariableHistoryCallback(collection_sets=[{layer_indices: [l_idx, l_idx+1]}, before_updates=True)
    > fit(model, train_data, callbacks=[gradients, activity, variables])
    > explain_near_zero_gradients(l_idx, epoch=..., gradients, activity, variables)
    Example output:
    > Examining layer 35
    >   weights shape: (None, 10, 10, 512, 256)
    >   bias shape: (None, 10, 10, 256)
    > Layer summary:
    >   units: 25600
    >   fan-in: 51200
    >   biases: 25600
    > Strict-zero gradients with strict zero or negative sources:
    >   units: 347 (1.36%)
    >
    > Neor-zero gradients (magnitude < 0.00035):
    >
    >
    """

    # handle arguments
    if epoch is not None:
        iteration = gradients.epochs.index(epoch)
    elif step is not None:
        iteration = gradients.steps.index(step)
    else:
        raise ValueError("One of epoch or step must be specified")

    # collect reference information
    model = gradients.model
    variable_indices_lookup = tot.variable_indices_by_layer(model, include_trainable_only=True)
    variable_indices = tot.variable_indices_by_layer(model, include_trainable_only=True)[layer_index]
    target_layer_variables = [variables.variables[var_index][iteration] for var_index in variable_indices]
    target_layer_activations = activity.layer_outputs[layer_index][iteration]
    target_layer_gradients = [gradients.gradients_list[var_index][iteration] for var_index in variable_indices]
    target_layer_weights = None # TODO - arbitrarily pick largest of target_layer_variables
    target_layer_other_vars = []  # TODO - all other layer vars except the biggest one

    # TODO be smarter by examining the model structure to identify which layer or layers are inputs
    input_activations = activity.layer_outputs[layer_index-1][iteration]

    # print basic summary
    print(f"Examining layer {layer_index}:")
    print(f"  units: {target_layer_activations.shape} = {np.size(target_layer_activations)}")  # need to remove batch dim
    print(f"  layer input: {input_activations.shape} = {np.size(input_activations)}")  # need to remove batch dim
    print(f"  fan-in per unit: {target_layer_weights.shape} = {np.size(target_layer_weights)}")  # need to remove output channel dim
    if len(target_layer_other_vars) == 0:
        print(f"  biases: <none>")
    else:
        for other_var in target_layer_other_vars:
            print(f"  biases: {other_var.shape} = {np.size(other_var)}")

    # identify gradients to investigate
    # - theory:
    #    A single unit of a dense or conv layer has fan-in weights and 1 bias. Any number of those may encounter
    #    zero or near near-zero gradients. On the bias we don't really care. And we probably don't care about
    #    individual near-zero grdients, but rather where a single unit is suffering from many near-zero gradients.
    # - approach:
    #    Need to think more about this.
    #    Currently selecting by gradient.
    #    Might need to also select units, and mix them according to different sources.
    if threshold is None:
        threshold = ... # TODO identify percentile level across all flottened values across all target_layer_gradients
    target_variable_threshold_masks = [np.abs(grad) <= threshold for grad in target_layer_gradients]  # works for gradients and variables
    target_variable_zero_masks = [np.exact(grad, 0.0) for grad in target_layer_gradients]  # works for gradients and variables

    print(f"Examining {np.sum(target_variable_zero_masks)} zero gradients "
          f"and {np.sum(target_variable_threshold_masks)} near-zero gradients total...")

    # Do lengthy processing
    # TODO show progress bar if having to iterate over everything

    # Source: activations
    # - theory:
    #     For dense targets layers, all input activations contribute as inputs to each individual unit
    #     on the target layer.
    #     For conv target layers, this is more complex because it's only true for a narrow spatial subset of
    #     the input activations, typically with size (K x K x input_channels)
    # - approach:
    #     Identify a threshold by magnitude-percentile across all input activations.
    #     For dense target layers, just get all mag <= threshold activations. This works globally for all target
    #     units.
    #     For conv target layers, use the shape of the weights to figure out the kernel size
    #     and select the correct subset for each target unit in question.
    input_activation_threshold = ... # TOOD identify percentile level across all flattened values across all input activaitons

    # Source: neor-zero weights
    # - theory:
    #     The variables of the target layer don't directly influence its gradient. Rather, they indirectly
    #     influence it by causing zero or negative z-outputs which push the ReLU activation into the zero-gradient zone.
    #     So we want to identify the causes of zero or negative z-outputs.
    #     Note: negative weights alone shouldn't lead to negative z-outputs, as they may counteract
    #     negative inputs. But, because the input layer most likely also used ReLU, it'll never produce
    #     negative values. Thus under most circumstances it is safe to assume that a negative weight will tend the
    #     target layer's output towards zero.
    #     For both dense and conv layers, it's sufficient to just look at the fraction of weights that are zero
    #     or negative, for each selected output unit.
    # - approach:

    # Source: zero or negative z-outputs
    # - theory:
    #     For dense target layers, the pre-activation z values can be calculated simply via
    #     np.matmul(weights, input_activations) + biases.
    #     For conv target layers, ....


