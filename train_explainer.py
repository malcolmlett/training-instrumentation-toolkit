import numpy as np
import tensorflow as tf
import train_observability_toolkit as tot


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


