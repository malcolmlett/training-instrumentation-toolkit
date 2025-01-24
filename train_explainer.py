import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import train_observability_toolkit as tot
import matmul_explainer as me


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


def _split_by_largest(tensors, labels=None):
    """
    Splits the given list of tensors into the single largest tensor, and the rest.
    Useful as a heuristic for identifying the main weights tensor in a layer and the rest, without assuming
    a particular order.
    If there are multiple largest tensors, the first one is returned.

    Optionally also correspondingly splits tensor labels.

    Args:
      tensors: list of tensors to split
      labels: optional, list of labels corresponding to the tensors, must be same size as tensors

    Returns:
      largest_tensor, [rest]                                       -- if labels is None
      largest_tensor, [rest], largest_tensor_label, [rest labels]  -- if labels is not None
    """
    biggest_t, biggest_idx = None, None

    # identify split
    for t_idx, t in enumerate(tensors):
        if t is not None:
            if biggest_t is None or tf.size(t) > tf.size(biggest_t):
                biggest_t = t
                biggest_idx = t_idx

    # split tensors
    rest = [t for t_idx, t in enumerate(tensors) if t_idx != biggest_idx]
    if labels is None:
        return biggest_t, rest

    # split labels
    biggest_t_label = labels[biggest_idx]
    rest_labels = [labels[t_idx] for t_idx, t in enumerate(labels) if t_idx != biggest_idx]
    return biggest_t, rest, biggest_t_label, rest_labels


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


# TODO consider returning a list of best explanandums and using this list as a better description
# than the grouping done by matmul_explainer
def describe_top_near_zero_explanandum(counts, sums=None, terms=None, mask=None, confidence=0.95):
    """
    Tries to find the single simplest and top-most explanandum for why the computed values are near-zero.
    Examines several common causes of near-zero values and picks the single
    cause with the strongest effect as the top explanandum.
    Prioritises simpler explanandums that require fewer components (eg: preferring "first input is zero" over
    "either input is zero").

    Generally you will supply a mask to pick the values being explained, otherwise it assumes
    all output values were near-zero.

    Terminology note:
      Each of the common causes is a possible "explanandum" - a part of the
      whole explanation for the entire output.

    Args:
      counts, sums, terms
      mask
      confidence:
        Used to calculate thresholds for near-zero in the values.
    Returns:
      - description - terse textual description of top explanandum, or None if no explanandum found
      - fraction - fraction of values that are at least partially explained by this explanandum, or None
    """
    # parse args
    counts, sums, terms_list = me.standardize(counts, sums, terms, mask=mask)

    # determine threshold
    # - construct original output value
    # - get percentile
    value = np.sum(sums, axis=-1)
    threshold = tfp.stats.percentile(tf.abs(value), 100 * (1 - confidence), interpolation='linear').numpy()

    if len(terms_list) == 3:
        return _describe_tensor_near_zero_explanandum(counts, sums, terms_list)
    else:
        return _describe_matmul_nero_zero_explanandum(counts, sums, terms_list, threshold)


def _describe_tensor_near_zero_explanandum(counts, sums, terms_list):
    """
    Internal function for describe_top_near_zero_explanandum().
    Currently only considers a single explanandum: that some of the values are near-zero.
    """
    term_index = terms_list.index('Z')
    total_count = np.sum(counts)
    count_zero = np.sum(counts[..., term_index])
    fraction = count_zero / total_count
    return f"{fraction * 100:.1f}% near-zero", fraction


def _describe_matmul_nero_zero_explanandum(counts, sums, terms_list, threshold):
    """
    Internal function for describe_top_near_zero_explanandum()
    Considers how different combinations of terms can contribute to the outcome.
    More terms increases probability of them contributing to the outcome,
    so we must normalise. It's something like:
      For each explanandum, we're measuring:
         P(outcome,explanandum)
      So we normalize via:
         P(outcome|explanandum) = P(outcome|explanandum) / P(explanandum)
      And then we arg-max over the explanandums to find the one that leads to the highest P(outcome|explandum)
    """
    total_count = np.sum(counts)

    def _term_indices(terms_list, wanted):
        return [terms_list.index(term) for term in wanted]

    def _zero_effect(selected_terms, description):
        count = np.sum(counts[..., _term_indices(terms_list, selected_terms)])
        fraction = count / total_count
        strength = count / len(selected_terms)
        return fraction, strength, description

    def _pos_neg_effect(selected_terms, description):
        # search for instances where the sum over the terms is near-zero then sum up the counts over just those
        # matching instances. note: when computing strength, don't need to scale by number of hits because that's
        # already achieved by masking against the source_count
        pos_neg_value = np.sum(sums[..., _term_indices(terms_list, selected_terms)], axis=-1)
        pos_neg_count = np.sum(counts[..., _term_indices(terms_list, selected_terms)], axis=-1)
        mask = (np.abs(pos_neg_value) < threshold) | (pos_neg_value == 0)  # shape: value_shape x bool
        source_count = np.sum(pos_neg_count * mask)

        fraction = source_count / total_count
        strength = source_count / len(selected_terms)
        return fraction, strength, description

    # measure effect of each individual explanandum
    explanandums = [_zero_effect(['ZZ', 'ZP', 'ZN'], "near-zero values from first input"),
                    _zero_effect(['ZZ', 'PZ', 'NZ'], "near-zero values from second input"),
                    _zero_effect(['ZZ', 'ZP', 'PZ', 'ZN', 'NZ'], "near-zero values from either input"),
                    _pos_neg_effect(['PP', 'PN'], "positive/negatives from second input cancelling out (PP ~= PN)"),
                    _pos_neg_effect(['NP', 'NN'], "positive/negatives from second input cancelling out (NP ~= NN)"),
                    _pos_neg_effect(['PP', 'NP'], "positive/negatives from first input cancelling out (PP ~= NP)"),
                    _pos_neg_effect(['PN', 'NN'], "positive/negatives from first input cancelling out (PN ~= NN)"),
                    _pos_neg_effect(['PP', 'NN', 'NP', 'PN'],
                                    "sums of positive/negatives from both inputs cancelling out (PP+NN ~= NP+PN)")]

    # pick strongest explanandum
    # - pick highest strength first, and then maximum fraction for tie-breaker
    index = np.argmax([strength + fraction for fraction, strength, description in explanandums])
    fraction, strength, description = explanandums[index]

    if fraction == 0.0:
        return None, None
    else:
        return description, fraction
