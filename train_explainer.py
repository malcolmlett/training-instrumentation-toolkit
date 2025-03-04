import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import conv_tools
import train_instrumentation as tinstr
import matmul_explainer as mmexpl


def explain_near_zero_gradients(callbacks: list,
                                layer_index: int,
                                epoch=None, step=None,
                                confidence: float = 0.99,
                                threshold: float = None,
                                verbose=False):
    """
    Attempts to identify the explanation for zero and near-zero gradients in a given layer.

    Callbacks must be populated with data from a call to fit(). Callbacks must
    be populated with full raw data collection for at least the layer in question
    plus:
    - activity: also must capture raw activity for the input layers to the layer in question.

    Args:
        callbacks: List of callbacks as supplied to fit() and now populated with data.
          See above for additional requirements.
        layer_index: layer to examine
        epoch: epoch number selection from callback histories, if callbacks captured against epochs
        step: step number selection from callback histories, if callbacks captured against update steps
        confidence: confidence level for near-zero classification throughout
            Ignored for the selected layer gradients if `threshold` is set, but still used for other thresholds
        threshold: this layer gradients equal or below this magnitude are considered "near-zero".
            Not used for other thresholding.
        verbose: whether to show more detailed information.

    Usage:
    > l_idx = 35
    > variables = VariableHistoryCallback(collection_sets=[{'layer_indices': [l_idx]}], before_updates=True)
    > gradients = GradientHistoryCallback(collection_sets=[{'layer_indices': [l_idx]}])
    > outputs = LayerOutputHistoryCallback(collection_sets=[{'layer_indices': [l_idx-1, l_idx]}], batch_reduction=None)
    > output_gradients = LayerOutputGradientHistoryCallback(collection_sets=[{'layer_indices': [l_idx]}], batch_reduction=None)
    > fit(model, train_data, callbacks=[variables, gradients, outputs, output_gradients])
    > explain_near_zero_gradients([variables, gradients, outputs, output_gradients], l_idx, epoch=...)
    """

    # parse arguments - extract individual callbacks by type
    variables = None
    gradients = None
    outputs = None
    output_gradients = None
    for cb in callbacks:
        if reload_safe_isinstance(cb, tinstr.VariableHistoryCallback):
            variables = cb
        elif reload_safe_isinstance(cb, tinstr.LayerOutputHistoryCallback):
            outputs = cb
        elif reload_safe_isinstance(cb, tinstr.GradientHistoryCallback):
            gradients = cb
        elif reload_safe_isinstance(cb, tinstr.LayerOutputGradientHistoryCallback):
            output_gradients = cb

    # validations
    missing_infos = []
    if variables is None:
        missing_infos.append("variable callback not supplied")
    elif variables.variables is None:
        missing_infos.append("variable values not collected during training")

    if gradients is None:
        missing_infos.append("gradients callback not supplied")
    elif gradients.gradients is None:
        missing_infos.append("gradients not collected during training")

    if outputs is None:
        missing_infos.append("layer output callback not supplied")
    elif outputs.layer_outputs is None:
        missing_infos.append("layer outputs not collected during training")
    elif outputs.batch_reduction is not None:
        missing_infos.append("layer outputs were collected with batch reduction (must use batch_reduction=None)")
        outputs = None

    if output_gradients is None:
        missing_infos.append("output gradients callback not supplied")
    elif output_gradients.gradients is None:
        missing_infos.append("output gradients not collected during training")
    elif output_gradients.batch_reduction is not None:
        missing_infos.append("output gradients were collected with batch reduction (must use batch_reduction=None)")
        output_gradients = None

    # parse arguments - identify iteration number
    if epoch is not None:
        indices = np.nonzero(gradients.epochs == epoch)[0]
        if len(indices) == 0:
            raise ValueError(f"Epoch {epoch} not found in gradient history")
        iteration = indices[0]
    elif step is not None:
        indices = np.nonzero(gradients.steps == step)[0]
        if len(indices) == 0:
            raise ValueError(f"Step {step} not found in gradient history")
        iteration = indices[0]
    else:
        raise ValueError("One of epoch or step must be specified")

    # get model and identify inbound and outbound layers
    model = gradients.model
    target_layer = model.layers[layer_index]
    inbound_layer_indices = _find_inbound_layers(model, target_layer, return_type='index')
    outbound_layer_indices = _find_outbound_layers(model, target_layer, return_type='index')
    l_to_var_indices = tinstr.variable_indices_by_layer(model, include_trainable_only=False)
    if not inbound_layer_indices:
        raise ValueError(f"Layer #{layer_index} has no inbound layers")
    if not outbound_layer_indices:
        raise ValueError(f"Layer #{layer_index} has no outbound layers")

    def _get_layer_handler(l_idx, subscript, warnings_out=None):
        """
        This complex looking function just does the "None-safe" equivalent of:
          layerSvars = [variables.variables[var_idx][iteration] for var_idx in l_to_var_indices[l_idx]]
          layer_grads = [gradients.gradients[var_idx][iteration] for var_idx in l_to_var_indices[l_idx]]
          layer_inps = [outputs.layer_outputs[l_idx2][iteration] for l_idx2 in inbound_layer_indices]
          layer_out = outputs.layer_outputs[l_idx][iteration]
          layer_out_grads = output_gradients.gradients[l_idx][iteration]
        """
        if warnings_out is None:
            warnings_out = []
        layer = model.layers[l_idx]
        layer_vars = []
        layer_grads = []
        layer_inps = []
        layer_out = None
        layer_out_grads = None
        for var_idx in l_to_var_indices[l_idx]:
            if variables and variables.variables is not None:
                if variables.variables[var_idx] is None:
                    warnings_out.append(f"variable {var_idx} not collected")
                else:
                    layer_vars.append(variables.variables[var_idx][iteration])
            if gradients and gradients.gradients is not None:
                if gradients.gradients[var_idx] is None:
                    warnings_out.append(f"gradients of variable {var_idx} not collected")
                else:
                    layer_grads.append(gradients.gradients[var_idx][iteration])
        for l_idx0 in inbound_layer_indices:
            if outputs and outputs.layer_outputs is not None:
                if outputs.layer_outputs[l_idx0] is None:
                    warnings_out.append(f"layer {l_idx0} outputs not collected")
                else:
                    layer_inps.append(outputs.layer_outputs[l_idx0][iteration])
        if outputs and outputs.layer_outputs is not None:
            if outputs.layer_outputs[l_idx] is None:
                warnings_out.append(f"layer {l_idx} outputs not collected")
            else:
                layer_out = outputs.layer_outputs[l_idx][iteration]
        if output_gradients and output_gradients.gradients is not None:
            if output_gradients.gradients[l_idx] is None:
                warnings_out.append(f"output gradients for layer {l_idx} not collected")
            else:
                layer_out_grads = output_gradients.gradients[l_idx][iteration]
        return get_layer_handler_for(
            layer=layer, layer_index=l_idx, layer_subscript=subscript, return_note=True, variables=layer_vars,
            gradients=layer_grads, inputs=layer_inps, output=layer_out, output_gradients=layer_out_grads)

    # get layer handlers
    # TODO do something with handler notes
    # TODO incorporate extra error checking that was in _estimate_backprop_from_layer()
    # TODO don't bother trying to get next_layer_handlers unless needed
    target_layer_handler, target_layer_handler_note = _get_layer_handler(layer_index, 'l', missing_infos)
    next_layer_handlers_and_notes = [_get_layer_handler(l_idx, '2') for l_idx in outbound_layer_indices]
    next_layer_handlers = [handler for handler, note in next_layer_handlers_and_notes]
    next_layer_handler_notes = [note for handler, note in next_layer_handlers_and_notes]

    # sanity check that minimum required information is present
    def produces_error(func):
        try:
            func()
            return False
        except:
            return True
    if produces_error(target_layer_handler.get_weight_gradients):
        print(f"Warning - unable to provide accurate report because {', '.join(missing_infos)}")
        print(f"Error - No gradients found for target layer")
        return
    if produces_error(target_layer_handler.get_weights):
        print(f"Warning - unable to provide accurate report because {', '.join(missing_infos)}")
        print(f"Error - No weights found for target layer")
        return
    if produces_error(target_layer_handler.get_A):
        print(f"Warning - unable to provide accurate report because {', '.join(missing_infos)}")
        print(f"Error - No activations found for target layer")
        return

    def _explain_tensor(name, tensor, mask_name=None, mask=None, negatives_are_bad=False, include_summary_by_unit=False):
        quantiles = [0, 25, 50, 75, 100]
        counts, sums, threshold = mmexpl.tensor_classify(tensor, confidence=confidence, return_threshold=True)
        summaries, _ = describe_tensor_near_zero_explanation(counts, sums, threshold=threshold, negatives_are_bad=negatives_are_bad, verbose=verbose)
        if not verbose:
           summaries = [summaries[0]]  # pick first only
        if include_summary_by_unit:
            extra, extra_fractions = describe_tensor_units_near_zero_explanation(counts, sums, threshold=threshold, negatives_are_bad=negatives_are_bad, verbose=verbose)
            if verbose:
                summaries.extend(extra)
            elif extra_fractions[0] > (1.1 * (1 - confidence)):
                summaries.append(extra[0])
        print(f"{name}:")
        print(f"  shape:             {np.shape(tensor)} -> total values: {np.size(tensor)}")
        print(f"  summary:           {', '.join(summaries)}")
        if verbose:
            print(f"  value percentiles: {quantiles} -> {tfp.stats.percentile(tensor, quantiles).numpy()}")
            print(f"  PZN counts/sums:   {mmexpl.summarise(counts, sums, show_percentages=False, show_means=False)}")
            print(f"  PZN counts/means:  {mmexpl.summarise(counts, sums, show_percentages=True, show_means=True)}")
        if mask_name:
            mask_summaries, _ = describe_tensor_near_zero_explanation(counts, sums, mask=mask, threshold=threshold, negatives_are_bad=negatives_are_bad, verbose=verbose)
            if not verbose:
              mask_summaries = [mask_summaries[0]]  # pick first only
            print(f"  {(mask_name + ':'):19} {', '.join(mask_summaries)}")
        if mask_name and verbose:
            print(f"  {(mask_name + ':'):19} {mmexpl.summarise(counts, sums, mask=mask, show_percentages=True, show_means=True)}")

    def _explain_actual_vs_estimate(name, actual, estimate):
        error = estimate - actual
        quantiles = [0, 25, 50, 75, 100]
        print(f"{name} estimate vs actual:")
        print(f"  RMSE:              {np.sqrt(np.mean(error**2))}")
        print(f"  error percentiles: {quantiles} -> {tfp.stats.percentile(error, quantiles).numpy()}")

    def _explain_combination(name, inputs, result, counts, sums, fixed_threshold=None):
        # note: has noteworthy level of zeros if exceed 1.1x the percent we'd expect from the confidence level alone.
        # That gives a little leeway for rounding errors.
        print(f"compute {name}:")
        print(f"  inputs examined:   {', '.join(inputs)}")
        mask, threshold = _mask_near_zero(result, confidence=confidence, fixed_threshold=fixed_threshold,
                                          return_threshold=True)
        orig_total = np.size(counts[..., 0])
        mask_total = np.sum(mask)
        fraction = mask_total / orig_total
        noteworthy = fraction > (1.1 * (1 - confidence))
        if noteworthy:
            top_description, top_fraction = describe_top_near_zero_explanandum(counts, sums, input_names=inputs, mask=mask, threshold=threshold)
            print(f"  summary:           {fraction * 100:.1f}% near-zero, of which {top_fraction * 100:.1f}% are affected by {top_description}")
        else:
            print(f"  summary:           {fraction * 100:.1f}% near-zero")
        if verbose:
            print(f"  PZN combinations:  {mmexpl.summarise(counts, sums, show_percentages=False, show_means=False)}")
            print(f"  PZN combinations:  {mmexpl.summarise(counts, sums, show_percentages=True, show_means=True)}")

    def _explain_combination_near_zero(name, inputs, result, counts, sums, fixed_threshold=None):
        if not verbose:
            return  # skip whole section if not in verbose mode
        mask, threshold = _mask_near_zero(result, confidence=confidence, fixed_threshold=fixed_threshold,
                                          return_threshold=True)
        orig_total = np.size(counts[..., 0])
        mask_total = np.sum(mask)
        print(f"{name} {mask_total / orig_total * 100:.1f}% near zero ({_format_threshold(threshold)}):")
        print(f"  PZN combinations:  {mmexpl.summarise(counts, sums, mask=mask, show_percentages=False, show_means=False)}")
        print(f"  PZN combinations:  {mmexpl.summarise(counts, sums, mask=mask, show_percentages=True, show_means=True)}")

        # preferred detail - list of causal explanandums
        descriptions, fractions = describe_near_zero_explanation(counts, sums, input_names=inputs, mask=mask, threshold=threshold)
        if descriptions:
            for description, fraction in zip(descriptions, fractions):
                print(f"  {fraction*100:.1f}% affected by {description}")

        # fall-back detailed description - filtered and grouped classifications
        # (less visually pleasing, so only using as fallback)
        if not descriptions:
            counts, sums, terms = mmexpl.filter_classifications(counts, sums)
            count_groups, sum_groups, term_groups = mmexpl.group_classifications(counts, sums, terms, mask=mask)
            count_groups, sum_groups, term_groups, coverage = mmexpl.filter_groups(
                count_groups, sum_groups, term_groups, completeness=0.75, max_groups=10, return_coverage=True)
            desc = mmexpl.describe_groups(count_groups, sum_groups, term_groups, show_ranges=True)
            for size, summary in zip(desc['sizes'], desc['summaries']):
                print(f"  {size / mask_total * 100:.1f}% explained as (counts/sums): {summary}")
            print(f"  (explains top most {coverage * 100:.1f}% of near-zero)")

    # identify points of interest
    dJdW_l = target_layer_handler.get_weight_gradients()
    near_zero_gradients_mask, near_zero_gradients_threshold = _mask_near_zero(
        dJdW_l, confidence, threshold, return_threshold=True)
    num_near_zero_gradients = np.sum(near_zero_gradients_mask)
    num_gradients = np.size(dJdW_l)

    # explain context and give summary
    print(f"Examining layer #{layer_index} at iteration {iteration}:")
    print(f"  layer:               {target_layer_handler.layer.name}")
    print(f"  units:               {target_layer_handler.get_weights().shape[-1]}")
    print(f"  weights:             {tf.size(target_layer_handler.get_weights())}")
    print(f"  input from layers:   {inbound_layer_indices}")
    print(f"  output to layers:    {outbound_layer_indices}")
    print(f"  near-zero gradients: {num_near_zero_gradients} ({num_near_zero_gradients / num_gradients * 100:0.1f}%) "
          f"{_format_threshold(near_zero_gradients_threshold)}")
    if missing_infos:
        print(f"  note:                unable to provide accurate report because {', '.join(missing_infos)}")

    print()
    print("Let:")
    print("  A_0      <- input activation to target layer")
    print("  W_l      <- weights of target layer")
    print("  Z_l      <- pre-activation output of target layer")
    print("  S_l      <- effect of activation function as element-wise multiplier")
    print("  A_l      <- output activation of target layer")
    if verbose:
        print("  PZN      <- breakdown of values into (P)ositive, near-(Z)ero, or (N)egative")
        print("  x . y    <- matmul (aka dot product)")
        print("  x (.) y  <- element-wise multiply")

    # explain forward pass
    # - let A_0 be input activation to this layer (not necessarily the first layer input)
    print()
    print(f"Forward pass...")
    for idx, A_0 in enumerate(target_layer_handler.inputs):
        name = "A_0" if len(target_layer_handler.inputs) == 1 else f"A_0#{idx}"
        _explain_tensor(name + ' - input value', A_0, include_summary_by_unit=True)

    W_l =
    _explain_tensor("W_l - weights of target layer", W_l,
                    "corresponding to near-zero gradients", near_zero_gradients_mask)

    try:
        # typically: Z_l = A_0 . W_l + b_l
        Z_l, Z_l_eqn, Z_l_note = target_layer_handler.calculate_Z(return_equation=True, return_note=True)
        counts, sums = target_layer_handler.classify_Z_calculation(confidence=confidence)
        _explain_combination(f"Z_l = {Z_l_eqn}", ['A_0', 'W_l'], Z_l, counts, sums)
        _explain_combination_near_zero("Z_l", ['A_0', 'W_l'], Z_l, counts, sums)
        _explain_tensor("Z_l - pre-activation output", Z_l, negatives_are_bad=True, include_summary_by_unit=True)
        if Z_l_note and verbose:
            print(f"  note:              {Z_l_note}")
    except UnsupportedNetworkError as e:
        print(f"Z_l - pre-activation output:")
        print(f"  note:              Unable to infer due to {e}")

    A_l = target_layer_handler.get_A()
    try:
        S_l, S_l_note = target_layer_handler.calculate_S(return_note=True)
        _explain_tensor("S_l - activation function", S_l, include_summary_by_unit=True)
        if S_l_note and verbose:
            print(f"  note:              {S_l_note}")
    except UnsupportedNetworkError as e:
        print(f"S_l - activation function:")
        print(f"  note:              Unable to infer due to {e}")

    _explain_tensor("A_l - activation output from target layer", A_l)

    # explain backprop pass
    # - starts with dJdA_l
    # - compute dJ/dZ_l = dJ/dA_l x S_l (element-wise)
    # - compute dJ/dW_l = A_0^T . dJ/dZ_l    <-- note: experimentally this produces very accurate results.
    print()
    print(f"Backward pass...")
    dJdA_l = target_layer_handler.output_gradients
    if dJdA_l is not None:
        # dJdA_l has been directly measured, so work off that
        _explain_tensor(f"dJ/dA_l - backprop into this layer", dJdA_l, include_summary_by_unit=True)
    else:
        # dJdA_l wasn't measured directly, but we can estimate it instead if we have data from the next layer.
        # estimate dJdA_l by estimating backprop from each output layer and then combining
        # - unlikely to need to do this, but I wrote this code before I realised I could directly extract output
        #   gradients
        dJdA_l_list = []
        for idx, next_layer_handler in enumerate(next_layer_handlers):
            suffix = "" if len(next_layer_handlers) == 1 else f"#{idx}"
            try:
                backprop, note = next_layer_handler.calculate_backprop(return_note=True)
                _explain_tensor(f"dJ/dA_l{suffix} - backprop from next layer {next_layer_handler.display_name}",
                                backprop, include_summary_by_unit=True)
                if note and verbose:
                    print(f"  note:              {note}")
                elif verbose:
                    print(f"  note:              estimated via weights and gradients in next layer")
                dJdA_l_list.append(backprop)
            except UnsupportedNetworkError as e:
                print(f"dJ/dA_l{suffix} - backprop from next layer {next_layer_handler.display_name}")
                print(f"  note:              unable to infer due to {e}")
                # ignore and continue on with backprops received from any other next layers
        # if multiple next layers, backprop to this layer is mean across their individual backprops
        if len(dJdA_l_list) > 0:
            dJdA_l = np.mean(np.stack(dJdA_l_list, axis=-1), axis=-1)
        else:
            dJdA_l = None
        if len(dJdA_l_list) > 1:
            _explain_tensor(f"dJ/dA_l - final backprop into this layer", dJdA_l, include_summary_by_unit=True)

    if dJdA_l is None:
        print(f"compute dJ/dZ_l:")
        print(f"  note:              Unable to infer")
        dJdZ_l = None
    else:
        # typically: dJ/dZ_l = dJ/dA_l (.) S_l   -- element-wise multiply
        try:
            dJdZ_l, dJdZ_l_eqn, dJdZ_l_note = target_layer_handler.calculate_dJdZ(
                dJdA_l, return_equation=True, return_note=True)
            counts, sums = target_layer_handler.classify_dJdZ_calculation(dJdA_l, confidence=confidence)
            _explain_combination(f"dJ/dZ_l = {dJdZ_l_eqn}", [f"dJ/dA_l", 'S_l'], dJdZ_l, counts, sums)
            _explain_combination_near_zero("dJ/dZ_l", ['dJ/dA_l', 'S_l'], dJdZ_l, counts, sums)
            _explain_tensor("dJ/dZ_l", dJdZ_l, include_summary_by_unit=True)
            if dJdZ_l_note and verbose:
                print(f"  note:              {dJdZ_l_note}")
        except UnsupportedNetworkError as e:
            print(f"compute dJ/dZ_l:")
            print(f"  note:              Unable to infer due to {e}")
            dJdZ_l = None

    dJdW_l_est = None
    if dJdZ_l is None:
        print(f"compute dJ/dW_l:")
        print(f"  note:              Unable to infer without dJ/dZ_l")
    else:
        # typically: dJ/dW_l = A_0^T . dJ/dZ_l
        # - using actual dJdW_l for determination of thresholds
        dJdW_l = target_layer_handler.get_weight_gradients()
        try:
            dJdW_l_est, dJdW_l_eqn, dJdW_l_note = target_layer_handler.calculate_dJdW(
                dJdZ_l, return_equation=True, return_note=True)
            counts, sums = target_layer_handler.classify_dJdW_calculation(dJdZ_l, confidence=confidence)
            _explain_combination(f"dJ/dW_l = {dJdW_l_eqn}", ['A_0^T', 'dJ/dZ_l'], dJdW_l, counts, sums, threshold)
            _explain_combination_near_zero("dJ/dW_l", ['A_0^T', 'dJ/dZ_l'], dJdW_l, counts, sums, threshold)
        except UnsupportedNetworkError as e:
            print(f"compute dJ/dW_l:")
            print(f"  note:              Unable to infer due to {e}")

    dJdW_l = target_layer_handler.get_weight_gradients()
    if dJdW_l is not None and dJdW_l_est is not None and verbose:
        _explain_actual_vs_estimate("dJ/dW_l", dJdW_l, dJdW_l_est)
    _explain_tensor("dJ/dW_l", dJdW_l)


def describe_top_near_zero_explanandum(counts, sums=None, terms=None, *, mask=None, input_names=None, confidence=0.95,
                                       threshold=None):
    """
    Gets the single top explanandum from describe_near_zero_explanation(), if any.
    Args:
        counts: counts from matmul-like classification, with shape: value_shape + (terms,),
            or supply tensor with (counts, sums, terms)
        sums: counts from matmul-like classification, with shape: value_shape + (terms,)
        terms: terms from matmul-like classification, with shape: value_shape + (terms,)
        mask: boolean mask against, with shape: value_shape
        input_names: names of left and right inputs, to be used in descriptions.
            Defaults to using 'first input' and 'second input'.
        confidence: used to calculate thresholds for near-zero values in the resultant value tensor.
            For best results, supply the same confidence that was used when masking.
            Alternatively, supply threshold.
        threshold: fixed threshold to use for determining near-zero values in the resultant value tensor.
    Returns:
      - description - terse textual description of top explanandum, or None if no explanandum found
      - fraction - fraction of values that are at least partially explained by this explanandum, or None
    """
    descriptions, fractions = describe_near_zero_explanation(counts, sums, terms, mask=mask, input_names=input_names,
                                                             confidence=confidence, threshold=threshold)
    if descriptions:
        return descriptions[0], fractions[0]
    else:
        return None, None


def describe_near_zero_explanation(counts, sums=None, terms=None, *, mask=None, input_names=None, confidence=0.95,
                                   threshold=None):
    """
    Tries to explain the cause for near-zero values after a matmul-like operation.
    The explanation is given in the form of a collection of "explanandums" - individual partial explanations
    that together might form the whole explanation (or still only a fraction of the full explanation).

    Examines several common causes of near-zero values and selects those that are able to explain at least
    part of the outcome.
    Results are ordered with the "strongest individual causal power" first - ie: roughly, the simplest explanation that
    explains the most.
    Results are not mutually exclusive.

    Generally you will supply a mask to pick the values being explained, otherwise it assumes
    all output values were near-zero. Note that the confidence/threshold args only used for some
    internal logic, not for automatic masking.

    Terminology note:
      Each of the common causes is a possible "explanandum" - a part of the
      whole explanation for the entire output.

    Args:
        counts: counts from matmul-like classification, with shape: value_shape + (terms,),
            or supply tensor with (counts, sums, terms)
        sums: counts from matmul-like classification, with shape: value_shape + (terms,)
        terms: terms from matmul-like classification, with shape: value_shape + (terms,)
        mask: boolean mask against, with shape: value_shape
        input_names: names of left and right inputs, to be used in descriptions.
            Defaults to using 'first input' and 'second input'.
        confidence: used to calculate thresholds for near-zero values in the resultant value tensor.
            For best results, supply the same confidence that was used when masking.
            Alternatively, supply threshold.
        threshold: fixed threshold to use for determining near-zero values in the resultant value tensor.
    Returns:
        - descriptions - list of terse textual descriptions, one for each explanandum
        - fractions - fraction of values that are at least partially explained by each description.
            Same length as descriptions.
    """
    # parse args
    orig_counts, orig_sums, orig_terms = counts, sums, terms
    counts, sums, terms_list = mmexpl.standardize(counts, sums, terms, mask=mask)
    if len(terms_list) != 9:
        raise ValueError("Not a matmul-like classification")
    if input_names and len(input_names) != 2:
        raise ValueError(f"Must be exactly two input_names, got {len(input_names)}")
    if input_names is None:
        input_names = ['first input', 'second input']
    total_count = np.sum(counts)

    # determine threshold
    # - construct original output value before masking
    # - get percentile
    if threshold is None:
        orig_counts, orig_sums, orig_terms = mmexpl.standardize(orig_counts, orig_sums, orig_terms)
        value = np.sum(orig_sums, axis=-1)
        threshold = tfp.stats.percentile(tf.abs(value), 100 * (1 - confidence), interpolation='linear').numpy()

    # Logic:
    # Considers how different combinations of terms can contribute to the outcome.
    # More terms increases probability of them contributing to the outcome,
    # so we must normalise. It's something like:
    #   For each explanandum, we're measuring:
    #      P(outcome,explanandum)
    #   So we normalize via:
    #      P(outcome|explanandum) = P(outcome|explanandum) / P(explanandum)
    #   And then we arg-max over the explanandums to find the one that leads to the highest P(outcome|explanandum)
    def _zero_effect(selected_terms, description):
        term_indices = [terms_list.index(term) for term in selected_terms]
        count = np.sum(counts[..., term_indices])
        fraction = count / total_count
        strength = count / len(selected_terms)
        return fraction, strength, description

    def _pos_neg_effect(selected_terms, description):
        # search for instances where the sum over the terms is near-zero then sum up the counts over just those
        # matching instances. note: when computing strength, don't need to scale by number of hits because that's
        # already achieved by masking against the source_count
        term_indices = [terms_list.index(term) for term in selected_terms]
        pos_neg_value = np.sum(sums[..., term_indices], axis=-1)
        pos_neg_count = np.sum(counts[..., term_indices], axis=-1)
        res_mask = (np.abs(pos_neg_value) < threshold) | (pos_neg_value == 0)  # shape: value_shape x bool
        count = np.sum(pos_neg_count * res_mask)

        fraction = count / total_count
        strength = count / len(selected_terms)
        return fraction, strength, description

    # measure effect of each individual explanandum
    explanandums = [_zero_effect(['ZZ', 'ZP', 'ZN'], f"near-zero values from {input_names[0]}"),
                    _zero_effect(['ZZ', 'PZ', 'NZ'], f"near-zero values from {input_names[1]}"),
                    _zero_effect(['ZZ', 'ZP', 'PZ', 'ZN', 'NZ'], "near-zero values from either input"),
                    _pos_neg_effect(['PP', 'NP'],
                                    f"positive/negatives from {input_names[0]} cancelling out (PP ~= NP)"),
                    _pos_neg_effect(['PN', 'NN'],
                                    f"positive/negatives from {input_names[0]} cancelling out (PN ~= NN)"),
                    _pos_neg_effect(['PP', 'PN'],
                                    f"positive/negatives from {input_names[1]} cancelling out (PP ~= PN)"),
                    _pos_neg_effect(['NP', 'NN'],
                                    f"positive/negatives from {input_names[1]} cancelling out (NP ~= NN)"),
                    _pos_neg_effect(['PP', 'NN', 'NP', 'PN'],
                                    "sums of positive/negatives from both inputs cancelling out (PP+NN ~= NP+PN)")]

    # filter explanandums
    # - keep only those with non-zero strength and non-zero fraction
    explanandums = [(fraction, strength, description) for fraction, strength, description in explanandums
                    if strength > 0 and fraction > 0]

    # sort explanandums
    # - highest strength first, and then maximise fraction for tie-breaker
    explanandums = sorted(explanandums, key=lambda x: x[0] + x[1], reverse=True)

    # return results
    descriptions = [description for fraction, strength, description in explanandums]
    fractions = [fraction for fraction, strength, description in explanandums]
    return descriptions, fractions


def describe_tensor_near_zero_explanation(counts, sums=None, *, mask=None, threshold=None, negatives_are_bad=False,
                                          verbose=True):
    """
    Gets a terse description of the tensor in relation to near-zero values, and optionally also in relation
    to negative values.
    Similar in spirit to describe_near_zero_explanation() though the behaviour is somewhat different.
    Args:
        counts: counts from tensor_classify(), with shape: value_shape + (terms,),
            or supply tensor with (counts, sums)
        sums: counts from tensor_classify(), with shape: value_shape + (terms,)
        mask: boolean mask against, with shape: value_shape
        threshold: indication of the threshold that was used for masking, used for display purposes only
        negatives_are_bad: whether negative values are equivalent to zeros
        verbose: whether to include extra working details in the description
    Returns:
        - descriptions - list of textual descriptions, most significant first
        - fractions - fraction of values that each description applies to.

    """
    # parse args
    counts, sums, _ = mmexpl.standardize(counts, sums, mask=mask)
    if counts.shape[-1] != 3:
        raise ValueError("Not a tensor classification")

    # compute details for zeros and negatives
    zero_fraction = np.sum(counts[..., 1]) / np.sum(counts)
    neg_fraction = 0.0
    if negatives_are_bad:
        # strictly include all negatives, even those close to zero
        value = np.sum(sums, axis=-1)
        neg_fraction = np.sum(value < 0) / np.size(value)

    # generate descriptions
    summaries = []
    if verbose and threshold is not None:
        summaries.append((f"{zero_fraction * 100:.1f}% near-zero ({_format_threshold(threshold)})", zero_fraction))
    else:
        summaries.append((f"{zero_fraction * 100:.1f}% near-zero", zero_fraction))
    if neg_fraction > 0:
        summaries.append((f"{neg_fraction * 100:.1f}% negative", neg_fraction))

    # order summaries, highest importance first
    summaries = sorted(summaries, key=lambda x: x[1], reverse=True)

    # return results
    descriptions = [description for description, fraction in summaries]
    fractions = [fraction for description, fraction in summaries]
    return descriptions, fractions


def describe_tensor_units_near_zero_explanation(counts, sums=None, *, mask=None, threshold=None,
                                                negatives_are_bad=False, verbose=True):
    """
    Similar to describe_tensor_near_zero_explanation() but treats the tensor as having
    shape `(batch, ..spatial_dims.., units)` and calculates on a per-unit basis before reporting
    overal stats.
    Args:
        counts: counts from tensor_classify(), with shape: value_shape + (terms,),
            or supply tensor with (counts, sums)
        sums: counts from tensor_classify(), with shape: value_shape + (terms,)
        mask: boolean mask against, with shape: value_shape
        threshold: indication of the threshold that was used for masking, used for display purposes only
        negatives_are_bad: whether negative values are equivalent to zeros
        verbose: whether to include more information (currently has no effect)
    Returns:
        list of textual descriptions, approximately with highest-degree summary first
    """
    # parse args
    counts, sums, _ = mmexpl.standardize(counts, sums, mask=mask)
    if counts.shape[-1] != 3:
        raise ValueError("Not a tensor classification")

    # compute details for zeros and negatives (as appropriate)
    if negatives_are_bad:
        zero_mask = np.logical_or(counts[..., 1] > 0, counts[..., 2] > 0)
    else:
        zero_mask = counts[..., 1] > 0

    # compute rates of occurrences of zero/neg across each unit
    # - per-unit fraction of zero/neg values
    # - overall fraction of units that are dead 100% of the time
    # - mean fraction across all units is the same as the whole-tensor near-zero rate, so don't report on it
    zero_rates = np.mean(zero_mask, axis=tuple(range(zero_mask.ndim-1)))
    dead_fraction = np.sum(zero_rates == 1.0) / np.size(zero_rates)

    condition_txt = "near-zero or negative" if negatives_are_bad else "near-zero"
    dims_txt = "batch" if counts.ndim <= 3 else "batch and spatial"

    # return summaries
    # - currently only one, but if we had more we'd need to sort them
    descriptions = [f"{dead_fraction * 100:.1f}% of units always {condition_txt} across all {dims_txt} dims"]
    fractions = [dead_fraction]
    return descriptions, fractions


def get_layer_handler_for(layer, layer_index, variables, gradients, inputs, output, output_gradients,
                          layer_subscript=None, return_note=False):
    """
    Factory for layer handlers. Identifies and instantiates the best layer handler for the given layer.
    Returns:
        - handler - a LayerHandler instance
        - note - a warning note, or none if no warnings (optional output)
    """
    display_name = f"{layer.name} (#{layer_index})"
    note = None
    if isinstance(layer, tf.keras.layers.Dense):
        handler = DenseLayerHandler(
            display_name=display_name, layer=layer, variables=variables, gradients=gradients, inputs=inputs,
            output=output, output_gradients=output_gradients, layer_subscript=layer_subscript)
    elif 'dense' in layer.name:
        handler = DenseLayerHandler(
            display_name=display_name, layer=layer, variables=variables, gradients=gradients, inputs=inputs,
            output=output, output_gradients=output_gradients, layer_subscript=layer_subscript)
        note = "Treating as standard Dense layer due to name, results may not be accurate"
    elif isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D, tf.keras.layers.Conv3D)):
        handler = ConvLayerHandler(
            display_name=display_name, layer=layer, variables=variables, gradients=gradients, inputs=inputs,
            output=output, output_gradients=output_gradients, layer_subscript=layer_subscript)
    elif 'conv' in layer.name:
        handler = ConvLayerHandler(
            display_name=display_name, layer=layer, variables=variables, gradients=gradients, inputs=inputs,
            output=output, output_gradients=output_gradients, layer_subscript=layer_subscript)
        note = "Treating as standard Conv layer due to name, results may not be accurate"
    else:
        # fallback to generic handler
        handler = LayerHandler(
            display_name=display_name, layer=layer, variables=variables, gradients=gradients, inputs=inputs,
            output=output, output_gradients=output_gradients, layer_subscript=layer_subscript)

    if return_note:
        return handler, note
    else:
        return handler


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


# no longer in use
# TODO take key error messages and merge with new code
def _estimate_backprop_from_layer(model, layer_index, iteration,
                                  gradients: tinstr.GradientHistoryCallback,
                                  activity: tinstr.LayerOutputHistoryCallback,
                                  variables: tinstr.VariableHistoryCallback):
    print(f"[BEGIN] _estimate_backprop_from_layer(layer_index={layer_index})")

    # get model and identify inbound and outbound layers
    model = gradients.model
    target_layer = model.layers[layer_index]
    inbound_layer_indices = _find_inbound_layers(model, target_layer, return_type='index')
    if not inbound_layer_indices:
        raise UnsupportedNetworkError(f"Layer has no inbound layers")
    if len(inbound_layer_indices) > 1:
        raise UnsupportedNetworkError(f"Layer has multiple inbound layers")
    # print(f"layer: #{layer_index} = {target_layer} at iteration {iteration}")
    # print(f"inbound_layer_indices: {inbound_layer_indices}")

    # pre-compute lookups
    l_to_var_indices = tinstr.variable_indices_by_layer(model, include_trainable_only=False)
    # print(f"l_to_var_indices: {l_to_var_indices}")

    # get variables, activities, and gradients of interest
    _, _, target_var_idx, target_rest_var_indices = _split_by_largest(target_layer.variables, l_to_var_indices[layer_index])
    if variables.variables[target_var_idx] is None:
        raise UnsupportedNetworkError(
            f"Weights (selected by heuristic) not collected (not trainable?), model.variables index: {target_var_idx}")
    if gradients.gradients[target_var_idx] is None:
        raise UnsupportedNetworkError(
            f"Gradients of weights (selected by heuristic) not collected (not trainable?), model.variables index: {target_var_idx}")
    target_layer_weights = variables.variables[target_var_idx][iteration]
    target_layer_weights_gradients = gradients.gradients[target_var_idx][iteration]
    # print(f"target_layer_weights: {target_layer_weights.shape}")
    # print(f"target_layer_weights_gradients: {target_layer_weights_gradients.shape}")

    # get data of interest from the inbound prev layers
    prev_layers_activations_list = [activity.layer_outputs[l_idx][iteration] for l_idx in inbound_layer_indices]
    # print(f"prev_layers_activations_list: {[a.shape for a in prev_layers_activations_list]}")
    if not prev_layers_activations_list:
        # should never occur
        raise UnsupportedNetworkError(f"No activations from previous layer")
    if len(prev_layers_activations_list) > 1:
        # should never occur
        raise UnsupportedNetworkError(
            f"Multiple activations from previous layer: {[a.shape for a in prev_layers_activations_list]}")

    # estimate backprop from this layer (dJ/dA_0)
    # - let:
    # - we need the backprop gradients that are produced by this layer and which feed into the gradient update step of
    #   the prev layer.
    # - we don't have that, but instead we have the gradients of the weights at this layer, which is one step in
    #   the wrong direction.
    # - we can estimate the backprop gradients as follows:
    #    given:
    #       0 = prev layer, 1 = this layer
    #       dJ/dW_1 = weight gradients at this layer
    #       dZ_1/dW_1 = A_0^T = transposed activations from previous layer
    #       dZ_1/dA_0 = W_1^T = transposed weights from this layer
    #    want:
    #       dJ/dA_0 = backprop gradients to previous layer
    #    #1:
    #       dJ/dW_1 = dZ_1/dW_1 . dJ/dZ_1 = A_0^T . dJ/dZ_1   <-- (f_0 x f_1) = (f_0 x b) . (b x f_1)
    #       => dJ/dZ_1 = (A_0+)^T . dJ/dW_1                   <-- (b x f_1) = (b x f_0) . (f_0 x f_1)
    #    #2:
    #       dJ/dA_0 = dJ/dZ_1 . dZ_1/dA_0 = dJ/dZ_1 . W_1^T   <-- (b x f_0) = (b x f_1) . (f_1 x f_0)
    # TODO replace with dJdA_0 = next_layer_handler.calculate_backprop()

    # TODO do trickery for tensors with spatial dims
    W_1 = target_layer_weights
    dJdW_1 = target_layer_weights_gradients
    A_0 = prev_layers_activations_list[0]
    A_0_inv = tf.linalg.pinv(A_0)
    print(f"dJdW_1: {dJdW_1.shape}, A_0: {A_0.shape}, A_0_inv: {A_0_inv.shape}")
    dJdZ_1 = tf.linalg.matmul(A_0_inv, dJdW_1, transpose_a=True)
    print(f"dJdZ_1: {dJdZ_1.shape}")
    dJdA_0 = tf.linalg.matmul(dJdZ_1, W_1, transpose_b=True)
    print(f"dJdA_0: {dJdA_0.shape}")

    print(f"[END] _estimate_backprop_from_layer()")
    return dJdA_0


def _mask_near_zero(tensor, confidence, fixed_threshold=None, return_threshold=False):
    if fixed_threshold is None:
        threshold = tfp.stats.percentile(tf.abs(tensor), q=100 * (1 - confidence), interpolation='midpoint').numpy()
    else:
        threshold = fixed_threshold

    if threshold == 0:
        mask = tensor == 0
    else:
        mask = tf.abs(tensor) < threshold

    if return_threshold:
        return mask, threshold
    else:
        return mask


def _format_threshold(threshold):
    if threshold == 0:
        return "= 0"
    else:
        return f"< {threshold}"


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


class UnsupportedNetworkError(Exception):
    """
    Represents that some aspect of the network is not supported by the explainer.
    Generally this results in the explainer falling back to more generic observations.
    """
    pass


class LayerHandler:
    """
    Gracefully degrades wherever possible, making as much information accessible as can be determined.

    Notation: 'A_l' and the like are used throughout to refer to "this" layer, while 'A_0' and the like
    are used to refer to the immediate inputs to this layer. This is not mathematically sound, but
    it's simpler than trying to find a code-friendly notation for 'A_{l-1}'.
    """

    # TODO need to better detect situations where only some of the layer's variables/gradients are available.
    #  Currently the main explain method will silently filter out the unknown variables and this method
    #  won't know they exist. So it's heuristics will produce bad results (eg: assuming there's no bias,
    #  which will affect display of Z calculations).
    def __init__(self, display_name, layer, variables, gradients, inputs, output, output_gradients,
                 layer_subscript=None):
        """
        Args:
            display_name:
                Made available as a property. For any use as needed.
            layer:
                The actual layer itself.
            variables: list of tensor.
                The state of the variables for this layer at the time of interest, or None if unknown.
            gradients: list of tensor.
                The gradients for the variables at the time of interest, or None if unknown.
            inputs: list of tensor.
                The actual input values to this layer at the time of interest, or None if unknown.
            output: tensor
                The actual post-activation output from this layer at the time of interest, or None if unknown.
            output_gradients: tensor
                Computed gradients w.r.t. the output from this layer at the time of interest, or None if unknown.
            layer_subscript: string
                Used in warnings and error messages to identify the layer via a subscript, eg: 'l' gets used as
                'A_l', 'Z_l', etc.
        """
        self._display_name = display_name
        self._layer = layer
        self._variables = variables
        self._gradients = gradients
        self._inputs = inputs
        self._output = output
        self._output_gradients = output_gradients
        if layer_subscript:
            self.subscript = f"_{layer_subscript}"
        else:
            self.subscript = ""

    @property
    def display_name(self):
        return self._display_name

    @property
    def layer(self):
        return self._layer

    @property
    def variables(self):
        return self._variables

    @property
    def gradients(self):
        return self._gradients

    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._output

    @property
    def output_gradients(self):
        return self._output_gradients

    def get_input(self):
        """
        Makes a mandatory fetch of the layer's single input.
        Fails if there is no single unambiguous input. This avoids the need for sanity checking logic in
        other calculation methods.
        Returns:
            The input tensor, if found.
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        """
        if self.inputs is None or len(self.inputs) == 0:
            raise UnsupportedNetworkError("input to layer unknown")
        if len(self.inputs) > 1:
            raise UnsupportedNetworkError(f"multiple activations from previous layers: {[a.shape for a in self.inputs]}")
        if self.inputs[0] is not None:
            return self.inputs[0]
        else:
            raise UnsupportedNetworkError("input to layer unknown")

    def get_weights(self):
        """
        Makes a mandatory fetch of the layer's main weights.
        Identifies which of the layer's variables is the main weights tensor, or kernel in the case of convolutional
        layers. The default implementation just picks the largest variable.
        Fails if the gradients cannot be found. This avoids the need for sanity checking logic in
        other calculation methods.
        Returns:
            The weights, if found.
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        """
        largest = None
        if self.variables is not None and len(self.variables) > 0:
            largest, _ = _split_by_largest(self.variables)
        if largest is not None:
            return largest
        else:
            raise UnsupportedNetworkError("weights unknown")

    def get_biases_or_none(self):
        """
        Makes an optional fetch of the layer's main weights.
        Identifies which of the layer's variables is the main bias tensor.
        The default implementation just picks the second largest variable, if one is present.
        Returns:
            The biases, or none if this layer is understood but has less than two variables.
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        """
        if self.variables is None:
            raise UnsupportedNetworkError("variables unknown")
        if len(self.variables) >= 2:
            _, rest = _split_by_largest(self.variables)
            return rest[0]
        else:
            return None

    def get_weight_gradients(self):
        """
        Makes a mandatory fetch of the gradients of the layer's main weights.
        Identifies which of the layer's gradients are the gradients for the main weights, or for the kernel in the case
        of convolutional layers. The default implementation just picks the largest gradients.
        Fails if the gradients cannot be found. This avoids the need for sanity checking logic in
        other calculation methods.
        Returns:
            The gradients, if known.
        Raises:
            UnsupportedNetworkError if cannot handle this for the given layer type
        """
        largest = None
        if self.gradients is not None and len(self.gradients) > 0:
            largest, _ = _split_by_largest(self.gradients)
        if largest is not None:
            return largest
        else:
            raise UnsupportedNetworkError("gradients unknown")

    def get_A(self):
        """
        Gets the post-activation output from this layer, if known.
        """
        if self.output is not None:
            return self.output
        else:
            raise UnsupportedNetworkError(f"A{self.subscript} unknown")

    def calculate_Z(self, return_equation=False, return_note=False):
        """
        Calculates or estimates the pre-activation output of the layer, known as Z, if possible.
        Args:
            return_equation: bool.
                Whether to additionally return text describing the equation.
            return_note: bool.
                Whether to additionally return any warning note.
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        Returns:
            - computed value
            - equation (optional output)
            - warning note or none if no warnings (optional output)
        """
        raise UnsupportedNetworkError(f"Calculation of Z{self.subscript} not implemented for this layer type")

    def classify_Z_calculation(self, confidence):
        """
        Provides a classification breakdown of the main weights multiplication for the calculation of Z,
        ignoring any effect from biases.
        Args:
            confidence: passed to matmul_explainer
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        Returns:
            counts, sums
        """
        raise UnsupportedNetworkError(f"Calculation of Z{self.subscript} not implemented for this layer type")

    def calculate_S(self, return_equation=False, return_note=False):
        """
        Calculates or estimates the S matrix, which emulates the behaviour of the activation function in the form
        of an element-wise multiplier against Z.
        In general this is independent of the type of layer, and more dependent on the activation function in use.
        However some layers don't have an activation function (eg: normalisation, dropout).
        The default implementation assumes an activation function is present, and
        accurately computes S if Z is computable, or estimates S from A otherwise.
        Args:
            return_equation: bool.
                Whether to additionally return text describing the equation.
            return_note: bool.
                Whether to additionally return any warning note.
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        Returns:
            - computed value
            - equation (optional output)
            - warning note or none if no warnings (optional output)
        """
        A_l = self.get_A()
        try:
            Z_l = self.calculate_Z()
        except UnsupportedNetworkError:
            Z_l = None

        note = None
        if Z_l is None:
            # estimate S_l from A_l, assuming ReLU
            # - original S_l = {1 if Z_l[ij] >= 0 else 0}
            # - even assuming ReLU, we cannot identify Z_l[ij] < 0 vs Z_l[ij] = 0
            # - so we assume A_l[ij] == 0 => Z_[ij] < 0, thus:
            # - estimated S_l = {1 if A_l[ij] > 0 else 0}
            S_l = tf.cast(A_l > 0, tf.float32)
            equation = f"A{self.subscript} > 0"
            note = f"estimated S{self.subscript} value from A{self.subscript}, assuming ReLU"
        else:
            # calculate accurate S_l from Z_l
            #   A_l = Z_l (.) S_l   -- element-wise
            # so:
            #   S_l = A_l / Z_l or assume 0.0 if A_l is zero (eg: compatible with sigmoid)
            equation = f"A{self.subscript} > 0"
            S_l = np.divide(A_l, Z_l, out=np.zeros_like(A_l), where=(Z_l > 0))

        res = [S_l]
        if return_equation:
            res.append(equation)
        if return_note:
            res.append(note)
        return tuple(res) if len(res) > 1 else res[0]

    def calculate_dJdZ(self, dJdA_l, return_equation=False, return_note=False):
        """
        Calculates or estimates the first step of backprop through this layer, dJdZ_l.
        The default implementation works so long as it can compute S_l.
        Note: where a layer was fed as input into multiple next layers, this method can be used
        against each of their backprops. The final result is usually just the mean.
        Args:
            dJdA_l: the backprop received by this layer
            return_equation: bool.
                Whether to additionally return text describing the equation.
            return_note: bool.
                Whether to additionally return any warning note.
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        Returns:
            - computed value
            - equation (optional output)
            - warning note or none if no warnings (optional output)
        """
        S_l = self.calculate_S()
        dJdZ_l = tf.multiply(dJdA_l, S_l)
        equation = f"dJdA{self.subscript} (.) S{self.subscript}"

        res = [dJdZ_l]
        if return_equation:
            res.append(equation)
        if return_note:
            res.append(None)
        return tuple(res) if len(res) > 1 else res[0]

    def classify_dJdZ_calculation(self, dJdA_l, confidence):
        """
        Provides a classification breakdown of the main weights multiplication for the calculation of Z,
        ignoring any effect from biases.
        The default implementation works so long as it can compute S_l.
        Args:
            dJdA_l: the backprop received by this layer
            confidence: passed to matmul_explainer
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        Returns:
            counts, sums
        """
        S_l = self.calculate_S()
        return mmexpl.multiply_classify(dJdA_l, S_l)

    def calculate_dJdW(self, dJdZ_l, return_equation=False, return_note=False):
        """
        Calculates or estimates the weight gradients, dJdW_l, for this layer.
        Args:
            dJdZ_l: the result of the first step of backgroup through this layer, averaged over all output layers if there are multiple.
            return_equation: bool.
                Whether to additionally return text describing the equation.
            return_note: bool.
                Whether to additionally return any warning note.
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        Returns:
            - computed value
            - equation (optional output)
            - warning note or none if no warnings (optional output)
        """
        raise UnsupportedNetworkError(f"Not implemented for this layer type")

    def classify_dJdW_calculation(self, dJdZ_l, confidence):
        """
        Provides a classification breakdown of the main weights multiplication for the calculation of Z,
        ignoring any effect from biases.
        Args:
            dJdZ_l: the result of the first step of backgroup through this layer, averaged over all output layers if there are multiple.
            confidence: passed to matmul_explainer
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        Returns:
            counts, sums
        """
        raise UnsupportedNetworkError(f"Not implemented for this layer type")

    # TODO to support layers that receive multiple inputs, this method needs to be extended to take an input
    #   number in order to calculate the value w.r.t. that input.
    def calculate_backprop(self, return_note=False):
        """
        Calculates or estimates the backprop gradients, dJdA_0, that would have been passed from
        this layer to the previous layer(s).
        Note that estimating backprop gradients can be unreliable. Generally you will want to directly capture the
        output gradients. But this approach can be used as a last resort.
        Args:
            return_note: bool.
                Whether to additionally return any warning note.
        Raises:
            UnsupportedNetworkError: if cannot handle this for the given layer type
        Returns:
            - computed value
            - warning note or none if no warnings (optional output)
        """
        raise UnsupportedNetworkError(f"Not implemented for this layer type")


class DenseLayerHandler(LayerHandler):
    def __init__(self, display_name, layer, variables, gradients, inputs, output, output_gradients,
                 layer_subscript=None):
        super().__init__(display_name, layer, variables, gradients, inputs, output, output_gradients, layer_subscript)

    def calculate_Z(self, return_equation=False, return_note=False):
        # The following mirrors how TF Dense layer copes with unexpected spatial dims - by treating them
        # as if they're part of the batch. All done via the magic of tensordot.
        # The following is the equivalent of this for a simple (batch, channels) input:
        #   Z_l = tf.matmul(A_0, W_l)
        A_0 = self.get_input()
        W_l = self.get_weights()
        b_l = self.get_biases_or_none()  # may be None
        Z_l = tf.tensordot(A_0, W_l, axes=([-1], [0]))
        equation = f"A_0 . W{self.subscript}"
        if b_l is not None:
            Z_l = Z_l + b_l
            equation = f"A_0 . W{self.subscript} + b{self.subscript}"

        res = [Z_l]
        if return_equation:
            res.append(equation)
        if return_note:
            res.append(None)
        return tuple(res) if len(res) > 1 else res[0]

    def classify_Z_calculation(self, confidence):
        A_0 = self.get_input()
        W_l = self.get_weights()
        return mmexpl.matmul_classify(A_0, W_l, confidence=confidence)

    def calculate_dJdW(self, dJdZ_l, return_equation=False, return_note=False):
        # The following mirrors how TF Dense layer copes with unexpected spatial dims - by treating them
        # as if they're part of the batch. All done via the magic of tensordot.
        # The following is the equivalent of this for simple (batch, in_channels) and (batch, out_channels) values:
        #   A_0t = tf.transpose(A_0)
        #   dJdW_l = tf.matmul(A_0t, dJdZ_l)
        # But it works for:
        #   A_0:    (batch, ..spatial_dims.., in_channels)
        #   dJdZ_l: (batch, ..spatial_dims.., out_channels)
        #   dJdW:   (in_channels, out_channels)
        A_0 = self.get_input()
        axes_A = list(range(len(A_0.shape) - 1))  # sum over all but last dim
        axes_dJdZ = list(range(len(dJdZ_l.shape) - 1))  # sum over all but last dim
        dJdW_l = tf.tensordot(A_0, dJdZ_l, axes=[axes_A, axes_dJdZ])
        if tf.rank(A_0) > 2 or tf.rank(dJdZ_l) > 2:
            equation = f"A_0 tensordot(axes=[{axes_A}, {axes_dJdZ}]) dJ/dZ{self.subscript})"
        else:
            equation = f"A_0^T . dJ/dZ{self.subscript}"

        res = [dJdW_l]
        if return_equation:
            res.append(equation)
        if return_note:
            res.append(None)
        return tuple(res) if len(res) > 1 else res[0]

    def classify_dJdW_calculation(self, dJdZ_l, confidence):
        A_0 = self.get_input()
        axes_A = list(range(len(A_0.shape) - 1))
        axes_dJdZ = list(range(len(dJdZ_l.shape) - 1))
        return mmexpl.tensordot_classify(A_0, dJdZ_l, axes=[axes_A, axes_dJdZ], confidence=confidence)

    def calculate_backprop(self, return_note=False):
        # We need the backprop gradients that are produced by this layer and which feed into the gradient update step of
        # the prev layer. However we don't have that, and instead have the gradients of the weights at this layer.
        # That's one step in the wrong direction, so we have to backtrace the calculations first.

        # steps to estimate backprop (dJ/dA_0) from this layer
        # given:
        #    dJ/dW_l = weight gradients at this layer
        #    dZ_l/dW_l = A_0^T = transposed activations from previous layer
        #    dZ_l/dA_0 = W_1^T = transposed weights from this layer
        # want:
        #    dJ/dA_0 = backprop gradients to previous layer
        # #1:
        #    dJ/dW_l = dZ_l/dW_l . dJ/dZ_l = A_0^T . dJ/dZ_l   <-- (f_0 x f_l) = (f_0 x b) . (b x f_l)
        #    => dJ/dZ_l = (A_0+)^T . dJ/dW_l                   <-- (b x f_l) = (b x f_0) . (f_0 x f_l)
        # #2:
        #    dJ/dA_0 = dJ/dZ_l . dZ_l/dA_0 = dJ/dZ_l . W_l^T   <-- (b x f_0) = (b x f_l) . (f_l x f_0)
        W_l = self.get_weights()
        dJdW_l = self.get_weight_gradients()
        A_0 = self.get_input()
        if tf.rank(A_0) > 2 or tf.rank(dJdW_l) > 2:
            raise UnsupportedNetworkError("backprop estimation not supported with spatial dims")

        A_0_inv = tf.linalg.pinv(A_0)
        dJdZ_l = tf.linalg.matmul(A_0_inv, dJdW_l, transpose_a=True)
        dJdA_0 = tf.linalg.matmul(dJdZ_l, W_l, transpose_b=True)

        res = [dJdA_0]
        if return_note:
            res.append(None)
        return tuple(res) if len(res) > 1 else res[0]


class ConvLayerHandler(LayerHandler):
    def __init__(self, display_name, layer, variables, gradients, inputs, output, output_gradients,
                 layer_subscript=None):
        super().__init__(display_name, layer, variables, gradients, inputs, output, output_gradients, layer_subscript)

    def calculate_Z(self, return_equation=False, return_note=False):
        # The following mirrors how TF Dense layer copes with unexpected spatial dims - by treating them
        # as if they're part of the batch. All done via the magic of tensordot.
        # The following is the equivalent of this for a simple (batch, channels) input:
        #   Z_l = tf.matmul(A_0, W_l)
        A_0 = self.get_input()    # shape: (batch, ..spatial_dims.., in_channels)
        W_l = self.get_weights()  # shape: (k_h, k_w, in_channels, out_channels)
        b_l = self.get_biases_or_none()   # shape: (out_channels,), may be None

        conv_params, note = self.get_conv_params()
        Z_l = tf.nn.convolution(input=A_0, filters=W_l, **conv_params)
        equation = f"A_0 conv W{self.subscript}"
        if b_l is not None:
            Z_l = Z_l + b_l
            equation = f"A_0 conv W{self.subscript} + b{self.subscript}"

        res = [Z_l]
        if return_equation:
            res.append(equation)
        if return_note:
            res.append(None)
        return tuple(res) if len(res) > 1 else res[0]

    def classify_Z_calculation(self, confidence):
        A_0 = self.get_input()
        W_l = self.get_weights()
        conv_params, _ = self.get_conv_params()
        return mmexpl.conv_classify(A_0, W_l, confidence=confidence, **conv_params)

    def calculate_dJdW(self, dJdZ_l, return_equation=False, return_note=False):
        # while tf.nn.conv2d_backprop_filter() and tf.nn.conv1d_backprop_filter() are available,
        # they are actually equivalent to a transposed convolution, and there is a multi-dimensional
        # equivalent of that, just like for the forward pass. So we use that to make life easy.
        A_0 = self.get_input()
        W = self.get_weights()
        conv_params, note = self.get_conv_params()
        try:
            dJdW_l = conv_tools.conv_backprop_filter(x=A_0, d_out=dJdZ_l, kernel_shape=W.shape, **conv_params)
            equation = f"A_0 conv^T dJ/dZ{self.subscript}"  # simplification
        except ValueError as e:
            raise UnsupportedNetworkError(f"unsupported convolution - {e}")

        res = [dJdW_l]
        if return_equation:
            res.append(equation)
        if return_note:
            res.append(None)
        return tuple(res) if len(res) > 1 else res[0]

    def classify_dJdW_calculation(self, dJdZ_l, confidence):
        A_0 = self.get_input()
        W = self.get_weights()
        conv_params, note = self.get_conv_params()
        try:
            return mmexpl.conv_backprop_filter_classify(x=A_0, d_out=dJdZ_l, kernel_shape=W.shape, **conv_params)
        except ValueError as e:
            raise UnsupportedNetworkError(f"unsupported convolution - {e}")

    def get_conv_params(self):
        # determines the other arguments to tf.nn.convolution() if possible
        # Returns: **kwargs, note
        kwargs = {}
        dodgy = False

        if hasattr(self.layer, 'strides'):
            kwargs['strides'] = self.layer.strides
        else:
            dodgy = True

        if hasattr(self.layer, 'padding'):
            kwargs['padding'] = None if self.layer.padding is None else self.layer.padding.upper()
        else:
            dodgy = True

        if hasattr(self.layer, 'dilation_rate'):
            if self.layer.dilation_rate is not None:
                if (np.array(self.layer.dilation_rate) > 1).any():
                    raise UnsupportedNetworkError(f"unable to handle convolutional dilations "
                                                  f"{self.layer.dilation_rate}")

        note = "padding and strides may be inaccurate" if dodgy else None
        return kwargs, note


def reload_safe_isinstance(obj, cls):
    """
    Because I do a lot of `reload(module)` during development, `isinstance` tests become unreliable.
    """
    if isinstance(obj, cls):
        return True
    obj_cls_fqn = f"{type(obj).__module__}.{type(obj).__name__}"
    cls_fqn = f"{cls.__module__}.{cls.__name__}"
    return obj_cls_fqn == cls_fqn
