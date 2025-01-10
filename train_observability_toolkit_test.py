# TODO needs to be converted to a proper set of unit tests once this toolkit is moved
# to something with a build tool
import tensorflow as tf
from train_observability_toolkit import *
from train_observability_toolkit import _normalize_collection_sets_for_layers, _normalize_collection_sets_for_variables


def run_test_suite():
    variable_indices_by_layer_test()
    trainable_variable_indices_by_layer_test()
    _normalize_collection_sets_for_layers_test()
    _normalize_collection_sets_for_variables_test()
    print("All train_observability_toolkit tests passed.")


def variable_indices_by_layer_test():
    model = _create_test_model()

    res = variable_indices_by_layer(model)
    expected = [[0, 1], [2, 3], [4], [5, 6], [7, 8], [9], [10, 11], [12, 13, 14, 15], [16, 17], [18, 19]]
    assert res == expected, f"include_trainable_only=Default: expected {expected}, but got {res}"

    res = variable_indices_by_layer(model, include_trainable_only=False)
    expected = [[0, 1], [2, 3], [4], [5, 6], [7, 8], [9], [10, 11], [12, 13, 14, 15], [16, 17], [18, 19]]
    assert res == expected, f"include_trainable_only=False: expected {expected}, but got {res}"

    res = variable_indices_by_layer(model, include_trainable_only=True)
    expected = [[0, 1], [2, 3], [], [5, 6], [7, 8], [], [10, 11], [12, 13], [16, 17], [18, 19]]
    assert res == expected, f"include_trainable_only=True: expected {expected}, but got {res}"


def trainable_variable_indices_by_layer_test():
    model = _create_test_model()

    res = trainable_variable_indices_by_layer(model)
    expected = [[0, 1], [2, 3], [], [4, 5], [6, 7], [], [8, 9], [10, 11], [12, 13], [14, 15]]
    assert res == expected, f"Expected {expected}, but got {res}"


def _normalize_collection_sets_for_layers_test():
    model = _create_test_model()

    # singleton closed collection sets
    res = _normalize_collection_sets_for_layers(model, [{'layer_indices': [0, 3]}])
    expected = [{'layer_indices': [0, 3]}]
    assert res == expected, f"Accepts layer_indices as is: expected {expected}, but got {res}"

    res = _normalize_collection_sets_for_layers(model, [{'layers': [model.layers[0], model.layers[3]]}])
    expected = [{'layers': [model.layers[0], model.layers[3]], 'layer_indices': [0, 3]}]
    assert res == expected, f"Translates layers: expected {expected}, but got {res}"

    res = _normalize_collection_sets_for_layers(model, [{'layer_names': [model.layers[0].name, model.layers[3].name]}])
    expected = [{'layer_names': [model.layers[0].name, model.layers[3].name], 'layer_indices': [0, 3]}]
    assert res == expected, f"Translates layer names: expected {expected}, but got {res}"

    # multiple closed collection sets
    res = _normalize_collection_sets_for_layers(model, [
        {'layer_indices': [0, 1]},
        {'layers': [model.layers[1], model.layers[4]]},
        {'layer_names': [model.layers[2].name, model.layers[5].name]}])
    expected = [
        {'layer_indices': [0, 1]},
        {'layers': [model.layers[1], model.layers[4]], 'layer_indices': [2, 3, 7, 8]},
        {'layer_names': [model.layers[2].name, model.layers[5].name], 'layer_indices': [4, 9]}]
    assert res == expected, f"Translates across multiple collection sets: expected {expected}, but got {res}"

    # open-ended collection sets
    res = _normalize_collection_sets_for_variables(model, [{}])
    expected = [{'layer_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}]
    assert res == expected, f"Expands single all-layers collection set: expected {expected}, but got {res}"

    res = _normalize_collection_sets_for_variables(model, [{'layer_indices': [2, 3, 4]}, {}])
    expected = [{'layer_indices': [2, 3, 4]},
                {'variable_indices': [0, 1, 5, 6, 7, 8, 9]}]
    assert res == expected, f"Expands open-ended collection set with remaining layers: expected {expected}, but got {res}"

    # error conditions
    try:
        _normalize_collection_sets_for_variables(model, [
            {'variable_indices': [2, 3, 4]},
            {'trainable_variable_indices': [4, 5, 6, 2]}])
        assert False, "Detects duplicate indices given variable_indices and trainable_variable_indices: " \
                      "Expected duplicate error"
    except ValueError:
        pass

    try:
        _normalize_collection_sets_for_variables(model, [
            {'variable_indices': [2, 3, 4]},
            {'layer_indices': [1]}])
        assert False, "Detects duplicate indices given variable_indices and layer_indices: " \
                      "Expected duplicate error"
    except ValueError:
        pass


def _normalize_collection_sets_for_variables_test():
    model = _create_test_model()

    # singleton closed collection sets
    res = _normalize_collection_sets_for_variables(model, [{'variable_indices': [2, 3, 4]}])
    expected = [{'variable_indices': [2, 3, 4]}]
    assert res == expected, f"Accepts as is when variable_indices given directly: expected {expected}, but got {res}"

    res = _normalize_collection_sets_for_variables(model, [{'trainable_variable_indices': [2, 3, 4, 5]}])
    expected = [{'trainable_variable_indices': [2, 3, 4, 5], 'variable_indices': [2, 3, 5, 6]}]
    assert res == expected, f"Translates trainable_variable_indices: expected {expected}, but got {res}"

    res = _normalize_collection_sets_for_variables(model, [{'layers': [model.layers[0], model.layers[3]]}])
    expected = [{'layers': [model.layers[0], model.layers[3]], 'variable_indices': [0, 1, 5, 6]}]
    assert res == expected, f"Translates layers: expected {expected}, but got {res}"

    res = _normalize_collection_sets_for_variables(model, [{'layer_indices': [0, 3]}])
    expected = [{'layer_indices': [0, 3], 'variable_indices': [0, 1, 5, 6]}]
    assert res == expected, f"Translates layer indices: expected {expected}, but got {res}"

    res = _normalize_collection_sets_for_variables(model, [{'layer_names': [model.layers[0].name, model.layers[3].name]}])
    expected = [{'layer_names': [model.layers[0].name, model.layers[3].name], 'variable_indices': [0, 1, 5, 6]}]
    assert res == expected, f"Translates layer names: expected {expected}, but got {res}"

    # multiple closed collection sets
    res = _normalize_collection_sets_for_variables(model, [
        {'variable_indices': [2, 3, 4]},
        {'layer_indices': [0]},
        {'trainable_variable_indices': [10, 12, 14]}])
    expected = [{'variable_indices': [2, 3, 4]},
                {'layer_indices': [0], 'variable_indices': [0, 1]},
                {'trainable_variable_indices': [10, 12, 14], 'variable_indices': [12, 16, 18]}]
    assert res == expected, f"Translates across multiple collection sets: expected {expected}, but got {res}"

    # open-ended collection sets
    res = _normalize_collection_sets_for_variables(model, [{}])
    expected = [{'layer_indices': [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 19]}]
    assert res == expected, f"Expands single all-variables collection set with all trainable variables: " \
                            f"expected {expected}, but got {res}"

    res = _normalize_collection_sets_for_variables(model, [{'variable_indices': [2, 3, 4]}, {}])
    expected = [
        {'variable_indices': [2, 3, 4]},
        {'variable_indices': [0, 1, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 18, 19]}]
    assert res == expected, f"Expands open-ended collection set with remaining trainable variables: " \
                            f"expected {expected}, but got {res}"

    res = _normalize_collection_sets_for_variables(model, [
        {'variable_indices': [2, 3, 4]},
        {'include_non_trainable': True}])
    expected = [
        {'variable_indices': [2, 3, 4]},
        {'variable_indices': [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}]
    assert res == expected, f"Expands open-ended collection set with all remaining variables: " \
                            f"expected {expected}, but got {res}"

    # error conditions
    try:
        _normalize_collection_sets_for_variables(model, [{'variable_indices': [2, 3, 4]},
                                                         {'trainable_variable_indices': [4, 5, 6, 2]}])
        assert False, "Detects duplicate indices given variable_indices and trainable_variable_indices: " \
                      "Expected duplicate error"
    except ValueError:
        pass

    try:
        _normalize_collection_sets_for_variables(model, [{'variable_indices': [2, 3, 4]},
                                                         {'layer_indices': [1]}])
        assert False, "Detects duplicate indices given variable_indices and layer_indices: " \
                      "Expected duplicate error"
    except ValueError:
        pass


def _create_test_model():
    # 20 variables total, 16 trainable
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),              # omitted from layers
        tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
        tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
        tf.keras.layers.Dropout(rate=0.2),              # 1 non-trainable var
        tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
        tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
        tf.keras.layers.Dropout(rate=0.2),              # 1 non-trainable var
        tf.keras.layers.Dense(100, activation='relu'),  # 2 trainable vars
        tf.keras.layers.BatchNormalization(),           # 2 trainable vars + 2 non-trainable vars
        tf.keras.layers.Dense(5, activation='relu'),    # 2 trainable vars
        tf.keras.layers.Dense(1, activation='sigmoid')  # 2 trainable vars
    ])
    return model
