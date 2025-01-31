# This module contains a few pieces of functionality that I couldn't find within Tensorflow.
import tensorflow as tf
import numpy as np


def dilate(tensor, strides=None):
    """
    Inserts zeros between pixels in the spatial dimensions to simulate fractional strides.
    This has the opposite effect of dilations within convolution, which dilate the kernel.

    Implementation:
        Built so that it can stride against all dims, because it's easier that way.
        But just has some sanity check and conversions on the input, because we never want
        to stride along the batch or channel dims.
    Args:
        tensor: A multi-dimensional tensor of N+2 dims, with shape (batch, ..spatial_dims.., channels)
        strides: sequence of N ints >= 1. Specifies the output stride along the spatial dims. Defaults to [1]*N.
    Returns:
        A new tensor with zeros interlaced along the spatial dimensions.
    """
    if strides is None:
        strides = [1] * (len(tensor.shape)-2)
    elif isinstance(strides, (int, float, complex, str, bool)):
        strides = [strides] * (len(tensor.shape)-2)
    if len(strides) != len(tensor.shape)-2:
        raise ValueError(f"Number of strides must match tensor spatial dims: "
                         f"{len(strides)} doesn't match {tensor.shape}")

    # prepare data
    # - convert strides to cover full tensor dims
    # - convert sizes and shapes to numpy for easier handling
    shape = np.array(tensor.shape)
    strides = np.array((1,) + tuple(strides) + (1,))

    # Create empty tensor with expanded shape
    new_shape = shape + (shape-1) * (strides-1)
    new_tensor = tf.zeros(new_shape, dtype=tensor.dtype)

    # Assign original values to non-zero positions
    indexing_by_dim = [tf.range(0, axis_len, axis_stride) for axis_len, axis_stride in zip(new_shape, strides)]
    indices = tf.stack(tf.meshgrid(*indexing_by_dim, indexing='ij'), axis=-1)
    indices = tf.reshape(indices, shape=(-1, len(indexing_by_dim)))  # Flatten the indices
    new_tensor = tf.tensor_scatter_nd_update(
        new_tensor,
        indices=indices,
        updates=tf.reshape(tensor, [-1])
    )

    return new_tensor
