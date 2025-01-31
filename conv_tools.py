# This module contains a few pieces of functionality that I couldn't find within Tensorflow.
import tensorflow as tf
import numpy as np


def conv_backprop_filter(x, d_out, kernel_shape, strides, padding, data_format='NHWC'):
    """
    Computes the gradients of N-D convolution with respect to the kernel.

    Support matrix:
    - supported: strides of up to 2
    - unsupported: strides > 2
    - unsupported: dilations
    - only 'NHWC' data format is supported

    This implementation was initially inspired by the following work by Yixing Lao, 2017:
    - https://gist.github.com/yxlao/ef50416011b9587835ac752aa3ce3530
    - https://stackoverflow.com/questions/39373230/what-does-tensorflows-conv2d-transpose-operation-do/44350789#44350789
    However that initial code was written for TensorFlow v2 and no longer worked. The final solution
    was devised from a lengthy round of trial-and-error. It may not be theoretically sound, but it seems to
    work well enough on the situations that it can handle.

    Args:
        x: original input tensor that had one of the `tf.nn.conv` operations applied against it.
        kernel_shape: tensor, list, or tuple identifying the shape of the filter used during convolution.
        d_out: tensor containing the `dJ/dOut` backprop gradients
        strides: scalar, list or tuple of strides in each spatial dimension. Can omit the batch and channel dims,
          which must have stride 1 if included.
        padding: one of 'VALID' or 'SAME'
        data_format: must be 'NHWC'
    Returns:
        tensor containing the `dJ/dW` gradients
    """
    if strides is None:
        strides = [1] * (len(x.shape)-2)
    elif isinstance(strides, (int, float, complex, str, bool)):
        strides = [strides] * (len(x.shape)-2)
    if padding is None:
        padding = 'VALID'

    if x.shape[0] != d_out.shape[0]:
        raise ValueError(f"Batch dims don't match between x and d_out, got: x={x.shape}, d_out={d_out.shape}")
    if len(x.shape) != len(d_out.shape):
        raise ValueError(f"Number of dims of x and d_out must match, got: x={x.shape}, d_out={d_out.shape}")
    if x.shape[-1] != kernel_shape[-2]:
        raise ValueError(f"Input channel dim doesn't match kernel input channel dim, got: x={x.shape}, kernel={kernel_shape}")
    if d_out.shape[-1] != kernel_shape[-1]:
        raise ValueError(f"Output channel dim doesn't match kernel output channel dim, got: d_out={d_out.shape}, kernel={kernel_shape}")
    if len(strides) != len(x.shape)-2 and len(strides) != len(x.shape):
        raise ValueError(f"Number of strides must match tensor dims or spatial dims: "
                         f"{len(strides)} doesn't match {x.shape}")
    if len(strides) == len(x.shape) and (strides[0] != 1 or strides[-1] != 1):
        raise ValueError(f"Strides for batch and channel dims must be 1, got: {strides}")
    if padding is not None and padding not in ('SAME', 'VALID'):
        raise ValueError(f"Unsupported padding: {padding}")
    if data_format != 'NHWC':
        raise ValueError("Only NHWC data format supported")

    # get meta-info on the dimensions
    x_spatial_shape = np.array(x.shape[1:-1])          # (batch, ..spatial_dims.., channel)
    k_spatial_shape = np.array(kernel_shape[:-2])      # (..spatial_filters.., in_channel, out_channel)
    spatial_strides = strides if len(strides) == len(x_spatial_shape) else strides[1:-1]

    # apply padding to x
    # - tip: more padding makes dJdW kernel larger
    # - the following applies independently for each spatial dim:
    #   - in general, if padding=SAME, then we need to distribute (k-1)/2 pad between before/after on each dim.
    #   - in general, if padding=VALID, then we don't need to add padding;
    #     however, when strides > 1, and k is even, then need to add a half-pad to one side
    #   - when k and x are both even length, and strides > 1, then we need to flip the bias direction for padding.
    x_paddings = []
    for dim in range(len(x_spatial_shape)):
        x_len = x_spatial_shape[dim]
        k = k_spatial_shape[dim]
        s = spatial_strides[dim]
        if padding == 'SAME':
            total = k - 1
            before = total // 2  # usually biased towards 'after'
            after = total - before
            if k % 2 == 0 and x_len % 2 == 0 and s > 1:
                # flip pad bias
                after = total // 2
                before = total - after
        else:
            before = 0
            after = 0
            if s > 1 and k % 2 == 0:
                # half-pad, with bias depending on odd/even of x length
                before, after = (0, 1) if x_len % 2 == 0 else (1, 0)
        x_paddings.append([before, after])
    x_paddings = [[0, 0]] + x_paddings + [[0, 0]]
    x_prepared = tf.pad(x, x_paddings)
    x_spatial_shape = np.array(x_prepared.shape[1:-1])  # revise for later processing

    # handle strides
    # - strides during the original convolution translate as "fractional strides" during backprop.
    # - to achieve that effect, we dilate d_out.
    d_out_prepared = dilate(d_out, spatial_strides)

    # apply padding to d_out
    # - tip: more padding makes dJdW kernel smaller
    # - the following applies independently for each spatial dim:
    #   - compute padding as: padding = prepared_x.shape - prepared_d_out.shape - kernel.shape + 1
    #   - distribute between before/after with bias direction depending on whether x is_len is even or odd
    d_out_spatial_shape = np.array(d_out_prepared.shape[1:-1])  # (batch, ..spatial_dims.., channel)
    d_out_paddings = []
    for dim in range(len(d_out_spatial_shape)):
        d_len = d_out_spatial_shape[dim]
        x_len = x_spatial_shape[dim]
        k = k_spatial_shape[dim]
        s = spatial_strides[dim]
        before = after = 0
        if s > 1:
            total = x_len - d_len - k + 1
            before = total // 2  # usually biased towards 'after'
            after = total - before
            if padding == 'SAME':
                # flip pad bias
                after = total // 2
                before = total - after
        d_out_paddings.append([before, after])
    d_out_paddings = [[0, 0]] + d_out_paddings + [[0, 0]]
    if (np.array(d_out_paddings) < 0).any():
        raise ValueError(f"Invalid dimensions, got negative padding: {d_out_paddings}")
    d_out_prepared = tf.pad(d_out_prepared, d_out_paddings)

    # rearrange dims
    # - we want to sum over the batch and spatial dimensions in order to compute dJdW
    # - so we rearrange the dimensions to make that work
    # Example pattern of rearrangements:
    #             internal conv layout        API layout
    #             --------------------     -----------------
    #    input:   (  5, 120, 145, 32)  <-  (32, 120, 145,  5)  <- A_0
    #    filters: (118, 143,  32, 16)  <-  (32, 118, 143, 16)  <- dJdZ
    #    result:  (  5,   3,   3, 16)   -> ( 3,   3,   5, 16)   -> dJdW
    x_prepared = tf_NHWC_to_CHWN(x_prepared)
    d_out_prepared = tf_NHWC_to_HWIO(d_out_prepared)

    # compute convolution
    # - natively supports any-dimensional convolution
    # - no strides and no padding, as that's already been taken care of
    # - initially produces d_w but with dimensions rearranged
    d_w = tf.nn.convolution(
        input=x_prepared,
        filters=d_out_prepared,
        strides=None,
        padding='VALID')
    return tf_NHWC_to_HWIO(d_w, reverse=True)


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


def tf_NHWC_to_HWIO(out, reverse=False):
    """
    Converts [batch, ..spatial_dims.., in_channels]
    to       [..spatial_dims.., in_channels, out_channels],
    by treating 'batch' as in_channels.
    Args:
      out: tensor
      reverse: whether to reverse the operation. Ignored, because the implementation
        is symmetric. But the argument makes the call-side intention clearer.
    """
    # eg: batch+2D+channels: tf.transpose(out, perm=[1, 2, 0, 3])
    # eg: batch+3D+channels: tf.transpose(out, perm=[1, 2, 3, 0, 4])
    shape = out.shape
    axes = list(range(1, len(shape)-1)) + [0, len(shape)-1]
    return tf.transpose(out, perm=axes)

def tf_NHWC_to_CHWN(out):
    """
    Flips the batch and channels dimensions.
    Converts [batch, ..spatial_dims.., channels]
    to       [in_channels, ..spatial_dims.., batch].
    """
    # eg: batch+2D+channels: tf.transpose(out, perm=[3, 1, 2, 0])
    # eg: batch+3D+channels: tf.transpose(out, perm=[4, 1, 2, 3, 0])
    shape = out.shape
    axes = [len(shape)-1] + list(range(1, len(shape)-1)) + [0,]
    return tf.transpose(out, perm=axes)


def tf_crop(x, croppings):
    """
    Mirrors the tf.pad() api for cropping.
    Crops a tensor by removing elements from the start and end of each dimension.
    Args:
        x: N-dimensional input tensor
        croppings: List of N tuples [(crop_before_1, crop_after_1), (crop_before_2, crop_after_2), ...]
                    specifying how much to crop from each dimension.
    Returns:
        Cropped tensor.
    """
    tensor_shape = tf.shape(x)
    if len(croppings) != len(tensor_shape):
        raise ValueError(f"Croppings length must match number of dimensions: {len(croppings)} != {len(tensor_shape)}")

    # Compute the start and new size for each dimension
    begin = [c[0] for c in croppings]  # Start indices for cropping
    size = [tensor_shape[i] - (c[0] + c[1]) for i, c in enumerate(croppings)]  # Compute new sizes

    return tf.slice(x, begin, size)


def tf_crop_spatial(x, croppings):
    """
    Like tf_crop() except that it operates only against spatial dimensions.
    Crops a tensor by removing elements from the start and end of each spatial dimension.
    Args:
        x: (N+2)-dimensional input tensor
        croppings: List of N tuples [(crop_before_1, crop_after_1), (crop_before_2, crop_after_2), ...]
                    specifying how much to crop from each dimension.
    Returns:
        Cropped N+2 tensor.
    """

    if croppings is None:
        return x  # no changes
    if len(croppings) != len(x.shape)-2:
        raise ValueError(f"Number of croppings must match tensor spatial dims: "
                         f"{len(croppings)} doesn't match {x.shape}")

    croppings = [[0, 0]] + list(croppings) + [[0, 0]]
    return tf_crop(x, croppings)

