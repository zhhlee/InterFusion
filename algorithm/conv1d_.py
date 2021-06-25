import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.ops import (assert_rank, assert_scalar_equal, flatten_to_ndims,
                           unflatten_from_ndims)
from tfsnippet.utils import (validate_positive_int_arg, ParamSpec,
                             is_tensor_object, assert_deps,
                             get_shape,
                             add_name_and_scope_arg_doc, model_variable,
                             maybe_check_numerics, maybe_add_histogram, InputSpec,
                             get_static_shape, validate_enum_arg, validate_int_tuple_arg)

__all__ = [
    'conv1d', 'deconv1d', 'validate_conv1d_input', 'get_deconv_output_length', 'batch_norm_1d'
]


@add_arg_scope
def batch_norm_1d(input, channels_last=True, training=False, name=None,
                  scope=None):
    """
    Apply batch normalization on 1D convolutional layer.

    Args:
        input (tf.Tensor): The input tensor.
        channels_last (bool): Whether or not the channel dimension is at last?
        training (bool or tf.Tensor): Whether or not the model is under
            training stage?

    Returns:
        tf.Tensor: The normalized tensor.
    """
    with tf.variable_scope(scope, default_name=name or 'batch_norm_1d'):
        input, s1, s2 = flatten_to_ndims(input, ndims=3)
        output = tf.layers.batch_normalization(
            input,
            axis=-1 if channels_last else -2,
            training=training,
            name='norm'
        )
        output = unflatten_from_ndims(output, s1, s2)
        return output


def validate_conv1d_input(input, channels_last, arg_name='input'):
    """
    Validate the input for 1-d convolution.
    Args:
        input: The input tensor, must be at least 3-d.
        channels_last (bool): Whether or not the last dimension is the
            channels dimension? (i.e., the data format is (batch, length, channels))
        arg_name (str): Name of the input argument.
    Returns:
        (tf.Tensor, int, str): The validated input tensor, the number of input
            channels, and the data format.
    """
    if channels_last:
        channel_axis = -1
        input_spec = InputSpec(shape=('...', '?', '?', '*'))
        data_format = "NWC"
    else:
        channel_axis = -2
        input_spec = InputSpec(shape=('...', '?', '*', '?'))
        data_format = "NCW"
    input = input_spec.validate(arg_name, input)
    input_shape = get_static_shape(input)
    in_channels = input_shape[channel_axis]

    return input, in_channels, data_format


def get_deconv_output_length(input_length, kernel_size, strides, padding):
    """
    Get the output length of deconvolution at a specific dimension.
    Args:
        input_length: Input tensor length.
        kernel_size: The size of the kernel.
        strides: The stride of convolution.
        padding: One of {"same", "valid"}, case in-sensitive
    Returns:
        int: The output length of deconvolution.
    """
    padding = validate_enum_arg(
        'padding', str(padding).upper(), ['SAME', 'VALID'])
    output_length = input_length * strides
    if padding == 'VALID':
        output_length += max(kernel_size - strides, 0)
    return output_length


@add_arg_scope
@add_name_and_scope_arg_doc
def conv1d(input,
           out_channels,
           kernel_size,
           strides=1,
           dilations=1,
           padding='same',
           channels_last=True,
           activation_fn=None,
           normalizer_fn=None,
           gated=False,
           gate_sigmoid_bias=2.,
           kernel=None,
           kernel_mask=None,
           kernel_initializer=None,
           kernel_regularizer=None,
           kernel_constraint=None,
           use_bias=None,
           bias=None,
           bias_initializer=tf.zeros_initializer(),
           bias_regularizer=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           scope=None):
    """
    1D convolutional layer.
    Args:
        input (Tensor): The input tensor, at least 3-d.
        out_channels (int): The channel numbers of the output.
        kernel_size (int or tuple(int,)): Kernel size over spatial dimensions.
        strides (int): Strides over spatial dimensions.
        dilations (int): The dilation factor over spatial dimensions.
        padding: One of {"valid", "same"}, case in-sensitive.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NWC")
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        gated (bool): Whether or not to use gate on output?
            `output = activation_fn(output) * sigmoid(gate)`.
        gate_sigmoid_bias (Tensor): The bias added to `gate` before applying
            the `sigmoid` activation.
        kernel (Tensor): Instead of creating a new variable, use this tensor.
        kernel_mask (Tensor): If specified, multiply this mask onto `kernel`,
            i.e., the actual kernel to use will be `kernel * kernel_mask`.
        kernel_initializer: The initializer for `kernel`.
            Would be ``default_kernel_initializer(...)`` if not specified.
        kernel_regularizer: The regularizer for `kernel`.
        kernel_constraint: The constraint for `kernel`.
        use_bias (bool or None): Whether or not to use `bias`?
            If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        bias (Tensor): Instead of creating a new variable, use this tensor.
        bias_initializer: The initializer for `bias`.
        bias_regularizer: The regularizer for `bias`.
        bias_constraint: The constraint for `bias`.
        trainable (bool): Whether or not the parameters are trainable?
    Returns:
        tf.Tensor: The output tensor.
    """
    if not channels_last:
        raise ValueError('Currently only channels_last=True is supported.')
    input, in_channels, data_format = \
        validate_conv1d_input(input, channels_last)
    out_channels = validate_positive_int_arg('out_channels', out_channels)
    dtype = input.dtype.base_dtype
    if gated:
        out_channels *= 2

    # check functional arguments
    padding = validate_enum_arg(
        'padding', str(padding).upper(), ['VALID', 'SAME'])
    dilations = validate_positive_int_arg('dilations', dilations)
    strides = validate_positive_int_arg('strides', strides)

    if dilations > 1 and not channels_last:
        raise ValueError('`channels_last` == False is incompatible with '
                         '`dilations` > 1.')

    if strides > 1 and dilations > 1:
        raise ValueError('`strides` > 1 is incompatible with `dilations` > 1.')

    if use_bias is None:
        use_bias = normalizer_fn is None

    # get the specification of outputs and parameters
    kernel_size = validate_int_tuple_arg('kernel_size', kernel_size)
    kernel_shape = kernel_size + (in_channels, out_channels)
    bias_shape = (out_channels,)

    # validate the parameters
    if kernel is not None:
        kernel_spec = ParamSpec(shape=kernel_shape, dtype=dtype)
        kernel = kernel_spec.validate('kernel', kernel)
    if kernel_mask is not None:
        kernel_mask_spec = InputSpec(dtype=dtype)
        kernel_mask = kernel_mask_spec.validate('kernel_mask', kernel_mask)
    if kernel_initializer is None:
        kernel_initializer = tf.glorot_normal_initializer()
    if bias is not None:
        bias_spec = ParamSpec(shape=bias_shape, dtype=dtype)
        bias = bias_spec.validate('bias', bias)

    # the main part of the conv1d layer
    with tf.variable_scope(scope, default_name=name or 'conv1d'):
        c_axis = -1 if channels_last else -2

        # create the variables
        if kernel is None:
            kernel = model_variable(
                'kernel',
                shape=kernel_shape,
                dtype=dtype,
                initializer=kernel_initializer,
                regularizer=kernel_regularizer,
                constraint=kernel_constraint,
                trainable=trainable
            )

        if kernel_mask is not None:
            kernel = kernel * kernel_mask

        maybe_add_histogram(kernel, 'kernel')
        kernel = maybe_check_numerics(kernel, 'kernel')

        if use_bias and bias is None:
            bias = model_variable(
                'bias',
                shape=bias_shape,
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                constraint=bias_constraint,
                trainable=trainable
            )
            maybe_add_histogram(bias, 'bias')
            bias = maybe_check_numerics(bias, 'bias')

        # flatten to 3d
        output, s1, s2 = flatten_to_ndims(input, 3)

        # do convolution
        if dilations > 1:
            output = tf.nn.convolution(
                input=output,
                filter=kernel,
                dilation_rate=(dilations,),
                padding=padding,
                data_format=data_format
            )
        else:
            output = tf.nn.conv1d(
                value=output,
                filters=kernel,
                stride=strides,
                padding=padding,
                data_format=data_format
            )

        # add bias
        if use_bias:
            output = tf.add(output, bias)

        # apply the normalization function if specified
        if normalizer_fn is not None:
            output = normalizer_fn(output)

        # split into halves if gated
        if gated:
            output, gate = tf.split(output, 2, axis=c_axis)

        # apply the activation function if specified
        if activation_fn is not None:
            output = activation_fn(output)

        # apply the gate if required
        if gated:
            if gate_sigmoid_bias is None:
                gate_sigmoid_bias = model_variable(
                    'gate_sigmoid_bias',
                    shape=bias_shape,
                    initializer=bias_initializer,
                    regularizer=bias_regularizer,
                    constraint=bias_constraint,
                    trainable=trainable
                )
                maybe_add_histogram(gate_sigmoid_bias, 'gate_sigmoid_bias')
                gate_sigmoid_bias = maybe_check_numerics(gate_sigmoid_bias, 'gate_sigmoid_bias')
            output = output * tf.sigmoid(gate + gate_sigmoid_bias, name='gate')

        # unflatten back to original shape
        output = unflatten_from_ndims(output, s1, s2)

        maybe_add_histogram(output, 'output')
        output = maybe_check_numerics(output, 'output')
    return output


@add_arg_scope
@add_name_and_scope_arg_doc
def deconv1d(input,
             out_channels,
             kernel_size,
             strides=1,
             padding='same',
             channels_last=True,
             output_shape=None,
             activation_fn=None,
             normalizer_fn=None,
             gated=False,
             gate_sigmoid_bias=2.,
             kernel=None,
             kernel_initializer=None,
             kernel_regularizer=None,
             kernel_constraint=None,
             use_bias=None,
             bias=None,
             bias_initializer=tf.zeros_initializer(),
             bias_regularizer=None,
             bias_constraint=None,
             trainable=True,
             name=None,
             scope=None):
    """
    1D deconvolutional layer.
    Args:
        input (Tensor): The input tensor, at least 3-d.
        out_channels (int): The channel numbers of the deconvolution output.
        kernel_size (int or tuple(int,)): Kernel size over spatial dimensions.
        strides (int): Strides over spatial dimensions.
        padding: One of {"valid", "same"}, case in-sensitive.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NWC")
        output_shape: If specified, use this as the shape of the
            deconvolution output; otherwise compute the size of each dimension
            by::
                output_size = input_size * strides
                if padding == 'valid':
                    output_size += max(kernel_size - strides, 0)
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        gated (bool): Whether or not to use gate on output?
            `output = activation_fn(output) * sigmoid(gate)`.
        gate_sigmoid_bias (Tensor): The bias added to `gate` before applying
            the `sigmoid` activation.
        kernel (Tensor): Instead of creating a new variable, use this tensor.
        kernel_initializer: The initializer for `kernel`.
            Would be ``default_kernel_initializer(...)`` if not specified.
        kernel_regularizer: The regularizer for `kernel`.
        kernel_constraint: The constraint for `kernel`.
        use_bias (bool or None): Whether or not to use `bias`?
            If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        bias (Tensor): Instead of creating a new variable, use this tensor.
        bias_initializer: The initializer for `bias`.
        bias_regularizer: The regularizer for `bias`.
        bias_constraint: The constraint for `bias`.
        trainable (bool): Whether or not the parameters are trainable?
    Returns:
        tf.Tensor: The output tensor.
    """
    if not channels_last:
        raise ValueError('Currently only channels_last=True is supported.')
    input, in_channels, data_format = \
        validate_conv1d_input(input, channels_last)
    out_channels = validate_positive_int_arg('out_channels', out_channels)
    dtype = input.dtype.base_dtype
    if gated:
        out_channels *= 2

    # check functional arguments
    padding = validate_enum_arg(
        'padding', str(padding).upper(), ['VALID', 'SAME'])
    strides = validate_positive_int_arg('strides', strides)

    if use_bias is None:
        use_bias = normalizer_fn is None

    # get the specification of outputs and parameters
    kernel_size = validate_int_tuple_arg('kernel_size', kernel_size)
    kernel_shape = kernel_size + (out_channels, in_channels)
    bias_shape = (out_channels,)

    given_w = None
    given_output_shape = output_shape

    if is_tensor_object(given_output_shape):
        given_output_shape = tf.convert_to_tensor(given_output_shape)
    elif given_output_shape is not None:
        given_w = given_output_shape

    # validate the parameters
    if kernel is not None:
        kernel_spec = ParamSpec(shape=kernel_shape, dtype=dtype)
        kernel = kernel_spec.validate('kernel', kernel)
    if kernel_initializer is None:
        kernel_initializer = tf.glorot_normal_initializer()
    if bias is not None:
        bias_spec = ParamSpec(shape=bias_shape, dtype=dtype)
        bias = bias_spec.validate('bias', bias)

    # the main part of the conv2d layer
    with tf.variable_scope(scope, default_name=name or 'deconv1d'):
        with tf.name_scope('output_shape'):
            # detect the input shape and axis arrangements
            input_shape = get_static_shape(input)
            if channels_last:
                c_axis, w_axis = -1, -2
            else:
                c_axis, w_axis = -2, -1

            output_shape = [None, None, None]
            output_shape[c_axis] = out_channels
            if given_output_shape is None:
                if input_shape[w_axis] is not None:
                    output_shape[w_axis] = get_deconv_output_length(
                        input_shape[w_axis], kernel_shape[0], strides[0],
                        padding
                    )
            else:
                if not is_tensor_object(given_output_shape):
                    output_shape[w_axis] = given_w

            # infer the batch shape in 3-d
            batch_shape = input_shape[:-2]
            if None not in batch_shape:
                output_shape[0] = int(np.prod(batch_shape))

            # now the static output shape is ready
            output_static_shape = tf.TensorShape(output_shape)

            # prepare for the dynamic batch shape
            if output_shape[0] is None:
                output_shape[0] = tf.reduce_prod(get_shape(input)[:-2])

            # prepare for the dynamic spatial dimensions
            if output_shape[w_axis] is None:
                if given_output_shape is None:
                    input_shape = get_shape(input)
                    if output_shape[w_axis] is None:
                        output_shape[w_axis] = get_deconv_output_length(
                            input_shape[w_axis], kernel_shape[0],
                            strides[0], padding
                        )
                else:
                    assert(is_tensor_object(given_output_shape))
                    with assert_deps([
                        assert_rank(given_output_shape, 1),
                        assert_scalar_equal(
                            tf.size(given_output_shape), 1)
                    ]):
                        output_shape[w_axis] = given_output_shape[0]

            # compose the final dynamic shape
            if any(is_tensor_object(s) for s in output_shape):
                output_shape = tf.stack(output_shape)
            else:
                output_shape = tuple(output_shape)

        # create the variables
        if kernel is None:
            kernel = model_variable(
                'kernel',
                shape=kernel_shape,
                dtype=dtype,
                initializer=kernel_initializer,
                regularizer=kernel_regularizer,
                constraint=kernel_constraint,
                trainable=trainable
            )

        maybe_add_histogram(kernel, 'kernel')
        kernel = maybe_check_numerics(kernel, 'kernel')

        if use_bias and bias is None:
            bias = model_variable(
                'bias',
                shape=bias_shape,
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                constraint=bias_constraint,
                trainable=trainable
            )
            maybe_add_histogram(bias, 'bias')
            bias = maybe_check_numerics(bias, 'bias')

        # flatten to 3d
        output, s1, s2 = flatten_to_ndims(input, 3)

        # do convolution or deconvolution
        output = tf.contrib.nn.conv1d_transpose(
            value=output,
            filter=kernel,
            output_shape=output_shape,
            stride=strides,
            padding=padding,
            data_format=data_format
        )
        if output_static_shape is not None:
            output.set_shape(output_static_shape)

        # add bias
        if use_bias:
            output = tf.add(output, bias)

        # apply the normalization function if specified
        if normalizer_fn is not None:
            output = normalizer_fn(output)

        # split into halves if gated
        if gated:
            output, gate = tf.split(output, 2, axis=c_axis)

        # apply the activation function if specified
        if activation_fn is not None:
            output = activation_fn(output)

        # apply the gate if required
        if gated:
            if gate_sigmoid_bias is None:
                gate_sigmoid_bias = model_variable(
                    'gate_sigmoid_bias',
                    shape=bias_shape,
                    initializer=bias_initializer,
                    regularizer=bias_regularizer,
                    constraint=bias_constraint,
                    trainable=trainable
                )
                maybe_add_histogram(gate_sigmoid_bias, 'gate_sigmoid_bias')
                gate_sigmoid_bias = maybe_check_numerics(gate_sigmoid_bias, 'gate_sigmoid_bias')
            output = output * tf.sigmoid(gate + gate_sigmoid_bias, name='gate')

        # unflatten back to original shape
        output = unflatten_from_ndims(output, s1, s2)

        maybe_add_histogram(output, 'output')
        output = maybe_check_numerics(output, 'output')

    return output
