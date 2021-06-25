import tensorflow as tf


__all__ = ['masked_reconstruct', 'mcmc_reconstruct']


def masked_reconstruct(reconstruct, x, u, mask, name=None):
    """
    Replace masked elements of `x` with reconstructed outputs.
    The potential anomaly points on x can be masked, and replaced by the reconstructed values.
    This can make the reconstruction more likely to be the normal pattern x should follow.
    Args:
        reconstruct ((tf.Tensor, tf.Tensor, tf.Tensor) -> tf.Tensor): Function for reconstructing `x`.
        x: The tensor to be reconstructed by `func`.
        u: Additional input for reconstructing `x`.
        mask: (tf.Tensor) mask, must be broadcastable into the shape of `x`.
            Indicating whether or not to mask each element of `x`.
        name (str): Name of this operation in TensorFlow graph.
            (default "masked_reconstruct")
    Returns:
        tf.Tensor: `x` with masked elements replaced by reconstructed outputs.
    """
    with tf.name_scope(name, default_name='masked_reconstruct'):
        x = tf.convert_to_tensor(x)  # type: tf.Tensor
        mask = tf.convert_to_tensor(mask, dtype=tf.int32)  # type: tf.Tensor

        mask = tf.broadcast_to(mask, tf.shape(x))

        # get reconstructed x. Currently only support mask the last point if pixelcnn decoder is used.
        x_recons = reconstruct(x, u, mask)

        # get masked outputs
        return tf.where(tf.cast(mask, dtype=tf.bool), x_recons, x)


def mcmc_reconstruct(reconstruct, x, u, mask, iter_count,
                                 back_prop=True, name=None):
    """
    Iteratively reconstruct `x` with `mask` for `iter_count` times.
    This method will call :func:`masked_reconstruct` for `iter_count` times,
    with the output from previous iteration as the input `x` for the next
    iteration.  The output of the final iteration would be returned.
    Args:
        reconstruct: Function for reconstructing `x`.
        x: The tensor to be reconstructed by `func`.
        u: Additional input for reconstructing `x`.
        mask: (tf.Tensor) mask, must be broadcastable into the shape of `x`.
            Indicating whether or not to mask each element of `x`.
        iter_count (int or tf.Tensor):
            Number of mcmc iterations(must be greater than 1).
        back_prop (bool): Whether or not to support back-propagation through
            all the iterations? (default :obj:`True`)
        name (str): Name of this operation in TensorFlow graph.
            (default "iterative_masked_reconstruct")
    Returns:
        tf.Tensor: The iteratively reconstructed `x`.
    """
    with tf.name_scope(name, default_name='mcmc_reconstruct'):

        # do the masked reconstructions
        x_recons, _ = tf.while_loop(
            lambda x_i, i: i < iter_count,
            lambda x_i, i: (masked_reconstruct(reconstruct, x_i, u, mask), i + 1),
            [x, tf.constant(0, dtype=tf.int32)],
            back_prop=back_prop
        )

        return x_recons
