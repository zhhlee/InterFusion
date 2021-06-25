import tensorflow as tf
import tfsnippet as spt
from tensorflow.contrib.framework import arg_scope
import numpy as np
from tfsnippet.layers.flows.utils import ZeroLogDet


class FeatureReversingFlow(spt.layers.FeatureMappingFlow):

    def __init__(self, axis=-1, value_ndims=1, name=None, scope=None):
        super(FeatureReversingFlow, self).__init__(
            axis=int(axis), value_ndims=value_ndims, name=name, scope=scope)

    @property
    def explicitly_invertible(self):
        return True

    def _build(self, input=None):
        pass

    def _reverse_feature(self, x, compute_y, compute_log_det):
        n_features = spt.utils.get_static_shape(x)[self.axis]
        if n_features is None:
            raise ValueError('The feature dimension must be fixed.')
        assert (0 > self.axis >= -self.value_ndims >=
                -len(spt.utils.get_static_shape(x)))
        permutation = np.asarray(list(reversed(range(n_features))),
                                 dtype=np.int32)

        # compute y
        y = None
        if compute_y:
            y = tf.gather(x, permutation, axis=self.axis)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = ZeroLogDet(spt.utils.get_shape(x)[:-self.value_ndims],
                                 x.dtype.base_dtype)

        return y, log_det

    def _transform(self, x, compute_y, compute_log_det):
        return self._reverse_feature(x, compute_y, compute_log_det)

    def _inverse_transform(self, y, compute_x, compute_log_det):
        return self._reverse_feature(y, compute_x, compute_log_det)


def dense_real_nvp(flow_depth: int,
                   activation,
                   kernel_regularizer,
                   scope: str,
                   use_invertible_flow=True,
                   strict_invertible=False,
                   use_actnorm_flow=False,
                   dense_coupling_n_hidden_layers=1,
                   dense_coupling_n_hidden_units=100,
                   coupling_scale_shift_initializer='zero',     # 'zero' or 'normal'
                   coupling_scale_shift_normal_initializer_stddev=0.001,
                   coupling_scale_type='sigmoid',               # 'sigmoid' or 'exp'
                   coupling_sigmoid_scale_bias=2.,
                   is_prior_flow=False) -> spt.layers.BaseFlow:
    def shift_and_scale(x1, n2):

        with arg_scope([spt.layers.dense],
                       activation_fn=activation,
                       kernel_regularizer=kernel_regularizer):
            h = x1
            for j in range(dense_coupling_n_hidden_layers):
                h = spt.layers.dense(h,
                                     units=dense_coupling_n_hidden_units,
                                     scope='hidden_{}'.format(j))

        # compute shift and scale
        if coupling_scale_shift_initializer == 'zero':
            pre_params_initializer = tf.zeros_initializer()
        else:
            pre_params_initializer = tf.random_normal_initializer(
                stddev=coupling_scale_shift_normal_initializer_stddev)
        pre_params = spt.layers.dense(h,
                                      units=n2 * 2,
                                      kernel_initializer=pre_params_initializer,
                                      scope='shift_and_scale',)

        shift = pre_params[..., :n2]
        scale = pre_params[..., n2:]

        return shift, scale

    with tf.variable_scope(scope):
        flows = []
        for i in range(flow_depth):
            level = []
            if use_invertible_flow:
                level.append(
                    spt.layers.InvertibleDense(
                        strict_invertible=strict_invertible)
                )
            else:
                level.append(FeatureReversingFlow())
            level.append(
                spt.layers.CouplingLayer(
                    tf.make_template(
                        'coupling', shift_and_scale, create_scope_now_=True),
                    scale_type=coupling_scale_type,
                    sigmoid_scale_bias=coupling_sigmoid_scale_bias,
                )
            )
            if use_actnorm_flow:
                level.append(spt.layers.ActNorm())
            flows.extend(level)
        flow = spt.layers.SequentialFlow(flows)

    if is_prior_flow:
        flow = flow.invert()

    return flow
