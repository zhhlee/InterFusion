from enum import Enum
from typing import Optional, List

import logging
import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn, static_bidirectional_rnn
from tensorflow.contrib.framework import arg_scope
import tfsnippet as spt
from tfsnippet.bayes import BayesianNet
from tfsnippet.utils import (instance_reuse,
                             VarScopeObject,
                             reopen_variable_scope)
from tfsnippet.distributions import FlowDistribution, Normal
from tfsnippet.layers import l2_regularizer

import mltk
from algorithm.recurrent_distribution import RecurrentDistribution
from algorithm.real_nvp import dense_real_nvp
from algorithm.conv1d_ import conv1d, deconv1d


class RNNCellType(str, Enum):
    GRU = 'GRU'
    LSTM = 'LSTM'
    Basic = 'Basic'


class ModelConfig(mltk.Config):
    x_dim: int = -1
    z_dim: int = 3
    u_dim: int = 1
    window_length = 100
    output_shape: List[int] = [25, 25, 50, 50, 100]
    z2_dim: int = 13
    l2_reg = 0.0001
    posterior_flow_type: Optional[str] = mltk.config_field(choices=['rnvp', 'nf'], default='rnvp')
    # can be 'rnvp' for RealNVP, 'nf' for planarNF, None for not using posterior flow.
    posterior_flow_layers = 20
    rnn_cell: RNNCellType = RNNCellType.GRU  # can be 'GRU', 'LSTM' or 'Basic'
    rnn_hidden_units = 500
    use_leaky_relu = False
    use_bidirectional_rnn = False       # whether to use bidirectional rnn or not
    use_self_attention = False          # whether to use self-attention on hidden states before infer qz or not.
    unified_px_logstd = False
    dropout_feature = False             # dropout on the features in arnn
    logstd_min = -5.
    logstd_max = 2.
    use_prior_flow = False              # If True, use RealNVP prior flow to enhance the representation of p(z).
    prior_flow_layers = 20

    connect_qz = True
    connect_pz = True


# The final InterFusion model.
class MTSAD(VarScopeObject):

    def __init__(self, config: ModelConfig, name=None, scope=None):
        self.config = config
        super(MTSAD, self).__init__(name=name, scope=scope)

        with reopen_variable_scope(self.variable_scope):
            if self.config.rnn_cell == RNNCellType.Basic:
                self.d_fw_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.rnn_hidden_units, name='d_fw_cell')
                self.a_fw_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.rnn_hidden_units, name='a_fw_cell')
                if self.config.use_bidirectional_rnn:
                    self.d_bw_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.rnn_hidden_units, name='d_bw_cell')
                    self.a_bw_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.rnn_hidden_units, name='a_bw_cell')
            elif self.config.rnn_cell == RNNCellType.LSTM:
                self.d_fw_cell = tf.nn.rnn_cell.LSTMCell(self.config.rnn_hidden_units, name='d_fw_cell')
                self.a_fw_cell = tf.nn.rnn_cell.LSTMCell(self.config.rnn_hidden_units, name='a_fw_cell')
                if self.config.use_bidirectional_rnn:
                    self.d_bw_cell = tf.nn.rnn_cell.LSTMCell(self.config.rnn_hidden_units, name='d_bw_cell')
                    self.a_bw_cell = tf.nn.rnn_cell.LSTMCell(self.config.rnn_hidden_units, name='a_bw_cell')
            elif self.config.rnn_cell == RNNCellType.GRU:
                self.d_fw_cell = tf.nn.rnn_cell.GRUCell(self.config.rnn_hidden_units, name='d_fw_cell')
                self.a_fw_cell = tf.nn.rnn_cell.GRUCell(self.config.rnn_hidden_units, name='a_fw_cell')
                if self.config.use_bidirectional_rnn:
                    self.d_bw_cell = tf.nn.rnn_cell.GRUCell(self.config.rnn_hidden_units, name='d_bw_cell')
                    self.a_bw_cell = tf.nn.rnn_cell.GRUCell(self.config.rnn_hidden_units, name='a_bw_cell')
            else:
                raise ValueError('rnn cell must be one of GRU, LSTM or Basic.')

            if self.config.posterior_flow_type == 'nf':
                self.posterior_flow = spt.layers.planar_normalizing_flows(n_layers=self.config.posterior_flow_layers,
                                                                          scope='posterior_flow')
            elif self.config.posterior_flow_type == 'rnvp':
                self.posterior_flow = dense_real_nvp(flow_depth=self.config.posterior_flow_layers,
                                                     activation=tf.nn.leaky_relu if self.config.use_leaky_relu else tf.nn.relu,
                                                     kernel_regularizer=l2_regularizer(self.config.l2_reg),
                                                     scope='posterior_flow')
            else:
                self.posterior_flow = None

            if self.config.use_prior_flow:
                self.prior_flow = dense_real_nvp(flow_depth=self.config.prior_flow_layers,
                                                 activation=tf.nn.leaky_relu if self.config.use_leaky_relu else tf.nn.relu,
                                                 kernel_regularizer=l2_regularizer(self.config.l2_reg),
                                                 is_prior_flow=True,
                                                 scope='prior_flow')
            else:
                self.prior_flow = None

    def _my_rnn_net(self, x, window_length, fw_cell, bw_cell=None,
                    time_axis=1, use_bidirectional_rnn=False):
        """
        Get the base rnn model.
        :param x: The rnn input.
        :param window_length: The window length of input along time axis.
        :param fw_cell: Forward rnn cell.
        :param bw_cell: Optional. Backward rnn cell, only use when config.use_bidirectional_rnn=True.
        :param time_axis: Which is the time axis in input x, default 1.
        :param use_bidirectional_rnn: Whether or not use bidirectional rnn. Default false.
        :return: Tensor (batch_size, window_length, rnn_hidden_units). The output of rnn.
        """

        x = tf.unstack(value=x, num=window_length, axis=time_axis)

        if use_bidirectional_rnn:
            outputs, _, _ = static_bidirectional_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
        else:
            outputs, _ = static_rnn(fw_cell, x, dtype=tf.float32)

        outputs = tf.stack(outputs, axis=time_axis)     # (batch_size, window_length, rnn_hidden_units)
        return outputs

    @instance_reuse
    def a_rnn_net(self, x, window_length, time_axis=1,
                  use_bidirectional_rnn=False, use_self_attention=False, is_training=False):
        """
        Reverse rnn network a, capture the future information in qnet.
        """
        def dropout_fn(input):
            return tf.layers.dropout(input, rate=.5, training=is_training)

        flag = False
        if len(x.shape) == 4:               # (n_samples, batch_size, window_length, x_dim)
            x, s1, s2 = spt.ops.flatten_to_ndims(x, 3)
            flag = True
        elif len(x.shape) != 3:
            logging.error('rnn input shape error.')

        # reverse the input sequence
        reversed_x = tf.reverse(x, axis=[time_axis])

        if use_bidirectional_rnn:
            reversed_outputs = self._my_rnn_net(x=reversed_x, window_length=window_length, fw_cell=self.a_fw_cell,
                                       bw_cell=self.a_bw_cell, time_axis=time_axis,
                                       use_bidirectional_rnn=use_bidirectional_rnn)
        else:
            reversed_outputs = self._my_rnn_net(x=reversed_x, window_length=window_length, fw_cell=self.a_fw_cell,
                                       time_axis=time_axis, use_bidirectional_rnn=use_bidirectional_rnn)

        outputs = tf.reverse(reversed_outputs, axis=[time_axis])

        # self attention
        if use_self_attention:
            outputs1 = spt.layers.dense(outputs, 500, activation_fn=tf.nn.tanh, use_bias=True, scope='arnn_attention_dense1')
            outputs1 = tf.nn.softmax(spt.layers.dense(outputs1, window_length,
                                                      use_bias=False, scope='arnn_attention_dense2'), axis=1)
            M_t = tf.matmul(tf.transpose(outputs, perm=[0, 2, 1]), outputs1)
            outputs = tf.transpose(M_t, perm=[0, 2, 1])

        # feature extraction layers
        outputs = spt.layers.dense(outputs, units=500, activation_fn=tf.nn.leaky_relu if self.config.use_leaky_relu else tf.nn.relu,
                                   kernel_regularizer=l2_regularizer(self.config.l2_reg), scope='arnn_feature_dense1')
        if self.config.dropout_feature:
            outputs = dropout_fn(outputs)
        outputs = spt.layers.dense(outputs, units=500, activation_fn=tf.nn.leaky_relu if self.config.use_leaky_relu else tf.nn.relu,
                                   kernel_regularizer=l2_regularizer(self.config.l2_reg), scope='arnn_feature_dense2')
        if self.config.dropout_feature:
            outputs = dropout_fn(outputs)

        if flag:
            outputs = spt.ops.unflatten_from_ndims(outputs, s1, s2)

        return outputs

    @instance_reuse
    def qz_mean_layer(self, x):
        return spt.layers.dense(x, units=self.config.z_dim, scope='qz_mean')

    @instance_reuse
    def qz_logstd_layer(self, x):
        return tf.clip_by_value(spt.layers.dense(x, units=self.config.z_dim, scope='qz_logstd'),
                                clip_value_min=self.config.logstd_min, clip_value_max=self.config.logstd_max)

    @instance_reuse
    def pz_mean_layer(self, x):
        return spt.layers.dense(x, units=self.config.z_dim, scope='pz_mean')

    @instance_reuse
    def pz_logstd_layer(self, x):
        return tf.clip_by_value(spt.layers.dense(x, units=self.config.z_dim, scope='pz_logstd'),
                                clip_value_min=self.config.logstd_min, clip_value_max=self.config.logstd_max)

    @instance_reuse
    def hz2_deconv(self, z2):
        with arg_scope([deconv1d],
                       kernel_size=5,
                       activation_fn=tf.nn.leaky_relu if self.config.use_leaky_relu else tf.nn.relu,
                       kernel_regularizer=l2_regularizer(self.config.l2_reg)):
            h_z = deconv1d(z2, out_channels=self.config.x_dim, output_shape=self.config.output_shape[0], strides=2)
            h_z = deconv1d(h_z, out_channels=self.config.x_dim, output_shape=self.config.output_shape[1], strides=1)
            h_z = deconv1d(h_z, out_channels=self.config.x_dim, output_shape=self.config.output_shape[2], strides=2)
            h_z = deconv1d(h_z, out_channels=self.config.x_dim, output_shape=self.config.output_shape[3], strides=1)
            h_z2 = deconv1d(h_z, out_channels=self.config.x_dim, output_shape=self.config.output_shape[4], strides=2)
        return h_z2

    @instance_reuse
    def q_net(self, x, observed=None, u=None, n_z=None, is_training=False):
        # vs.name = self.variable_scope.name + "/q_net"
        logging.info('q_net builder: %r', locals())

        net = BayesianNet(observed=observed)

        def dropout_fn(input):
            return tf.layers.dropout(input, rate=.5, training=is_training)

        # use the pretrained z2 which compress along the time dimension
        qz2_mean, qz2_logstd = self.h_for_qz(x, is_training=is_training)

        qz2_distribution = Normal(mean=qz2_mean, logstd=qz2_logstd)

        qz2_distribution = qz2_distribution.batch_ndims_to_value(2)

        z2 = net.add('z2', qz2_distribution, n_samples=n_z, is_reparameterized=True)

        # d_{1:t} from deconv
        h_z = self.h_for_px(z2)

        # a_{1:t}, (batch_size, window_length, dense_hidden_units)
        arnn_out = self.a_rnn_net(h_z, window_length=self.config.window_length,
                                  use_bidirectional_rnn=self.config.use_bidirectional_rnn,
                                  use_self_attention=self.config.use_self_attention,
                                  is_training=is_training)

        if self.config.connect_qz:
            qz_distribution = RecurrentDistribution(arnn_out,
                                                    mean_layer=self.qz_mean_layer, logstd_layer=self.qz_logstd_layer,
                                                    z_dim=self.config.z_dim, window_length=self.config.window_length)
        else:
            qz_mean = spt.layers.dense(arnn_out, units=self.config.z_dim, scope='qz1_mean')
            qz_logstd = tf.clip_by_value(spt.layers.dense(arnn_out, units=self.config.z_dim, scope='qz1_logstd'),
                                         clip_value_min=self.config.logstd_min, clip_value_max=self.config.logstd_max)
            qz_distribution = Normal(mean=qz_mean, logstd=qz_logstd)

        if self.posterior_flow is not None:
            qz_distribution = FlowDistribution(distribution=qz_distribution, flow=self.posterior_flow).batch_ndims_to_value(1)
        else:
            qz_distribution = qz_distribution.batch_ndims_to_value(2)

        z1 = net.add('z1', qz_distribution, is_reparameterized=True)

        return net

    @instance_reuse
    def p_net(self, observed=None, u=None, n_z=None, is_training=False):
        logging.info('p_net builder: %r', locals())

        net = BayesianNet(observed=observed)

        pz2_distribution = Normal(mean=tf.zeros([self.config.z2_dim, self.config.x_dim]),
                                 logstd=tf.zeros([self.config.z2_dim, self.config.x_dim])).batch_ndims_to_value(2)

        z2 = net.add('z2', pz2_distribution, n_samples=n_z, is_reparameterized=True)

        # e_{1:t} from deconv, shared params
        h_z2 = self.h_for_px(z2)

        if self.config.connect_pz:
            pz_distribution = RecurrentDistribution(h_z2,
                                                    mean_layer=self.pz_mean_layer, logstd_layer=self.pz_logstd_layer,
                                                    z_dim=self.config.z_dim, window_length=self.config.window_length)
        else:
            # non-recurrent pz
            pz_mean = spt.layers.dense(h_z2, units=self.config.z_dim, scope='pz_mean')
            pz_logstd = tf.clip_by_value(spt.layers.dense(h_z2,
                                                          units=self.config.z_dim, scope='pz_logstd'),
                                                          clip_value_min=self.config.logstd_min,
                                                          clip_value_max=self.config.logstd_max)
            pz_distribution = Normal(mean=pz_mean, logstd=pz_logstd)

        if self.prior_flow is not None:
            pz_distribution = FlowDistribution(distribution=pz_distribution, flow=self.prior_flow).batch_ndims_to_value(1)
        else:
            pz_distribution = pz_distribution.batch_ndims_to_value(2)

        z1 = net.add('z1', pz_distribution, is_reparameterized=True)

        h_z1 = spt.layers.dense(z1, units=self.config.x_dim)

        h_z = spt.ops.broadcast_concat(h_z1, h_z2, axis=-1)

        h_z = spt.layers.dense(h_z, units=500, activation_fn=tf.nn.leaky_relu if self.config.use_leaky_relu else tf.nn.relu,
                               kernel_regularizer=l2_regularizer(self.config.l2_reg), scope='feature_dense1')

        h_z = spt.layers.dense(h_z, units=500, activation_fn=tf.nn.leaky_relu if self.config.use_leaky_relu else tf.nn.relu,
                               kernel_regularizer=l2_regularizer(self.config.l2_reg), scope='feature_dense2')

        x_mean = spt.layers.dense(h_z, units=self.config.x_dim, scope='x_mean')
        if self.config.unified_px_logstd:
            x_logstd = tf.clip_by_value(
                tf.get_variable(name='x_logstd', shape=(), trainable=True, dtype=tf.float32,
                                initializer=tf.constant_initializer(-1., dtype=tf.float32)),
                                clip_value_min=self.config.logstd_min, clip_value_max=self.config.logstd_max)
        else:
            x_logstd = tf.clip_by_value(spt.layers.dense(h_z, units=self.config.x_dim, scope='x_logstd'),
                                        clip_value_min=self.config.logstd_min, clip_value_max=self.config.logstd_max)

        x = net.add('x',
                    Normal(mean=x_mean, logstd=x_logstd).batch_ndims_to_value(2),
                    is_reparameterized=True)

        return net

    def reconstruct(self, x, u, mask, n_z=None):
        with tf.name_scope('model.reconstruct'):
            qnet = self.q_net(x=x, u=u, n_z=n_z)
            pnet = self.p_net(observed={'z1': qnet['z1'], 'z2': qnet['z2']}, u=u)
        return pnet['x']

    def get_score(self, x_embed, x_eval, u, n_z=None):
        with tf.name_scope('model.get_score'):
            qnet = self.q_net(x=x_embed, u=u, n_z=n_z)
            pnet = self.p_net(observed={'z1': qnet['z1'], 'z2': qnet['z2']}, u=u)
            score = pnet['x'].distribution.base_distribution.log_prob(x_eval)
            recons_mean = pnet['x'].distribution.base_distribution.mean
            recons_std = pnet['x'].distribution.base_distribution.std
            if n_z is not None:
                score = tf.reduce_mean(score, axis=0)
                recons_mean = tf.reduce_mean(recons_mean, axis=0)
                recons_std = tf.reduce_mean(recons_std, axis=0)
        return score, recons_mean, recons_std

    @instance_reuse
    def h_for_qz(self, x, is_training=False):
        with arg_scope([conv1d],
                       kernel_size=5,
                       activation_fn=tf.nn.leaky_relu if self.config.use_leaky_relu else tf.nn.relu,
                       kernel_regularizer=l2_regularizer(self.config.l2_reg)):
            h_x = conv1d(x, out_channels=self.config.x_dim, strides=2)   # 50
            h_x = conv1d(h_x, out_channels=self.config.x_dim)
            h_x = conv1d(h_x, out_channels=self.config.x_dim, strides=2)        # 25
            h_x = conv1d(h_x, out_channels=self.config.x_dim)
            h_x = conv1d(h_x, out_channels=self.config.x_dim, strides=2)        # 13

        qz_mean = conv1d(h_x, kernel_size=1, out_channels=self.config.x_dim)
        qz_logstd = conv1d(h_x, kernel_size=1, out_channels=self.config.x_dim)
        qz_logstd = tf.clip_by_value(qz_logstd, clip_value_min=self.config.logstd_min,
                                     clip_value_max=self.config.logstd_max)
        return qz_mean, qz_logstd

    @instance_reuse
    def h_for_px(self, z):
        with arg_scope([deconv1d],
                       kernel_size=5,
                       activation_fn=tf.nn.leaky_relu if self.config.use_leaky_relu else tf.nn.relu,
                       kernel_regularizer=l2_regularizer(self.config.l2_reg)):
            h_z = deconv1d(z, out_channels=self.config.x_dim, output_shape=self.config.output_shape[0], strides=2)
            h_z = deconv1d(h_z, out_channels=self.config.x_dim, output_shape=self.config.output_shape[1], strides=1)
            h_z = deconv1d(h_z, out_channels=self.config.x_dim, output_shape=self.config.output_shape[2], strides=2)
            h_z = deconv1d(h_z, out_channels=self.config.x_dim, output_shape=self.config.output_shape[3], strides=1)
            h_z = deconv1d(h_z, out_channels=self.config.x_dim, output_shape=self.config.output_shape[4], strides=2)
        return h_z

    @instance_reuse
    def pretrain_q_net(self, x, observed=None, n_z=None, is_training=False):
        # vs.name = self.variable_scope.name + "/q_net"
        logging.info('pretrain_q_net builder: %r', locals())

        net = BayesianNet(observed=observed)

        def dropout_fn(input):
            return tf.layers.dropout(input, rate=.5, training=is_training)

        qz_mean, qz_logstd = self.h_for_qz(x, is_training=is_training)

        qz_distribution = Normal(mean=qz_mean, logstd=qz_logstd)

        qz_distribution = qz_distribution.batch_ndims_to_value(2)

        z = net.add('z', qz_distribution, n_samples=n_z, is_reparameterized=True)

        return net

    @instance_reuse
    def pretrain_p_net(self, observed=None, n_z=None, is_training=False):
        logging.info('p_net builder: %r', locals())

        net = BayesianNet(observed=observed)

        pz_distribution = Normal(mean=tf.zeros([self.config.z2_dim, self.config.x_dim]),
                                 logstd=tf.zeros([self.config.z2_dim, self.config.x_dim]))

        pz_distribution = pz_distribution.batch_ndims_to_value(2)

        z = net.add('z',
                    pz_distribution,
                    n_samples=n_z, is_reparameterized=True)

        h_z = self.h_for_px(z)

        px_mean = conv1d(h_z, kernel_size=1, out_channels=self.config.x_dim, scope='pre_px_mean')
        px_logstd = conv1d(h_z, kernel_size=1, out_channels=self.config.x_dim, scope='pre_px_logstd')
        px_logstd = tf.clip_by_value(px_logstd, clip_value_min=self.config.logstd_min,
                                     clip_value_max=self.config.logstd_max)

        x = net.add('x',
                    Normal(mean=px_mean, logstd=px_logstd).batch_ndims_to_value(2),
                    is_reparameterized=True)

        return net
