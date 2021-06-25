import tensorflow as tf
import tfsnippet as spt
from tfsnippet.distributions import Distribution, Normal
from tfsnippet.stochastic import StochasticTensor
import numpy as np


class RecurrentDistribution(Distribution):
    def __init__(self, input, mean_layer, logstd_layer, z_dim, window_length,
                 is_reparameterized=True, check_numerics=False):
        batch_shape = spt.utils.concat_shapes([spt.utils.get_shape(input)[:-1], [z_dim]])
        batch_static_shape = tf.TensorShape(spt.utils.get_static_shape(input)[:-1] + (z_dim,))

        super(RecurrentDistribution, self).__init__(dtype=input.dtype,
                                                    is_continuous=True,
                                                    is_reparameterized=is_reparameterized,
                                                    value_ndims=0,
                                                    batch_shape=batch_shape,
                                                    batch_static_shape=batch_static_shape)

        self.mean_layer = mean_layer
        self.logstd_layer = logstd_layer
        self.z_dim = z_dim
        self._check_numerics = check_numerics
        self.window_length = window_length
        self.origin_input = input
        if len(input.shape) > 3:
            input, s1, s2 = spt.ops.flatten_to_ndims(input, 3)
            self.time_first_input = tf.transpose(input, [1, 0, 2])
            self.s1 = s1
            self.s2 = s2
            self.need_unflatten = True
        elif len(input.shape) == 3:
            self.time_first_input = tf.transpose(input, [1, 0, 2])  # (window_length, batch_size, feature_dim)
            self.s1 = None
            self.s2 = None
            self.need_unflatten = False
        else:
            raise ValueError('Invalid input shape in recurrent distribution.')
        self._mu = None
        self._logstd = None

    def mean(self):
        return self._mu

    def logstd(self):
        return self._logstd

    def _normal_pdf(self, x, mu, logstd):
        c = -0.5 * np.log(2 * np.pi)
        precision = tf.exp(-2 * logstd)
        if self._check_numerics:
            precision = tf.check_numerics(precision, "precision")
        log_prob = c - logstd - 0.5 * precision * tf.square(x - mu)
        if self._check_numerics:
            log_prob = tf.check_numerics(log_prob, 'log_prob')
        return log_prob

    def sample_step(self, a, t):
        z_previous, mu_z_previous, logstd_z_previous, _ = a
        noise, input = t

        # use the sampled z to derive the (mu. sigma) on next timestamp. may introduce small noise for each sample step.
        concat_input = spt.ops.broadcast_concat(input, z_previous, axis=-1)

        mu = self.mean_layer(concat_input)  # n_sample * batch_size * z_dim

        logstd = self.logstd_layer(concat_input)  # n_sample * batch_size * z_dim

        std = spt.utils.maybe_check_numerics(tf.exp(logstd), name='recurrent_distribution_z_std',
                                             message='z_std in recurrent distribution exceeds.')

        z_n = mu + std * noise

        log_prob = self._normal_pdf(z_n, mu, logstd)

        return z_n, mu, logstd, log_prob

    def log_prob_step(self, a, t):
        z_previous, _, _, log_prob_previous = a
        given_n, input_n = t

        concat_input = spt.ops.broadcast_concat(z_previous, input_n, axis=-1)

        mu = self.mean_layer(concat_input)

        logstd = self.logstd_layer(concat_input)

        log_prob_n = self._normal_pdf(given_n, mu, logstd)

        return given_n, mu, logstd, log_prob_n

    def sample(self, n_samples=None, is_reparameterized=None, group_ndims=0, compute_density=False,
               name=None):

        if n_samples is None:
            n_samples = 1
            n_samples_is_none = True
        else:
            n_samples_is_none = False

        with tf.name_scope(name=name, default_name='sample'):
            noise = tf.random_normal(shape=[n_samples, tf.shape(self.time_first_input)[0],
                                            tf.shape(self.time_first_input)[1], self.z_dim])  # (n_samples, window_length, batch_size, z_dim)
            noise = tf.transpose(noise, [1, 0, 2, 3])   # (window_length, n_samples, batch_size, z_dim)

            time_indices_shape = tf.convert_to_tensor([n_samples, tf.shape(self.time_first_input)[1], self.z_dim])  # (n_samples, batch_size, z_dim)

            results = tf.scan(fn=self.sample_step,
                              elems=(noise, self.time_first_input),
                              initializer=(tf.zeros(time_indices_shape),
                                           tf.zeros(time_indices_shape),
                                           tf.zeros(time_indices_shape),
                                           tf.zeros(time_indices_shape)),
                              back_prop=True
                              )  # 4 * window_length * n_samples * batch_size * z_dim

            samples = tf.transpose(results[0], [1, 2, 0, 3])  # n_samples * batch_size * window_length * z_dim

            log_prob = tf.transpose(results[-1], [1, 2, 0, 3])  # (n_samples, batch_size, window_length, z_dim)

            if self.need_unflatten:
                # unflatten to (n_samples, n_samples_of_input_tensor, batch_size, window_length, z_dim)
                samples = tf.stack([spt.ops.unflatten_from_ndims(samples[i], self.s1, self.s2) for i in range(n_samples)], axis=0)
                log_prob = tf.stack([spt.ops.unflatten_from_ndims(log_prob[i], self.s1, self.s2) for i in range(n_samples)], axis=0)

            log_prob = spt.reduce_group_ndims(tf.reduce_sum, log_prob, group_ndims)

            if n_samples_is_none:
                t = StochasticTensor(
                    distribution=self,
                    tensor=tf.reduce_mean(samples, axis=0),
                    group_ndims=group_ndims,
                    is_reparameterized=self.is_reparameterized,
                    log_prob=tf.reduce_mean(log_prob, axis=0)
                )
                self._mu = tf.reduce_mean(tf.transpose(results[1], [1, 2, 0, 3]), axis=0)
                self._logstd = tf.reduce_mean(tf.transpose(results[2], [1, 2, 0, 3]), axis=0)
                if self.need_unflatten:
                    self._mu = spt.ops.unflatten_from_ndims(self._mu, self.s1, self.s2)
                    self._logstd = spt.ops.unflatten_from_ndims(self._logstd, self.s1, self.s2)
            else:
                t = StochasticTensor(
                    distribution=self,
                    tensor=samples,
                    n_samples=n_samples,
                    group_ndims=group_ndims,
                    is_reparameterized=self.is_reparameterized,
                    log_prob=log_prob
                )
                self._mu = tf.transpose(results[1], [1, 2, 0, 3])
                self._logstd = tf.transpose(results[2], [1, 2, 0, 3])
                if self.need_unflatten:
                    self._mu = tf.stack([spt.ops.unflatten_from_ndims(self._mu[i], self.s1, self.s2) for i in range(n_samples)], axis=0)
                    self._logstd = tf.stack([spt.ops.unflatten_from_ndims(self._logstd[i], self.s1, self.s2) for i in range(n_samples)], axis=0)

            return t

    def log_prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='log_prob'):
            if self.need_unflatten:
                assert len(given.shape) == len(self.origin_input.shape)
                assert given.shape[0] == self.origin_input.shape[0]
                time_first_input = tf.transpose(self.origin_input, [2, 0, 1, 3])    # (window, sample, batch, feature)
                # time_indices_shape: (n_sample, batch_size, z_dim)
                time_indices_shape = tf.convert_to_tensor([tf.shape(given)[0], tf.shape(time_first_input)[2], self.z_dim])
                given = tf.transpose(given, [2, 0, 1, 3])
            else:
                if len(given.shape) > 3:    # (n_sample, batch_size, window_length, z_dim)
                    time_indices_shape = tf.convert_to_tensor([tf.shape(given)[0], tf.shape(self.time_first_input)[1], self.z_dim])
                    given = tf.transpose(given, [2, 0, 1, 3])
                    time_first_input = self.time_first_input
                else:                       # (batch_size, window_length, z_dim)
                    time_indices_shape = tf.convert_to_tensor([tf.shape(self.time_first_input)[1], self.z_dim])
                    given = tf.transpose(given, [1, 0, 2])
                    time_first_input = self.time_first_input
            results = tf.scan(fn=self.log_prob_step,
                               elems=(given, time_first_input),
                               initializer=(tf.zeros(time_indices_shape),
                                            tf.zeros(time_indices_shape),
                                            tf.zeros(time_indices_shape),
                                            tf.zeros(time_indices_shape)),
                               back_prop=True
                               )        # (window_length, ?, batch_size, z_dim)
            if len(given.shape) > 3:
                log_prob = tf.transpose(results[-1], [1, 2, 0, 3])
            else:
                log_prob = tf.transpose(results[-1], [1, 0, 2])

            log_prob = spt.reduce_group_ndims(tf.reduce_sum, log_prob, group_ndims)
            return log_prob

    def prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='prob'):
            log_prob = self.log_prob(given, group_ndims, name)
            return tf.exp(log_prob)
