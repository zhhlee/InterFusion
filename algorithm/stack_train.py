# -*- coding: utf-8 -*-
import os

import logging
import time
import numpy as np
import tensorflow as tf
import tfsnippet as spt
from tfsnippet.scaffold import TrainLoop
from tfsnippet.trainer import Trainer, Evaluator
import mltk
from algorithm.utils import get_data_dim, get_data, get_sliding_window_data_flow, time_generator, GraphNodes
import random

from algorithm.InterFusion import ModelConfig, MTSAD
from algorithm.InterFusion_swat import MTSAD_SWAT
from algorithm.stack_predict import PredictConfig


class TrainConfig(mltk.Config):
    # training params
    batch_size = 100
    pretrain_max_epoch = 20
    max_epoch = 20
    train_start = 0
    max_train_size = None  # `None` means full train set
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 10
    lr_anneal_step_freq = None
    pretrain_lr_anneal_epoch_freq = 10

    early_stopping = True
    valid_portion = 0.3

    save_test_stats = True


class ExpConfig(mltk.Config):
    seed = int(time.time())

    dataset = 'omi-1'

    # model params
    model = ModelConfig()

    @mltk.root_checker()
    def _model_post_checker(self, v: 'ExpConfig'):
        if v.model.x_dim == -1:
            v.model.x_dim = get_data_dim(v.dataset)
        if v.dataset == 'SWaT':
            v.model.z_dim = 2
        if v.dataset == 'WADI':
            v.model.z_dim = 4

    use_time_info = False  # whether to use time information (minute, hour, day) as input u. discarded.

    model_type = 'mtsad'

    # train params
    train = TrainConfig()

    @mltk.root_checker()
    def _train_post_checker(self, v: 'ExpConfig'):
        if v.dataset == 'SWaT' or v.dataset == 'WADI':
            v.train.max_epoch = 15
            v.train.save_test_stats = False
            v.train.pretrain_max_epoch = 10
            v.train.pretrain_lr_anneal_epoch_freq = 5
            v.train.lr_anneal_epoch_freq = 5
        if v.dataset == 'SWaT':
            v.train.initial_lr = 0.0005
        if v.dataset == 'WADI':
            v.train.initial_lr = 0.0002

    test = PredictConfig()

    # debugging params
    write_summary = False
    write_histogram_summary = False
    check_numerics = False
    save_results = True
    save_ckpt = True
    ckpt_epoch_freq = 10
    ckpt_max_keep = 10
    pretrain_ckpt_epoch_freq = 20
    pretrain_ckpt_max_keep = 10

    exp_dir_save_path = None    # The file path to save the exp dirs for batch run training on different datasets.


def get_lr_value(init_lr,
                 anneal_factor,
                 anneal_freq,
                 loop: spt.TrainLoop,
                 ) -> spt.DynamicValue:
    """
    Get the learning rate scheduler for specified experiment.

    Args:
        exp: The experiment object.
        loop: The train loop object.

    Returns:
        A dynamic value, which returns the learning rate each time
        its `.get()` is called.
    """
    return spt.AnnealingScalar(
        loop=loop,
        initial_value=init_lr,
        ratio=anneal_factor,
        epochs=anneal_freq,
    )


def sgvb_loss(qnet, pnet, metrics_dict: GraphNodes, prefix='train_', name=None):
    with tf.name_scope(name, default_name='sgvb_loss'):
        logpx_z = pnet['x'].log_prob(name='logpx_z')
        logpz1_z2 = pnet['z1'].log_prob(name='logpz1_z2')
        logpz2 = pnet['z2'].log_prob(name='logpz2')
        logpz = logpz1_z2 + logpz2
        logqz1_x = qnet['z1'].log_prob(name='logqz1_x')
        logqz2_x = qnet['z2'].log_prob(name='logqz2_x')
        logqz_x = logqz1_x + logqz2_x

        recons_term = tf.reduce_mean(logpx_z)
        kl_term = tf.reduce_mean(logqz_x - logpz)
        metrics_dict[prefix + 'recons'] = recons_term
        metrics_dict[prefix + 'kl'] = kl_term

        return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def main(exp: mltk.Experiment[ExpConfig], config: ExpConfig):
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # print the current seed and generate three seeds
    logging.info('Current random seed: %s', config.seed)
    np.random.seed(config.seed)
    random.seed(np.random.randint(0xffffffff))
    tf.set_random_seed(np.random.randint(0xffffffff))
    np.random.seed(np.random.randint(0xffffffff))

    spt.settings.check_numerics = config.check_numerics
    spt.settings.enable_assertions = False

    # print the config
    print(mltk.format_key_values(config, title='Configurations'))
    print('')

    # open the result object and prepare for result directories
    exp.make_dirs('train_summary')
    exp.make_dirs('result_params')
    exp.make_dirs('ckpt_params')
    exp.make_dirs(config.test.output_dirs)

    # prepare the data
    # simple data
    (x_train, _), (x_test, y_test) = \
        get_data(config.dataset, config.train.max_train_size, config.test.max_test_size,
                 train_start=config.train.train_start, test_start=config.test.test_start,
                 valid_portion=config.train.valid_portion)

    if config.use_time_info:
        u_train = np.asarray([time_generator(_i) for _i in range(len(x_train))])  # (train_size, u_dim)
        u_test = np.asarray([time_generator(len(x_train) + _i) for _i in range(len(x_test))])  # (test_size, u_dim)
    else:
        u_train = np.zeros([len(x_train), config.model.u_dim])  # (train_size, u_dim)
        u_test = np.zeros([len(x_test), config.model.u_dim])

    split_idx = int(len(x_train) * config.train.valid_portion)
    x_train, x_valid = x_train[:-split_idx], x_train[-split_idx:]
    u_train, u_valid = u_train[:-split_idx], u_train[-split_idx:]

    # prepare data_flow
    train_flow = get_sliding_window_data_flow(window_size=config.model.window_length,
                                              batch_size=config.train.batch_size,
                                              x=x_train, u=u_train, shuffle=True, skip_incomplete=True)
    valid_flow = get_sliding_window_data_flow(window_size=config.model.window_length,
                                              batch_size=config.train.batch_size,
                                              x=x_valid, u=u_valid, shuffle=False, skip_incomplete=False)

    # build computation graph
    if config.dataset == 'SWaT' or config.dataset == 'WADI':
        model = MTSAD_SWAT(config.model, scope='model')
    else:
        model = MTSAD(config.model, scope='model')

    # input placeholders
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, config.model.window_length, config.model.x_dim],
                             name='input_x')
    input_u = tf.placeholder(dtype=tf.float32, shape=[None, config.model.window_length, config.model.u_dim],
                             name='input_u')
    learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

    # derive training nodes
    with tf.name_scope('training'):
        # pretrain time-vae to get z2
        pretrain_q_net = model.pretrain_q_net(input_x, is_training=is_training)
        pretrain_chain = pretrain_q_net.chain(model.pretrain_p_net, observed={'x': input_x}, is_training=is_training)
        pretrain_loss = tf.reduce_mean(pretrain_chain.vi.training.sgvb()) + tf.losses.get_regularization_loss()
        pretrain_train_recons = tf.reduce_mean(pretrain_chain.model['x'].log_prob())

        # train the whole network with z1 and z2
        train_q_net = model.q_net(input_x, u=input_u, is_training=is_training)
        train_chain = train_q_net.chain(model.p_net, observed={'x': input_x}, u=input_u, is_training=is_training)
        train_metrics = GraphNodes()
        vae_loss = sgvb_loss(train_chain.variational, train_chain.model, train_metrics, name='train_sgvb_loss')
        reg_loss = tf.losses.get_regularization_loss()
        loss = vae_loss + reg_loss
        train_metrics['loss'] = loss

    with tf.name_scope('validation'):
        # pretrain validation
        pretrain_valid_q_net = model.pretrain_q_net(input_x, n_z=config.test.test_n_z)
        pretrain_valid_chain = pretrain_valid_q_net.chain(model.pretrain_p_net, observed={'x': input_x}, latent_axis=0)
        pretrain_valid_loss = tf.reduce_mean(pretrain_valid_chain.vi.training.sgvb()) + tf.losses.get_regularization_loss()
        pretrain_valid_recons = tf.reduce_mean(pretrain_valid_chain.model['x'].log_prob())

        # validation of the whole network
        valid_q_net = model.q_net(input_x, u=input_u, n_z=config.test.test_n_z)
        valid_chain = valid_q_net.chain(model.p_net, observed={'x': input_x}, latent_axis=0, u=input_u)
        valid_metrics = GraphNodes()
        valid_loss = sgvb_loss(valid_chain.variational, valid_chain.model, valid_metrics, prefix='valid_',
                               name='valid_sgvb_loss') + tf.losses.get_regularization_loss()
        valid_metrics['valid_loss'] = valid_loss

    # pretrain
    pre_variables_to_save = sum(
            [tf.global_variables('model/pretrain_q_net'), tf.global_variables('model/pretrain_p_net'),
             tf.global_variables('model/h_for_qz'), tf.global_variables('model/h_for_px')],
            []
    )
    pre_train_params = sum(
            [tf.trainable_variables('model/pretrain_q_net'), tf.trainable_variables('model/pretrain_p_net'),
             tf.trainable_variables('model/h_for_qz'), tf.trainable_variables('model/h_for_px')],
            []
    )
    pre_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    pre_gradients = pre_optimizer.compute_gradients(pretrain_loss, var_list=pre_train_params)
    with tf.name_scope('PreClipGradients'):
        for i, (g, v) in enumerate(pre_gradients):
            if g is not None:
                pre_gradients[i] = (tf.clip_by_norm(
                    spt.utils.maybe_check_numerics(g, message='gradient on %s exceed' % str(v.name)), 10), v)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        pre_train_op = pre_optimizer.apply_gradients(pre_gradients)

    # obtain params and gradients (whole model)
    variables_to_save = tf.global_variables()
    train_params = tf.trainable_variables()

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients = optimizer.compute_gradients(loss, var_list=train_params)
    # clip gradient by norm
    with tf.name_scope('ClipGradients'):
        for i, (g, v) in enumerate(gradients):
            if g is not None:
                gradients[i] = (tf.clip_by_norm(
                    spt.utils.maybe_check_numerics(g, message="gradient on %s exceed" % str(v.name)), 10), v)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.apply_gradients(gradients)

    pre_var_groups = [
        model.variable_scope.name + '/pretrain_q_net',

        model.variable_scope.name + '/pretrain_p_net',

        model.variable_scope.name + '/h_for_qz',

        model.variable_scope.name + '/h_for_px'
    ]

    var_groups = [
        # for q_net
        model.variable_scope.name + '/q_net',

        # for p_net
        model.variable_scope.name + '/p_net',

        # for flow
        model.variable_scope.name + '/posterior_flow'
    ]

    var_initializer = tf.variables_initializer(tf.global_variables())

    train_flow = train_flow.threaded(5)
    valid_flow = valid_flow.threaded(5)

    pre_loop = TrainLoop(param_vars=pre_variables_to_save,
                         var_groups=pre_var_groups,
                         max_epoch=config.train.pretrain_max_epoch,
                         summary_dir=(exp.abspath('pre_train_summary') if config.write_summary else None),
                         summary_graph=tf.get_default_graph(),
                         summary_commit_freqs={'pretrain_loss': 10},
                         early_stopping=config.train.early_stopping,
                         valid_metric_name='pretrain_valid_loss',
                         valid_metric_smaller_is_better=True,
                         checkpoint_dir=(exp.abspath('pre_ckpt_params') if config.save_ckpt else None),
                         checkpoint_epoch_freq=config.pretrain_ckpt_epoch_freq,
                         checkpoint_max_to_keep=config.pretrain_ckpt_max_keep)

    loop = TrainLoop(param_vars=variables_to_save,
                     var_groups=var_groups,
                     max_epoch=config.train.max_epoch,
                     summary_dir=(exp.abspath('train_summary')
                                  if config.write_summary else None),
                     summary_graph=tf.get_default_graph(),
                     summary_commit_freqs={'loss': 10},
                     early_stopping=config.train.early_stopping,
                     valid_metric_name='valid_loss',
                     valid_metric_smaller_is_better=True,
                     checkpoint_dir=(exp.abspath('ckpt_params')
                                     if config.save_ckpt else None),
                     checkpoint_epoch_freq=config.ckpt_epoch_freq,
                     checkpoint_max_to_keep=config.ckpt_max_keep
                     )

    if config.write_histogram_summary:
        summary_op = tf.summary.merge_all()
    else:
        summary_op = None

    pre_lr_value = get_lr_value(config.train.initial_lr, config.train.lr_anneal_factor,
                                config.train.pretrain_lr_anneal_epoch_freq, pre_loop)
    lr_value = get_lr_value(config.train.initial_lr, config.train.lr_anneal_factor,
                            config.train.lr_anneal_epoch_freq, loop)

    pre_trainer = Trainer(loop=pre_loop,
                          train_op=pre_train_op,
                          inputs=[input_x, input_u],
                          data_flow=train_flow,
                          feed_dict={learning_rate: pre_lr_value, is_training: True},
                          metrics={'pretrain_loss': pretrain_loss, 'pretrain_train_recons': pretrain_train_recons},
                          summaries=summary_op)

    trainer = Trainer(loop=loop,
                      train_op=train_op,
                      inputs=[input_x, input_u],
                      data_flow=train_flow,
                      feed_dict={learning_rate: lr_value, is_training: True},
                      metrics=train_metrics,
                      summaries=summary_op)

    pre_validator = Evaluator(loop=pre_loop,
                              metrics={'pretrain_valid_loss': pretrain_valid_loss, 'pretrain_valid_recons': pretrain_valid_recons},
                              inputs=[input_x, input_u],
                              data_flow=valid_flow,
                              time_metric_name='pre_valid_time')

    pre_validator.events.on(
        spt.EventKeys.AFTER_EXECUTION,
        lambda e: exp.update_results(pre_validator.last_metrics_dict)
    )

    validator = Evaluator(loop=loop,
                          metrics=valid_metrics,
                          inputs=[input_x, input_u],
                          data_flow=valid_flow,
                          time_metric_name='valid_time')

    validator.events.on(
        spt.EventKeys.AFTER_EXECUTION,
        lambda e: exp.update_results(validator.last_metrics_dict)
    )

    train_losses = []
    tmp_collector = []
    valid_losses = []

    def on_metrics_collected(loop: TrainLoop, metrics):
        if 'loss' in metrics:
            tmp_collector.append(metrics['loss'])
        if loop.epoch % 1 == 0:
            if 'valid_loss' in metrics:
                valid_losses.append(metrics['valid_loss'])
                train_losses.append(np.mean(tmp_collector))
                tmp_collector.clear()

    loop.events.on(spt.EventKeys.METRICS_COLLECTED, on_metrics_collected)

    pre_trainer.evaluate_after_epochs(pre_validator, freq=1)
    pre_trainer.log_after_epochs(freq=1)

    trainer.evaluate_after_epochs(validator, freq=1)
    trainer.log_after_epochs(freq=1)

    with spt.utils.create_session().as_default() as session:

        session.run(var_initializer)

        with pre_loop:
            pre_trainer.run()

        print('')
        print('PreTraining Finished.')

        if config.save_results:
            saver = tf.train.Saver(var_list=pre_variables_to_save)
            saver.save(session, os.path.join(exp.abspath('result_params'), 'restored_pretrain_params.dat'))

        print('')
        print('Pretrain Model saved.')

        print('************Start train the whole network***********')

        with loop:
            trainer.run()

        print('')
        print('Training Finished.')

        if config.save_results:
            saver = tf.train.Saver(var_list=variables_to_save)
            saver.save(session, os.path.join(exp.abspath('result_params'), "restored_params.dat"))

        print('')
        print('Model saved.')


if __name__ == '__main__':
    with mltk.Experiment(ExpConfig()) as exp:
        exp.save_config()
        main(exp, exp.config)
        if exp.config.exp_dir_save_path is not None:
            with open(exp.config.exp_dir_save_path, 'a') as f:
                f.write("'" + exp.config.dataset + ' ' + exp.output_dir + "'" + '\n')
