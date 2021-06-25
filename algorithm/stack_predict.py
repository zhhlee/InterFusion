import mltk
import os

from explib.eval_methods import get_best_f1, get_adjusted_composite_metrics
from algorithm.utils import GraphNodes, get_data, time_generator, get_sliding_window_data_flow, get_score, \
    get_avg_recons
import tfsnippet as spt
import tensorflow as tf
from tqdm import tqdm
from algorithm.InterFusion import MTSAD
from algorithm.InterFusion_swat import MTSAD_SWAT
import numpy as np
from typing import Optional
import pickle
from algorithm.mcmc_recons import mcmc_reconstruct, masked_reconstruct
from algorithm.cal_IPS import cal_IPS

__all__ = ['PredictConfig', 'final_testing', 'build_test_graph']


class PredictConfig(mltk.Config):
    load_model_dir: Optional[str]

    # evaluation params
    test_n_z = 100
    test_batch_size = 50
    test_start = 0
    max_test_size = None  # `None` means full test set

    save_results = True

    output_dirs = 'analysis_results'
    train_score_filename = 'train_score.pkl'
    test_score_filename = 'test_score.pkl'
    preserve_feature_dim = False  # whether to preserve the feature dim in score. If `True`, the score will be a 2-dim ndarray
    anomaly_score_calculate_latency = 1   # How many scores are averaged for the final score at a timestamp. `1` means use last point in each sliding window only.
    plot_recons_results = True

    use_mcmc = True               # use mcmc on the last point for anomaly detection
    mcmc_iter = 10
    mcmc_rand_mask = False
    n_mc_chain: int = 10
    pos_mask = True
    mcmc_track = True             # use mcmc tracker for anomaly interpretation and calculate IPS.


def build_test_graph(chain: spt.VariationalChain, input_x, origin_chain: spt.VariationalChain=None) -> GraphNodes:
    test_recons = tf.reduce_mean(chain.model['x'].log_prob(), axis=0)

    logpx = chain.model['x'].log_prob()
    logpz = chain.model['z2'].log_prob() + chain.model['z1'].log_prob()
    logqz_x = chain.variational['z1'].log_prob() + chain.variational['z2'].log_prob()
    test_lb = tf.reduce_mean(logpx + logpz - logqz_x, axis=0)

    log_joint = logpx + logpz
    latent_log_prob = logqz_x
    test_ll = spt.importance_sampling_log_likelihood(log_joint=log_joint, latent_log_prob=latent_log_prob, axis=0)
    test_nll = -test_ll

    # average over sample dim
    if origin_chain is not None:
        full_recons_prob = tf.reduce_mean(
            (chain.model['x'].distribution.base_distribution.log_prob(input_x) -
             origin_chain.model['x'].distribution.base_distribution.log_prob(input_x)),
            axis=0
        )
    else:
        full_recons_prob = tf.reduce_mean(chain.model['x'].distribution.base_distribution.log_prob(input_x), axis=0)

    if origin_chain is not None:
        origin_log_joint = origin_chain.model['x'].log_prob() + origin_chain.model['z1'].log_prob() + origin_chain.model['z2'].log_prob()
        origin_latent_log_prob = origin_chain.variational['z1'].log_prob() + origin_chain.variational['z2'].log_prob()
        origin_ll = spt.importance_sampling_log_likelihood(log_joint=origin_log_joint, latent_log_prob=origin_latent_log_prob, axis=0)
        test_ll_score = test_ll - origin_ll
    else:
        test_ll_score = test_ll

    outputs = {
        'test_nll': test_nll,
        'test_lb': test_lb,
        'test_recons': test_recons,
        'test_kl': test_recons - test_lb,
        'full_recons_prob': full_recons_prob,
        'test_ll': test_ll_score
    }

    return GraphNodes(outputs)


def build_recons_graph(chain: spt.VariationalChain, window_length, feature_dim, unified_x_std=False) -> GraphNodes:
    # average over sample dim
    recons_x = tf.reduce_mean(chain.model['x'].distribution.base_distribution.mean, axis=0)
    recons_x = spt.utils.InputSpec(shape=['?', window_length, feature_dim]).validate('recons', recons_x)
    if unified_x_std:
        recons_x_std = chain.model['x'].distribution.base_distribution.std
        recons_x_std = spt.ops.broadcast_to_shape(recons_x_std, tf.shape(recons_x))
    else:
        recons_x_std = tf.reduce_mean(chain.model['x'].distribution.base_distribution.std, axis=0)
    recons_x_std = spt.utils.InputSpec(shape=['?', window_length, feature_dim]).validate('recons_std', recons_x_std)
    return GraphNodes({'recons_x': recons_x, 'recons_x_std': recons_x_std})


def get_recons_results(recons_nodes: GraphNodes, input_x, input_u, data_flow: spt.DataFlow, total_batch_count, dataset,
                       mask=None, rand_x=None):
    data_flow = data_flow.threaded(5)
    recons_collector = []
    recons_std_collector = []
    session = spt.utils.get_default_session_or_error()
    with data_flow:
        for batch_x, batch_u in tqdm(data_flow, unit='step', total=total_batch_count, ascii=True):
            if mask is not None:
                batch_mask = np.zeros(shape=batch_x.shape)
                batch_mask[:, -1, :] = 1    # mask all dims of the last point in x
                if rand_x is not None:
                    batch_output = recons_nodes.eval(session,
                                                     feed_dict={input_x: batch_x, input_u: batch_u, mask: batch_mask,
                                                                rand_x: np.random.random(batch_x.shape)})
                else:
                    batch_output = recons_nodes.eval(session, feed_dict={input_x: batch_x, input_u: batch_u, mask: batch_mask})
            else:
                batch_output = recons_nodes.eval(session, feed_dict={input_x: batch_x, input_u: batch_u})
            for k, v in batch_output.items():
                if k == 'recons_x':
                    if dataset == 'SWaT' or dataset == 'WADI':
                        # idx = min(10, v.shape[1])
                        recons_collector.append(v[:, -10:, :])
                    else:
                        recons_collector.append(v)
                elif k == 'recons_x_std':
                    if dataset == 'SWaT' or dataset == 'WADI':
                        # idx = min(10, v.shape[1])
                        recons_std_collector.append(v[:, -10:, :])
                    else:
                        recons_std_collector.append(v)

    all_recons = np.concatenate(recons_collector, axis=0)   # (data_length - window_length + 1, window_length, x_dim)
    print(all_recons.shape)
    all_recons_std = np.concatenate(recons_std_collector, axis=0)
    return all_recons, all_recons_std


def final_testing(test_metrics: GraphNodes, input_x, input_u,
                  data_flow: spt.DataFlow, total_batch_count, y_test=None, mask=None, rand_x=None):
    data_flow = data_flow.threaded(5)
    full_recons_collector = []
    ll_collector = []
    epoch_out = {}
    stats = {}
    session = spt.utils.get_default_session_or_error()
    with data_flow:
        for batch_x, batch_u in tqdm(data_flow, unit='step', total=total_batch_count, ascii=True):
            if mask is not None:
                batch_mask = np.zeros(shape=batch_x.shape)
                batch_mask[:, -1, :] = 1  # mask all dims of the last point in x
                if rand_x is not None:
                    batch_output = test_metrics.eval(session, feed_dict={input_x: batch_x, input_u: batch_u, mask: batch_mask,
                                                                         rand_x: np.random.random(batch_x.shape)})
                else:
                    batch_output = test_metrics.eval(session,
                                                     feed_dict={input_x: batch_x, input_u: batch_u, mask: batch_mask})
            else:
                batch_output = test_metrics.eval(session, feed_dict={input_x: batch_x, input_u: batch_u})
            for k, v in batch_output.items():

                if k == 'full_recons_prob':
                    full_recons_collector.append(v)
                elif k == 'test_ll':
                    ll_collector.append(v)
                    if k not in epoch_out:
                        epoch_out[k] = []
                    epoch_out[k].append(v)
                else:
                    if k not in epoch_out:
                        epoch_out[k] = []
                    epoch_out[k].append(v)

    # save the results of this epoch, and compute epoch stats. Take average over both batch and window_length dim.
    for k, v in epoch_out.items():
        epoch_out[k] = np.concatenate(epoch_out[k], axis=0)
        if k not in stats:
            stats[k] = []
        stats[k].append(float(np.mean(epoch_out[k])))

    # collect full recons prob for calculate anomaly score
    full_recons_probs = np.concatenate(full_recons_collector, axis=0)   # (data_length-window_length+1, window_length, x_dim)
    ll = np.concatenate(ll_collector, axis=0)

    if y_test is not None:
        assert full_recons_probs.shape[0] + full_recons_probs.shape[1] - 1 == len(y_test)
        tmp1 = []
        for i in range(full_recons_probs.shape[0]):
            if y_test[i + full_recons_probs.shape[1] - 1] < 0.5:
                tmp1.append(np.sum(full_recons_probs[i, -1], axis=-1))  # normal point recons score
        stats['normal_point_test_recons'] = [float(np.mean(tmp1))]

    # calculate average statistics
    for k, v in stats.items():
        stats[k] = float(np.mean(v))

    return stats, full_recons_probs, ll


def mcmc_tracker(flow: spt.DataFlow, baseline, model, input_x, input_u, mask, max_iter, total_window_num,
                 window_length, x_dim, mask_last=False, pos_mask=False, use_rand_mask=False, n_mc_chain=1):
    # the baseline is the avg total score in a window on training set.
    session = spt.utils.get_default_session_or_error()
    last_x = tf.placeholder(dtype=tf.float32, shape=[None, window_length, x_dim], name='last_x')

    x_r = masked_reconstruct(model.reconstruct, last_x, input_u, mask)
    score, recons_mean, recons_std = model.get_score(x_embed=x_r, x_eval=input_x, u=input_u)
    tot_score = tf.reduce_sum(tf.multiply(score, tf.cast((1-mask), score.dtype)))

    def avg_multi_chain(x, n_chain):
        shape = (-1,) + (n_chain,) + x.shape[1:]
        return np.mean(x.reshape(shape), axis=1)

    res = {}
    with flow.threaded(5) as flow:
        for batch_x, batch_u, batch_score, batch_ori_recons, batch_ori_std, batch_idx \
                in tqdm(flow, unit='step', total=total_window_num, ascii=True):
            batch_idx = batch_idx[0]
            res[batch_idx] = {'x': [batch_x], 'recons': [batch_ori_recons], 'std': [batch_ori_std], 'score': [batch_score],
                              'K': [0], 'iter': [-1], 'mask': [np.zeros(shape=batch_x.shape)],
                              'total_score': [np.mean(np.sum(batch_score, axis=-1))]}
            best_score = batch_score
            best_total_score = np.mean(np.sum(batch_score, axis=-1))
            best_K = 0
            if pos_mask:
                pos_scores = np.mean(batch_score, axis=0)   # (window, x_dim)
                sorted_pos_idx = np.argsort(pos_scores, axis=None)
                potential_dim_num = np.sum((pos_scores < (baseline/(x_dim*window_length))).astype(np.int32))
            else:
                dim_scores = np.mean(batch_score, axis=(-2,-3))     # (x_dim, )
                sorted_dim_idx = np.argsort(dim_scores)
                potential_dim_num = np.sum((dim_scores < (baseline/(x_dim*window_length))).astype(np.int32))    # num of dims whose avg score < baseline

            if potential_dim_num > 0:
                K_init = max(potential_dim_num//5, 1)
                K_inc = max(potential_dim_num//10, 1)
            else:
                res[batch_idx]['best_score'] = best_score
                res[batch_idx]['best_total_score'] = best_total_score
                res[batch_idx]['best_K'] = best_K
                continue
            if use_rand_mask:
                rand_x = np.random.random(size=batch_x.shape)
            if pos_mask:
                max_K = x_dim * window_length
            else:
                max_K = x_dim
            for K in range(K_init, min(potential_dim_num+1, max_K), K_inc):
                if pos_mask:
                    mask_idx = sorted_pos_idx[:K]
                    batch_mask = np.zeros(shape=batch_x.shape)
                    batch_mask = batch_mask.reshape([batch_x.shape[0], -1])
                    batch_mask[:, mask_idx] = 1
                    batch_mask = batch_mask.reshape(batch_x.shape)
                else:
                    mask_idx = sorted_dim_idx[:K]
                    batch_mask = np.zeros(shape=batch_x.shape)
                    batch_mask[:, :, mask_idx] = 1
                if mask_last:
                    batch_mask[:, -1, :] = 1

                batch_last_x = batch_x
                if use_rand_mask:
                    batch_last_x = np.where(batch_mask.astype(np.bool), rand_x, batch_last_x)
                if n_mc_chain > 1:
                    init_x = np.repeat(batch_x, n_mc_chain, axis=0)
                    init_u = np.repeat(batch_u, n_mc_chain, axis=0)
                    init_mask = np.repeat(batch_mask, n_mc_chain, axis=0)
                    init_last_x = np.repeat(batch_last_x, n_mc_chain, axis=0)
                for i in range(max_iter):
                    if n_mc_chain > 1:
                        x_mc, x_recons, x_std, x_score, x_tot_score = \
                            session.run([x_r, recons_mean, recons_std, score, tot_score],
                                        feed_dict={input_x: init_x, input_u: init_u, mask: init_mask,
                                                   last_x: init_last_x})
                        init_last_x = x_mc
                        x_mc = avg_multi_chain(x_mc, n_mc_chain)
                        x_recons = avg_multi_chain(x_recons, n_mc_chain)
                        x_std = avg_multi_chain(x_std, n_mc_chain)
                        x_score = avg_multi_chain(x_score, n_mc_chain)
                        x_tot_score = float(x_tot_score) / float(n_mc_chain)
                    else:
                        x_mc, x_recons, x_std, x_score, x_tot_score = \
                            session.run([x_r, recons_mean, recons_std, score, tot_score],
                                        feed_dict={input_x: batch_x, input_u: batch_u, mask: batch_mask, last_x: batch_last_x})
                        batch_last_x = x_mc
                    total_score = float(x_tot_score) / (window_length * x_dim - np.sum(batch_mask)) / batch_x.shape[0] * x_dim
                    res[batch_idx]['x'].append(x_mc)
                    res[batch_idx]['recons'].append(x_recons)
                    res[batch_idx]['std'].append(x_std)
                    res[batch_idx]['score'].append(x_score)
                    res[batch_idx]['K'].append(K)
                    res[batch_idx]['iter'].append(i)
                    res[batch_idx]['mask'].append(batch_mask)
                    res[batch_idx]['total_score'].append(total_score)

                last_score = res[batch_idx]['total_score'][-1]
                if last_score >= best_total_score:
                    best_total_score = last_score
                    best_score = res[batch_idx]['score'][-1]
                    best_K = res[batch_idx]['K'][-1]

                if best_total_score >= (baseline/window_length):
                    break
            res[batch_idx]['best_score'] = best_score
            res[batch_idx]['best_total_score'] = best_total_score
            res[batch_idx]['best_K'] = best_K
    return res


def log_mean_exp(x, axis, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=True)
    ret = x_max + np.log(np.mean(np.exp(x - x_max), axis=axis, keepdims=True))
    if not keepdims:
        ret = np.squeeze(ret, axis=axis)
    return ret


def log_sum_exp(x, axis, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=True)
    ret = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if not keepdims:
        ret = np.squeeze(ret, axis=axis)
    return ret


def main(exp: mltk.Experiment[PredictConfig], test_config: PredictConfig):
    if test_config.load_model_dir is None:
        raise ValueError('`--load_model_dir` is required.')

    exp_config_path = os.path.join(test_config.load_model_dir, 'config.json')
    from algorithm.stack_train import ExpConfig
    loader = mltk.ConfigLoader(ExpConfig())
    loader.load_file(exp_config_path)
    train_config = loader.get()

    print(mltk.format_key_values(train_config, title='Train configurations'))
    print('')
    print(mltk.format_key_values(test_config, title='Test configurations'))
    print('')

    # set TFSnippet settings
    spt.settings.enable_assertions = False
    spt.settings.check_numerics = train_config.check_numerics

    exp.make_dirs(test_config.output_dirs)

    # prepare the data
    # simple data
    (x_train, _), (x_test, y_test) = \
        get_data(train_config.dataset, train_config.train.max_train_size, train_config.test.max_test_size,
                 train_start=train_config.train.train_start, test_start=train_config.test.test_start,
                 valid_portion=train_config.train.valid_portion)

    if train_config.use_time_info:
        u_train = np.asarray([time_generator(_i) for _i in range(len(x_train))])  # (train_size, u_dim)
        u_test = np.asarray([time_generator(len(x_train) + _i) for _i in range(len(x_test))])  # (test_size, u_dim)
    else:
        u_train = np.zeros([len(x_train), train_config.model.u_dim])  # (train_size, u_dim)
        u_test = np.zeros([len(x_test), train_config.model.u_dim])

    # prepare data_flow
    test_flow = get_sliding_window_data_flow(window_size=train_config.model.window_length, batch_size=test_config.test_batch_size,
                                             x=x_test, u=u_test, shuffle=False, skip_incomplete=False)
    evaluate_score_train_flow = get_sliding_window_data_flow(window_size=train_config.model.window_length,
                                                             batch_size=test_config.test_batch_size,
                                                             x=x_train, u=u_train, shuffle=False,
                                                             skip_incomplete=False)

    # build computation graph
    if train_config.dataset == 'SWaT' or train_config.dataset == 'WADI':
        model = MTSAD_SWAT(train_config.model, scope='model')
    else:
        model = MTSAD(train_config.model, scope='model')

    # input placeholders
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, train_config.model.window_length, train_config.model.x_dim], name='input_x')
    input_u = tf.placeholder(dtype=tf.float32, shape=[None, train_config.model.window_length, train_config.model.u_dim], name='input_u')
    mask = tf.placeholder(dtype=tf.int32, shape=[None, train_config.model.window_length, train_config.model.x_dim], name='mask')
    rand_x = tf.placeholder(dtype=tf.float32, shape=[None, train_config.model.window_length, train_config.model.x_dim], name='rand_x')

    tmp_out = None
    if test_config.use_mcmc:
        with tf.name_scope('mcmc_init'):
            tmp_qnet = model.q_net(input_x, u=input_u, n_z=test_config.test_n_z)
            tmp_chain = tmp_qnet.chain(model.p_net, observed={'x': input_x}, latent_axis=0, u=input_u)
            tmp_out = tf.reduce_mean(tmp_chain.vi.lower_bound.elbo())

    # derive testing nodes
    with tf.name_scope('testing'):

        if test_config.use_mcmc:
            if test_config.mcmc_rand_mask:      # use random value to mask the initial input for mcmc (otherwise use the original one)
                if test_config.n_mc_chain > 1:  # average the results of multi-mcmc chain for each input x.
                    init_x = tf.where(tf.cast(mask, dtype=tf.bool), rand_x, input_x)
                    init_x, s1, s2 = spt.ops.flatten_to_ndims(tf.tile(tf.expand_dims(init_x, 1), [1, test_config.n_mc_chain, 1, 1]), 3)
                    init_u, _, _ = spt.ops.flatten_to_ndims(tf.tile(tf.expand_dims(input_u, 1), [1, test_config.n_mc_chain, 1, 1]), 3)
                    init_mask, _, _ = spt.ops.flatten_to_ndims(tf.tile(tf.expand_dims(mask, 1), [1, test_config.n_mc_chain, 1, 1]), 3)
                    x_mcmc = mcmc_reconstruct(model.reconstruct, init_x, init_u, init_mask, test_config.mcmc_iter, back_prop=False)
                    x_mcmc = spt.ops.unflatten_from_ndims(x_mcmc, s1, s2)
                    x_mcmc = tf.reduce_mean(x_mcmc, axis=1)
                else:
                    init_x = tf.where(tf.cast(mask, dtype=tf.bool), rand_x, input_x)
                    x_mcmc = mcmc_reconstruct(model.reconstruct, init_x, input_u, mask, test_config.mcmc_iter, back_prop=False)
            else:
                if test_config.n_mc_chain > 1:
                    init_x, s1, s2 = spt.ops.flatten_to_ndims(tf.tile(tf.expand_dims(input_x, 1), [1, test_config.n_mc_chain, 1, 1]), 3)
                    init_u, _, _ = spt.ops.flatten_to_ndims(tf.tile(tf.expand_dims(input_u, 1), [1, test_config.n_mc_chain, 1, 1]), 3)
                    init_mask, _, _ = spt.ops.flatten_to_ndims(tf.tile(tf.expand_dims(mask, 1), [1, test_config.n_mc_chain, 1, 1]), 3)
                    x_mcmc = mcmc_reconstruct(model.reconstruct, init_x, init_u, init_mask, test_config.mcmc_iter, back_prop=False)
                    x_mcmc = spt.ops.unflatten_from_ndims(x_mcmc, s1, s2)
                    x_mcmc = tf.reduce_mean(x_mcmc, axis=1)
                else:
                    x_mcmc = mcmc_reconstruct(model.reconstruct, input_x, input_u, mask, test_config.mcmc_iter, back_prop=False)
        else:
            x_mcmc = input_x

        test_q_net = model.q_net(x_mcmc, u=input_u, n_z=test_config.test_n_z)
        test_chain = test_q_net.chain(model.p_net, observed={'x': input_x}, latent_axis=0, u=input_u)

        test_metrics = build_test_graph(test_chain, input_x)

        if test_config.plot_recons_results:
            recons_nodes = build_recons_graph(test_chain, train_config.model.window_length, train_config.model.x_dim, train_config.model.unified_px_logstd)

    # obtain params to restore
    variables_to_restore = tf.global_variables()

    restore_path = os.path.join(test_config.load_model_dir, 'result_params/restored_params.dat')

    # obtain the variables initializer
    var_initializer = tf.variables_initializer(tf.global_variables())

    test_flow = test_flow.threaded(5)
    evaluate_score_train_flow = evaluate_score_train_flow.threaded(5)

    with spt.utils.create_session().as_default() as session:

        session.run(var_initializer)

        saver = tf.train.Saver(var_list=variables_to_restore)
        saver.restore(session, restore_path)

        print('Model params restored.')

        # Evaluate the whole network
        if test_config.use_mcmc:
            for batch_x, batch_u in test_flow:
                _ = session.run(tmp_out, feed_dict={input_x: batch_x, input_u: batch_u})
                break

        # do evaluation
        print('')
        print('*************Evaluate score on testing set************')

        test_batch_count = (len(x_test) - train_config.model.window_length + test_config.test_batch_size) // test_config.test_batch_size

        test_stats, test_full_recons_probs, test_ll = final_testing(test_metrics,
                                                                    input_x,
                                                                    input_u,
                                                                    test_flow,
                                                                    test_batch_count,
                                                                    y_test,
                                                                    mask=mask if test_config.use_mcmc else None,
                                                                    rand_x=rand_x if test_config.mcmc_rand_mask else None)

        print('')
        print(mltk.format_key_values(test_stats, 'Final testing statistics'))
        exp.update_results(test_stats)

        test_score = get_score(test_full_recons_probs, preserve_feature_dim=test_config.preserve_feature_dim,
                               score_avg_window_size=test_config.anomaly_score_calculate_latency)

        # evaluate score on train set
        print('')
        print('*************Evaluate score on training set************')

        if train_config.dataset != 'WADI':
            train_set_batch_count = (len(x_train) - train_config.model.window_length + test_config.test_batch_size) // test_config.test_batch_size

            train_stats, train_full_recons_probs, _ = final_testing(test_metrics,
                                                                    input_x,
                                                                    input_u,
                                                                    evaluate_score_train_flow,
                                                                    train_set_batch_count,
                                                                    mask=mask if test_config.use_mcmc else None,
                                                                    rand_x=rand_x if test_config.mcmc_rand_mask else None)

            print('')
            print(mltk.format_key_values(train_stats, 'Training set evaluation statistics'))

            train_score = get_score(train_full_recons_probs, preserve_feature_dim=test_config.preserve_feature_dim,
                                    score_avg_window_size=test_config.anomaly_score_calculate_latency)

            print('train_score shape: ', train_score.shape)

            if test_config.save_results:
                np.savez(os.path.join((exp.abspath(test_config.output_dirs)), 'full_recons_window_probs.npz'),
                         full_full_test_recons_window_probs=test_full_recons_probs,
                         full_train_recons_window_probs=get_score(train_full_recons_probs, preserve_feature_dim=True,
                                       score_avg_window_size=test_config.anomaly_score_calculate_latency),
                         full_test_recons_window_probs=get_score(test_full_recons_probs, preserve_feature_dim=True,
                                       score_avg_window_size=test_config.anomaly_score_calculate_latency))

            if not test_config.mcmc_track:
                del test_full_recons_probs

            del train_full_recons_probs

        # for reconstruct plotting
        import matplotlib.pyplot as plt
        exp.make_dirs('figures')
        if test_config.plot_recons_results:
            print('')
            print('*************Calculating and plotting reconstruction data************')
            all_test_recons, all_test_recons_std = get_recons_results(recons_nodes,
                                                                      input_x,
                                                                      input_u,
                                                                      test_flow,
                                                                      test_batch_count,
                                                                      train_config.dataset,
                                                                      mask=mask if test_config.use_mcmc else None,
                                                                      rand_x=rand_x if test_config.mcmc_rand_mask else None)
            all_train_recons, all_train_recons_std = get_recons_results(recons_nodes,
                                                                        input_x,
                                                                        input_u,
                                                                        evaluate_score_train_flow,
                                                                        train_set_batch_count,
                                                                        train_config.dataset,
                                                                        mask=mask if test_config.use_mcmc else None,
                                                                        rand_x=rand_x if test_config.mcmc_rand_mask else None)
            final_test_recons = get_avg_recons(all_test_recons, window_length=train_config.model.window_length, recons_avg_window_size=test_config.anomaly_score_calculate_latency)
            final_train_recons = get_avg_recons(all_train_recons, window_length=train_config.model.window_length, recons_avg_window_size=test_config.anomaly_score_calculate_latency)
            if not test_config.mcmc_track:
                del all_test_recons
            del all_train_recons
            if test_config.anomaly_score_calculate_latency == 1:
                final_test_recons_std = get_avg_recons(all_test_recons_std, window_length=train_config.model.window_length)
                final_train_recons_std = get_avg_recons(all_train_recons_std, window_length=train_config.model.window_length)
                np.savez(os.path.join(exp.abspath(test_config.output_dirs), 'recons_plotting_data.npz'),
                         x_train=x_train, x_test=x_test, x_train_recons=final_train_recons,
                         x_train_recons_std=final_train_recons_std, x_test_recons=final_test_recons,
                         x_test_recons_std=final_test_recons_std, y_test=y_test)
            else:       # average x_recons_std is meaningless if calculate latency > 1
                np.savez(os.path.join(exp.abspath(test_config.output_dirs), 'recons_plotting_data.npz'),
                     x_train=x_train, x_test=x_test, x_train_recons=final_train_recons, x_test_recons=final_test_recons, y_test=y_test)
            if not test_config.mcmc_track:
                del all_test_recons_std
            del all_train_recons_std

        if test_config.train_score_filename is not None and train_config.dataset != 'WADI':
            with open(os.path.join(exp.abspath(test_config.output_dirs), test_config.train_score_filename), 'wb') as file:
                pickle.dump(train_score, file)

        if test_config.test_score_filename is not None:
            with open(os.path.join(exp.abspath(test_config.output_dirs), test_config.test_score_filename), 'wb') as file:
                pickle.dump(test_score, file)

        print('')
        print('*************Calculating best F1-score*****************')

        y_test = y_test[-len(test_score):]

        # get best f1
        t, th = get_best_f1(test_score, y_test)

        best_thresh = th

        # output the results
        exp.update_results({
            'best-f1': t[0],
            'precision': t[1],
            'recall': t[2],
            'TP': t[3],
            'TN': t[4],
            'FP': t[5],
            'FN': t[6],
            'threshold': th
        })

        auroc, ap, _, _, _, _, _ = get_adjusted_composite_metrics(test_score, y_test)
        exp.update_results({
            'auroc': auroc,
            'ap': ap
        })

        if test_config.mcmc_track:
            # find the TP points that need to be interpreted according to best_thresh
            best_idx = np.logical_and(test_score <= best_thresh, y_test > 0.5)
            best_idx = np.where(best_idx)[0]
            total_x, total_u, total_score, total_recons, total_std, total_idx = [], [], [], [], [], []
            for i in best_idx:
                total_x.append(x_test[i:i+train_config.model.window_length, :])
                total_u.append(u_test[i:i+train_config.model.window_length, :])
                total_score.append(test_full_recons_probs[i])
                total_recons.append(all_test_recons[i])
                total_std.append(all_test_recons_std[i])
                total_idx.append(i + train_config.model.window_length - 1)
            total_x = np.stack(total_x, axis=0)
            total_u = np.stack(total_u, axis=0)
            total_score = np.stack(total_score, axis=0)
            total_recons = np.stack(total_recons, axis=0)
            total_std = np.stack(total_std, axis=0)
            total_idx = np.stack(total_idx, axis=0)
            best_flow = spt.DataFlow.arrays([total_x, total_u, total_score, total_recons, total_std, total_idx],
                                            batch_size=1, shuffle=False, skip_incomplete=False)

            best_baseline = train_stats['test_recons']
            best_window_num = total_x.shape[0]

            res = mcmc_tracker(best_flow, best_baseline, model, input_x, input_u, mask, test_config.mcmc_iter,
                               best_window_num, train_config.model.window_length, train_config.model.x_dim,
                               test_config.use_mcmc, test_config.pos_mask,
                               test_config.mcmc_rand_mask, test_config.n_mc_chain)

            with open(os.path.join(exp.abspath(test_config.output_dirs), 'mcmc_tracker.pkl'), 'wb') as file:
                pickle.dump(res, file)

            del res

            res = cal_IPS(path=exp.abspath(test_config.output_dirs), dataset=train_config.dataset,
                                    mcmc=True,
                                    is_pretrain=False)
            exp.update_results({'IPS': res['mcmc_p=100: wd_last_itv_min_weight'],
                                'IPS@150%': res['mcmc_p=150: wd_last_itv_min_weight']})

        print('')
        print(mltk.format_key_values(exp.results), 'Results')


if __name__ == '__main__':
    with mltk.Experiment(PredictConfig()) as exp:
        main(exp, exp.config)
