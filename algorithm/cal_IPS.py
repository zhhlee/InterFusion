import os
import pickle
import numpy as np


def cal_IPS(path, dataset, mcmc, is_pretrain):
    path += '/'
    # load labels
    f = open("./data/processed/" + dataset + "_test_label.pkl", "rb")
    test_label = pickle.load(f).reshape((-1))
    f.close()
    label_dir = './data/interpretation_label/' + dataset + '.txt'
    try:
        labels = str(open(label_dir, 'rb').read(), encoding='utf8')
    except:
        raise FileNotFoundError('cannot find label via path: ' + label_dir)
    # parse labels
    labels = labels.split('\n')
    if len(labels[-1]) == 0:
        labels = labels[:-1]
    intervals = []
    dims = []
    for i in labels:
        t = i.split(':')
        assert len(t) == 2
        intervals.append([int(_) for _ in t[0].split('-')])
        assert len(intervals[-1]) == 2
        dims.append([int(_) for _ in t[1].split(',')])

    assert len(intervals) == len(dims)
    interpret_dict = {}
    tp_res = {}

    # form the interpret dict: {(idx_st, idx_ed): interpret_dims}
    for _ in range(len(intervals)):
        interpret_dict[tuple(intervals[_])] = dims[_]

    if mcmc:
        assert is_pretrain is False
        try:
            tracker = pickle.load(open(path + 'mcmc_tracker.pkl', 'rb'))
        except:
            raise FileNotFoundError('cannot find mcmc_tracker.pkl')
        # preprocess tracker to get:
        # {idx_ed: best_score_mcmc (np.array(shape=[window_size, x_dim]))}
        for ed_idx in tracker:
            shape = tracker[ed_idx]['best_score'].shape
            assert len(shape) == 3
            tp_res[ed_idx] = tracker[ed_idx]['best_score'].reshape([shape[1], shape[2]])
    else:
        try:
            if is_pretrain:
                full_recons = np.load(path + 'pretrain_full_recons_window_probs.npz')
            else:
                full_recons = np.load(path + 'full_recons_window_probs.npz')
        except:
            raise FileNotFoundError('cannot find full_recons_window_probs.npz!')
        try:
            if is_pretrain:
                test_score = pickle.load(open(path + 'pretrain_test_score.pkl', 'rb'))
            else:
                test_score = pickle.load(open(path + 'test_score.pkl', 'rb'))
        except:
            raise FileNotFoundError('cannot find test_score.pkl!')
        # get all tp points idx
        full_recons = full_recons['full_full_test_recons_window_probs']
        assert full_recons.shape[0] == len(test_label) - 100 + 1

        test_label = test_label[-len(test_score):]
        assert len(test_score) == len(test_label)

        t, th = get_best_f1(test_score, test_label)
        tp_idx = np.logical_and(test_label > 0.5, test_score <= th)
        tp_idx = np.where(tp_idx)[0]
        tp_idx += (100 - 1)
        # get the windows which end by tp_idx
        for ed_idx in tp_idx:
            tp_res[ed_idx] = full_recons[ed_idx - 100 + 1]
            assert tp_res[ed_idx].shape[0] == 100

    results = {}

    for p in [100, 150]:
        prefix = 'p=' + str(p) + ': '
        # segment-wise aggregation
        within_window_funcs = [lambda x: x[-1]]
        within_window_func_names = ['wd_last']
        assert len(within_window_func_names) == len(within_window_funcs)

        def min_aggr_and_keep(x):
            temp = np.min(x, axis=0)
            return [temp for _ in range(len(x))]

        def ave_aggr_and_keep(x):
            temp = np.mean(x, axis=0)
            return [temp for _ in range(len(x))]

        def max_aggr_and_keep(x):
            temp = np.max(x, axis=0)
            return [temp for _ in range(len(x))]

        def min_aggr(x):
            return [np.min(x, axis=0)]

        def ave_aggr(x):
            return [np.mean(x, axis=0)]

        def max_aggr(x):
            return [np.max(x, axis=0)]

        within_interval_sc_funcs = [min_aggr_and_keep]
        within_interval_sc_func_names = ['itv_min_weight']

        for wd_idx, window_func in enumerate(within_window_funcs):
            for iv_idx, itv_func in enumerate(within_interval_sc_funcs):
                combine_name = prefix + within_window_func_names[wd_idx] + '_' + within_interval_sc_func_names[iv_idx]
                # compute aggr score
                itv = {}
                for ed_idx in tp_res:
                    for interval in interpret_dict:
                        if interval[0] <= ed_idx <= interval[1]:
                            # this TP in this interval
                            dim_scores = window_func(tp_res[ed_idx]).reshape((-1))
                            if interval in itv:
                                itv[interval].append(dim_scores)
                            else:
                                itv[interval] = [dim_scores]
                            break
                scores = []
                labels = []
                for interval in itv:
                    temp = itv_func(itv[interval])
                    scores += temp
                    labels += [interpret_dict[interval] for _ in range(len(temp))]
                assert len(scores) == len(labels)
                # compute Interpretation score
                hit_rate_collector = []
                for idx, dim_scores in enumerate(scores):
                    dim_order = np.argsort(dim_scores) + 1
                    hit_rate = get_hit_rate(pred=dim_order, label=labels[idx], p=p)
                    hit_rate_collector.append(hit_rate)
                hit_rate = np.mean(hit_rate_collector)
                results[combine_name] = hit_rate

    if is_pretrain:
        res = {}
        for _ in results:
            res['pretrain_' + _] = results[_]
        results = res
    if mcmc:
        res = {}
        for _ in results:
            res['mcmc_' + _] = results[_]
        results = res
    return results


def get_hit_rate(pred, label, p):
    chance_num = min(int(p / 100 * len(label)), len(pred))
    cnt = 0
    for _ in range(chance_num):
        if pred[_] in label:
            cnt += 1
    hit_rate = cnt / len(label)
    return hit_rate


# here for our refined best-f1 search method
def get_best_f1(score, label):
    '''
    :param score: 1-D array, input score, tot_length
    :param label: 1-D array, standard label for anomaly
    :return: list for results, threshold
    '''

    assert score.shape == label.shape
    print('***computing best f1***')
    search_set = []
    tot_anomaly = 0
    for i in range(label.shape[0]):
        tot_anomaly += (label[i] > 0.5)
    flag = 0
    cur_anomaly_len = 0
    cur_min_anomaly_score = 1e5
    for i in range(label.shape[0]):
        if label[i] > 0.5:
            # here for an anomaly
            if flag == 1:
                cur_anomaly_len += 1
                cur_min_anomaly_score = score[i] if score[i] < cur_min_anomaly_score else cur_min_anomaly_score
            else:
                flag = 1
                cur_anomaly_len = 1
                cur_min_anomaly_score = score[i]
        else:
            # here for normal points
            if flag == 1:
                flag = 0
                search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
                search_set.append((score[i], 1, False))
            else:
                search_set.append((score[i], 1, False))
    if flag == 1:
        search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
    search_set.sort(key=lambda x: x[0])
    best_f1_res = - 1
    threshold = 1
    P = 0
    TP = 0
    best_P = 0
    best_TP = 0
    for i in range(len(search_set)):
        P += search_set[i][1]
        if search_set[i][2]:  # for an anomaly point
            TP += search_set[i][1]
        precision = TP / (P + 1e-5)
        recall = TP / (tot_anomaly + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        if f1 > best_f1_res:
            best_f1_res = f1
            threshold = search_set[i][0]
            best_P = P
            best_TP = TP

    print('***  best_f1  ***: ', best_f1_res)
    print('*** threshold ***: ', threshold)
    return (best_f1_res,
            best_TP / (best_P + 1e-5),
            best_TP / (tot_anomaly + 1e-5),
            best_TP,
            score.shape[0] - best_P - tot_anomaly + best_TP,
            best_P - best_TP,
            tot_anomaly - best_TP), threshold
