import tfsnippet as spt
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from typing import *
import tensorflow as tf
from functools import partial

# here, use 'min_max' or 'mean_std' for different method
# method = 'min_max' or 'mean_std'
method = 'min_max'
alpha = 4.0  # mean +/- alpha * std


def get_sliding_window_data_flow(window_size, batch_size, x, u=None, y=None, shuffle=False, skip_incomplete=False) -> spt.DataFlow:
    n = len(x)
    seq = np.arange(window_size - 1, n, dtype=np.int32).reshape([-1, 1])
    seq_df: spt.DataFlow = spt.DataFlow.arrays(
        [seq], shuffle=shuffle, skip_incomplete=skip_incomplete, batch_size=batch_size)
    offset = np.arange(-window_size + 1, 1, dtype=np.int32)

    if y is not None:
        if u is not None:
            df = seq_df.map(lambda idx: (x[idx + offset], u[idx + offset], y[idx + offset]))
        else:
            df = seq_df.map(lambda idx: (x[idx + offset], y[idx + offset]))
    else:
        if u is not None:
            df = seq_df.map(lambda idx: (x[idx + offset], u[idx + offset]))
        else:
            df = seq_df.map(lambda idx: (x[idx + offset],))

    return df


def time_generator(timestamp):
    mins = 60
    hours = 24
    days = 7
    timestamp %= (mins * hours * days)
    res = np.zeros([mins + hours + days])
    res[int(timestamp / hours / mins)] = 1  # day
    res[days + int((timestamp % (mins * hours)) / mins)] = 1  # hours
    res[days + hours + int(timestamp % mins)] = 1  # min
    return res


def get_data_dim(dataset):
    if dataset == 'SWaT':
        return 51
    elif dataset == 'WADI':
        return 118
    elif str(dataset).startswith('machine'):
        return 38
    elif str(dataset).startswith('omi'):
        return 19
    else:
        raise ValueError('unknown dataset '+str(dataset))


def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
             test_start=0, valid_portion=0.3, prefix="./data/processed"):
    """
    get data from pkl files
    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
    train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None
    if do_preprocess:
        # train_data = preprocess(train_data)
        # test_data = preprocess(test_data)
        train_data, test_data = preprocess(train_data, test_data, valid_portion=valid_portion)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)


def preprocess(train, test, valid_portion=0):
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)

    if len(train.shape) == 1 or len(test.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(train)) != 0):
        print('Train data contains null values. Will be replaced with 0')
        train = np.nan_to_num(train)

    if np.any(sum(np.isnan(test)) != 0):
        print('Test data contains null values. Will be replaced with 0')
        test = np.nan_to_num(test)

    # revise here for other preprocess methods
    if method == 'min_max':
        if valid_portion > 0:
            split_idx = int(len(train) * valid_portion)
            train, valid = train[:-split_idx], train[-split_idx:]
            scaler = MinMaxScaler().fit(train)
            train = scaler.transform(train)
            valid = scaler.transform(valid)
            valid = np.clip(valid, a_min=-3.0, a_max=3.0)
            test = scaler.transform(test)
            test = np.clip(test, a_min=-3.0, a_max=3.0)
            train = np.concatenate([train, valid], axis=0)
            print('Data normalized with min-max scaler')
        else:
            scaler = MinMaxScaler().fit(train)
            train = scaler.transform(train)
            test = scaler.transform(test)
            test = np.clip(test, a_min=-3.0, a_max=3.0)
            print('Data normalized with min-max scaler')

    elif method == 'mean_std':

        def my_transform(value, ret_all=True, mean=None, std=None):
            if mean is None:
                mean = np.mean(value, axis=0)
            if std is None:
                std = np.std(value, axis=0)
            for i in range(value.shape[0]):
                clip_value = mean + alpha * std  # compute clip value: (mean - a * std, mean + a * std)
                temp = value[i] < clip_value
                value[i] = temp * value[i] + (1 - temp) * clip_value
                clip_value = mean - alpha * std
                temp = value[i] > clip_value
                value[i] = temp * value[i] + (1 - temp) * clip_value
                std = np.maximum(std, 1e-5)  # to avoid std -> 0
                value[i] = (value[i] - mean) / std  # normalization
            return value, mean, std if ret_all else value

        train, _mean, _std = my_transform(train)
        test = my_transform(test, False, _mean, _std)[0]
        print('Data normalized with standard scaler method')

    elif method == 'none':
        print('No pre-processing')

    else:
        raise RuntimeError('unknown preprocess method')

    return train, test


TensorLike = Union[tf.Tensor, spt.StochasticTensor]


class GraphNodes(Dict[str, TensorLike]):
    """A dict that maps name to TensorFlow graph nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k, v in self.items():
            if not spt.utils.is_tensor_object(v):
                raise TypeError(f'The value of `{k}` is not a tensor: {v!r}.')

    def eval(self,
             session: tf.Session = None,
             feed_dict: Dict[tf.Tensor, Any] = None) -> Dict[str, Any]:
        """
        Evaluate all the nodes with the specified `session`.
        Args:
            session: The TensorFlow session.
            feed_dict: The feed dict.
        Returns:
            The node evaluation outputs.
        """
        if session is None:
            session = spt.utils.get_default_session_or_error()

        keys = list(self)
        tensors = [self[key] for key in keys]
        outputs = session.run(tensors, feed_dict=feed_dict)

        return dict(zip(keys, outputs))

    def add_prefix(self, prefix: str) -> 'GraphNodes':
        """
        Add a common prefix to all metrics in this collection.
        Args:
             prefix: The common prefix.
        """
        return GraphNodes({f'{prefix}{k}': v for k, v in self.items()})


def get_score(recons_probs, preserve_feature_dim=False, score_avg_window_size=1):
    """
    Evaluate the anomaly score at each timestamp according to the reconstruction probability obtained by model.
    :param recons_probs: (data_length-window_length+1, window_length, x_dim). The reconstruction probabilities correspond
    to each timestamp and each dimension of x, evaluated in sliding windows with length 'window_length'. The larger the
    reconstruction probability, the less likely a point is an anomaly.
    :param preserve_feature_dim: bool. Whether sum over the feature dimension. If True, preserve the anomaly score on
    each feature dimension. If False, sum over the anomaly scores along feature dimension and return a single score on
    each timestamp.
    :param score_avg_window_size: int. How many scores in different sliding windows are used to evaluate the anomaly score
    at a given timestamp. By default score_avg_window_size=1, only the score of last point are used in each sliding window,
    and this score is directly used as the final anomaly score at this timestamp. When score_avg_window_size > 1, then
    the last 'score_avg_window_size' scores are used in each sliding window. Then for timestamp t, if t is the last point
    of sliding window k, then the anomaly score of t is now evaluated as the average score_{t} in sliding windows
    [k, k+1, ..., k+score_avg_window_size-1].
    :return: Anomaly scores (reconstruction probability) at each timestamps.
    With shape ``(data_length - window_size + score_avg_window_size,)`` if `preserve_feature_dim` is `False`,
    or ``(data_length - window_size + score_avg_window_size, x_dim)`` if `preserve_feature_dim` is `True`.
    The first `window_size - score_avg_window_size` points are discarded since there aren't enough previous values to evaluate the score.
    """
    data_length = recons_probs.shape[0] + recons_probs.shape[1] - 1
    window_length = recons_probs.shape[1]
    score_collector = [[] for i in range(data_length)]
    for i in range(recons_probs.shape[0]):
        for j in range(score_avg_window_size):
            score_collector[i + window_length - j - 1].append(recons_probs[i, -j-1])

    score_collector = score_collector[window_length-score_avg_window_size:]
    scores = []
    for i in range(len(score_collector)):
        scores.append(np.mean(score_collector[i], axis=0))
    scores = np.array(scores)                 # average over the score_avg_window. (data_length-window_length+score_avg_window_size, x_dim)
    if not preserve_feature_dim:
        scores = np.sum(scores, axis=-1)
    return scores


def get_avg_recons(recons_vals, window_length, recons_avg_window_size=1):
    """
    Get the averaged reconstruction values for plotting. The last `recons_avg_window_size` points in each reconstruct
    sliding windows are used, the final reconstruction values at each timestamp is the mean of each value at this timestamp.
    :param recons_vals: original reconstruction values. shape: (data_length - window_length + 1, window_length, x_dim)
    :param recons_avg_window_size:  int. How many points are used in each reconstruct sliding window.
    :return: final reconstruction curve. shape: (data_length, x_dim)
    The first `window_size - recons_avg_window_size` points use the reconstruction value of the first reconstruction window,
    others use the averaged values according to `recons_vals` and `recons_avg_window_size`.
    """
    data_length = recons_vals.shape[0] + window_length - 1
    recons_collector = [[] for i in range(data_length)]
    for i in range(recons_vals.shape[0]):
        for j in range(recons_avg_window_size):
            recons_collector[i + window_length - j - 1].append(recons_vals[i, -j-1, :])

    if recons_vals.shape[1] < window_length:
        for i in range(window_length - recons_avg_window_size):
            recons_collector[i] = [recons_vals[0, -1, :]]
    else:
        for i in range(window_length - recons_avg_window_size):
            recons_collector[i] = [recons_vals[0, i, :]]

    final_recons = []
    for i in range(len(recons_collector)):
        final_recons.append(np.mean(recons_collector[i], axis=0))
    final_recons = np.array(final_recons)    # average over the recons_avg_window. (data_length, x_dim)
    return final_recons
