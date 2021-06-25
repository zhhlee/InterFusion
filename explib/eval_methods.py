# -*- coding: utf-8 -*-
import numpy as np
import sklearn.metrics


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


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


# calculate evaluation metrics (best-F1, AUROC, AP) under point-adjust approach.
def get_adjusted_composite_metrics(score, label):
    score = -score  # change the recons prob to anomaly score, higher anomaly score means more anomalous
    # adjust the score for segment detection. i.e., for each ground-truth anomaly segment, use the maximum score
    # as the score of all points in that segment. This corresponds to point-adjust f1-score.
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])

    # now get the adjust score for segment evaluation.
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=label, y_score=score, drop_intermediate=False)
    auroc = sklearn.metrics.auc(fpr, tpr)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true=label, probas_pred=score)
    # validate best f1
    f1 = np.max(2 * precision * recall / (precision + recall + 1e-5))
    ap = sklearn.metrics.average_precision_score(y_true=label, y_score=score, average=None)
    return auroc, ap, f1, precision, recall, fpr, tpr
