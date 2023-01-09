"""
Metric: https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54
"""

def _f1(k, test, predict):
    """ Compute f1 score for each user

    :param k: int
    :param test: list of int
    :param predict: list of int
    :return tp_value: float
    """
    max_eval_size = min(k, len(predict)) # in case user has little candidate item
    tp = 0
    for i in range(max_eval_size):
        if predict[i] in test:
            tp += 1
    return 2*tp/(max_eval_size + len(test))


def f1(k, test_dict, pred_dict):
    res = []
    for user in test_dict:
        res.append(_f1(k, test_dict[user], pred_dict[user]))
    return sum(res)/len(test_dict)


def _hr(k, test, pred):
    if isinstance(pred, dict):
        pred = list(pred.keys())

    if k - len(set(pred[:k])-set(test)) > 0:
        return 1
    else:
        return 0


def hit_rate(k, test_dict, pred_dict):
    res = []
    for user in test_dict:
        res.append(_hr(k, list(test_dict[user]), pred_dict[user]))
    return sum(res)/len(test_dict)


def _map(k, test, pred):
    if isinstance(pred, dict):
        pred = list(pred.keys())

    if not k:
        return 0

    max_eval_size = min(k, len(pred))
    x, s = 0, 0
    for i in range(max_eval_size):
        if pred[i] in test:
            x += 1
            s += x/(i+1)

    return s/max_eval_size # average precision


def map(k, test_dict, pred_dict):
    res = []
    for user in test_dict:
        res.append(_map(k, list(test_dict[user]), pred_dict[user]))
    # mean of average precision
    return sum(res)/len(test_dict)


def _dcg(score):
    if len(score) == 0:
        return 0

    from math import log
    dcg = 0
    for ix, score_i in enumerate(score):
        if score_i != 0:
            dcg += score_i / log(ix+2, 2)
    return dcg


def _ndcg(k, test, pred, use_score):
    if isinstance(pred, dict) and use_score:
        score = [test[pred_i] if pred_i in test else 0
                 for pred_i in pred[:k]]
    else:
        score = [1 if pred_i in test else 0
                 for pred_i in pred[:k]]

    return _dcg(score)


def ndcg(k, test, predict, use_score=True):
    res = []
    for user in test:
        res.append(_ndcg(k, test[user], predict[user], use_score))
    return sum(res) / len(test)