# test local

flatten_dict_score = lambda rat_dic: [item for user in rat_dic
                                      for item in rat_dic[user].values()]


def compute_rat_bias(rat_dict, user_size, item_size):
    user_bias = dict(zip(range(user_size)
                         , [0]*user_size))
    item_bias = dict(zip(range(item_size)
                         , [0]*item_size))
    item_rat_dict = dict(zip(range(item_size)
                             , [[] for _ in range(item_size)]))

    all_scores = flatten_dict_score(rat_dict)
    average_score = sum(all_scores)/len(all_scores)

    for user in rat_dict:
        N = len(rat_dict[user])
        for item in rat_dict[user]:
            # add item rating into
            item_rat_dict[item].append(rat_dict[user][item])
        # compute user's rating
        user_bias[user] = sum(rat_dict[user].values())/N - average_score

    for item in item_rat_dict: # compute item bias
        N = len(item_rat_dict[item])
        if N!=0:
            item_bias[item] = sum(item_rat_dict[item])/N - average_score

    return average_score, user_bias, item_bias


def rating_min_max_scalar(rat_dict, report_min_max=False):
    flatten_dict = flatten_dict_score(rat_dict)
    score_min = min(flatten_dict)
    score_max = max(flatten_dict)

    res = {}
    for user in rat_dict:
        res[user] = {}
        for item in rat_dict[user]:
            res[user][item] = (rat_dict[user][item]-score_min)/(score_max-score_min)

    if not report_min_max:
        return res
    else:
        return {'rating':res, 'min':score_min, 'max':score_max}

