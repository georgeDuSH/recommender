def compute_rat_bias(rat_dict, user_size, item_size):
    user_bias = dict(zip(range(user_size)
                         , [0]*user_size))
    item_bias = dict(zip(range(item_size)
                         , [0]*item_size))
    item_rat_dict = dict(zip(range(item_size)
                             , [[] for _ in range(item_size)]))

    for user in rat_dict:
        N = len(rat_dict[user])
        for item in rat_dict[user]:
            # add item rating into
            item_rat_dict[item].append(rat_dict[user][item])
        # compute user's rating
        user_bias[user] = sum(rat_dict[user].values())/N

    for item in item_rat_dict: # compute item bias
        N = len(item_rat_dict[item])
        if N!=0:
            item_bias[item] = sum(item_rat_dict[item])/N

    return user_bias, item_bias