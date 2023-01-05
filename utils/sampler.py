def pairwise_loader_even_prob(training_dict, item2ix, user_size, pos_size=1, neg_size=0, batch_size=128):
    from random import choices, choice
    from torch.utils.data import DataLoader

    all_item_set = set(item2ix.keys())

    train_data = []
    for _ in range(user_size):
        user = choice(list(training_dict.keys()))
        pos_cands = list(training_dict[user].keys())
        pos_items = choices(pos_cands, k=pos_size)
        # Using Negative Sampling
        if neg_size!=0:
            neg_cands = list(all_item_set-set(pos_items))
            neg_set_size = len(pos_items)*neg_size
            neg_items = choices(neg_cands, k=neg_set_size)
            pos_items *= neg_size
            target_user = [user] * neg_size * len(pos_items)
            train_data.extend(zip(target_user, pos_items, neg_items))
        # Not Using Negative Sampling
        else:
            target_user = [user] * len(pos_items)
            train_data.extend(zip(target_user, pos_items))

    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    return train_data_loader

def list_loader_even_prob(training_dict, item2ix, user_size, pos_size=1, neg_size=0, batch_size=128):
    """ sample data from the

    :param training_dict:
    :param items:
    :param user_size:
    :param pos_size:
    :param neg_size:
    :param batch_size:
    :return:
    """
    from random import choices, choice
    from torch import tensor
    from torch.utils.data import DataLoader

    all_item_set = set(item2ix.keys())

    train_data = []
    for _ in range(user_size):
        user = choice(list(training_dict.keys()))
        pos_cands = list(training_dict[user].keys())
        pos_items = choices(pos_cands, k=pos_size)
        # Using Negative Sampling
        if neg_size!=0:
            neg_cands = list(all_item_set-set(pos_items))
            neg_set_size = len(pos_items)*neg_size
            neg_items = choices(neg_cands, k=neg_set_size)
            neg_items = [neg_items[i:i+neg_size] for i in range(0,len(neg_items), neg_size)]
            target_user = [user] * len(pos_items)

            # return
            train_data.extend(zip(target_user, pos_items, tensor(neg_items)))
        # Not Using Negative Sampling
        else:
            target_user = [user] * len(pos_items)
            train_data.extend(zip(target_user, pos_items))

    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    return train_data_loader