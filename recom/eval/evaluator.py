
def generate_testing_candidates(rating_train, n_item):
    items = list(range(n_item))

    testing_cand = {
        u:list(set(items)-set(rating_train[u].keys()))
        for u in rating_train
    }
    return testing_cand


def predict_recommendation(model, test_cands):
    from torch import argsort

    predict = {}
    for user in test_cands:
        items = test_cands[user]
        # dot product as estimation of rating
        pred = model.pred_score(user, items)
        pred_ix = argsort(-pred)
        predict[user] = [items[ix] for ix in pred_ix]

    return predict