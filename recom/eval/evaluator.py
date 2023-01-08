
def generate_testing_candidates(rating_train, items):
    items_set = set(list(items.keys()))
    testing_cand = {}

    for user in rating_train:
        testing_cand[user] = list(items_set-set(rating_train[user].keys()))

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