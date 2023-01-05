import torch
import torch.nn.functional as F
# local methods
from .sampler import list_loader_even_prob, pairwise_loader_even_prob

def MF_loss(preds):
    return -F.logsigmoid(preds).sum()

def train_model(model, ratings, items, n_epochs, opt_fn, user_size, pos_size, neg_size=0, negative_sampling=True, method='list', cuda=True):

    import time
    from torch import autograd, LongTensor
    import numpy as np

    assert method in ['list', 'pairwise'], 'Invalid method %s' % method
    loader = list_loader_even_prob if method == 'list' else pairwise_loader_even_prob

    device = torch.device('cuda' if cuda == True else 'cpu')

    if cuda:
        model.cuda()

    losses = []

    t0 = time.time()
    for epoch in range(n_epochs):
        train_data = loader(
            ratings
            , items
            , user_size=user_size
            , pos_size=pos_size
            , neg_size=neg_size)

        for i, batch in enumerate(train_data):
            if not negative_sampling:
                user, pos_item = batch
            else:
                user, pos_item, neg_item = batch

            model.zero_grad()

            # variablize
            user = autograd.Variable(LongTensor(user)).to(device)
            pos_item = autograd.Variable(LongTensor(pos_item)).to(device)
            if negative_sampling:
                neg_item = autograd.Variable(LongTensor(neg_item)).to(device)

            preds = model(user=user, pos_item=pos_item).mean() \
                    if not negative_sampling else \
                    model(user=user, pos_item=pos_item, neg_item=neg_item).mean()

            loss = MF_loss(preds)

            loss.backward()
            opt_fn.step()
            losses.append(loss.data.to(device).tolist())

        if epoch % 10 == 0:
            t1=time.time()
            print(f'Epoch: {epoch}, Time: {round(t1-t0,2)}, /Average loss {np.mean(losses[-10:]).round(5)}')
            t0=time.time()

    return model, losses