import sys, os
sys.path.append(os.getcwd())

import torch.nn as nn
from torch import Tensor
from recom.utils.sampler import mf_data_loader

def RMSE(preds):
    """ RMSE loss of prediction

    :param preds: tensor of predictions

    :return: rmse value of the predictions
    """
    return (preds**2).mean()


class _MF(nn.Module):
    """ Base class of Matrix Factorization.
        Latent factor based model for recommendation.
    """
    def __init__(self, n_user, n_item, k_dim):
        super(_MF, self).__init__()

    def pred(self, user, item):
        """ Predict rating for user and a set of item based on their latent factors/ embeddings.
        """
        pass

    def forward(self, user, item, rating):
        """ The way model train itself by inputs
        """
        pass

    def fit(self):
        """ How the model train itself via inputs
        """
        pass

    def pred_on_candidate_dict(self, cand_dict):
        """ The result of predictions on ratings
        """
        pass


class FunkSvd(_MF):
    def __init__(self, n_user, n_item, k_dim, mu_emb=0, sig_emb=1):
        super(FunkSvd, self).__init__()
        self.items = list(range(n_item))
        # embeddings of interest
        self.embedding_user = nn.Embedding(n_user, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        self.embedding_user.weight.data.uniform_(mu_emb, sig_emb)
        self.embedding_item.weight.data.uniform_(mu_emb, sig_emb)

    def pred(self, user, item):
        """ Predict rating for user and a set of item based on their latent factors/ embeddings.

        :param user: int, list, or tensor
            The index of user(s) of interest.

        :param item: int, list, or tensor
            The index of item(s) of interest

        :return: prediction of score
        """
        from torch import LongTensor

        # Tensorize
        if isinstance(user, int): user = LongTensor([user])
        if isinstance(user, list): user = LongTensor(user)
        if isinstance(item, int): item = LongTensor([item])
        if isinstance(item, list): item = LongTensor(item)

        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)

        return user_emb @ item_emb.T # dot product

    def forward(self, user, item, rating):
        preds = self.pred(user, item)
        return rating - preds

    def fit(self, train_dict, opt_fn
            , batch_size=128
            , n_epochs=128
            , user_per_ep=128, item_per_ep=32
            , report_interval=10, use_cuda=False):

        import time
        from torch import autograd, LongTensor, device

        if use_cuda:
            compute_device = device('cuda')
            self.cuda()
        else:
            compute_device = device('cpu')

        losses = []

        t0 = time.time()
        for epoch in range(n_epochs):
            train_data = mf_data_loader(train_dict, user_per_ep, item_per_ep, batch_size)

            for i, batch in enumerate(train_data):
                user, item, rate = batch

                self.zero_grad()

                # variablize
                user = autograd.Variable(LongTensor(user)).to(compute_device)
                item = autograd.Variable(LongTensor(item)).to(compute_device)
                rate = autograd.Variable(rate).to(compute_device)

                preds = self(user, item, rate)
                loss = RMSE(preds)

                loss.backward()
                opt_fn.step()
                losses.append(loss.data.to(compute_device).tolist())
            if report_interval > 0 \
                    and ((epoch+1) % report_interval == 0):
                t1=time.time()
                print(f'Epoch: {epoch+1}, Time: {round(t1-t0,2)}, /Average loss {round(sum(losses[-report_interval:])/report_interval, 5)}')
                t0=time.time()

        self.last_train_loss = losses

        return self

    def pred_on_candidate_dict(self, cand_dict):
        from torch import argsort

        predict = {}

        for user in cand_dict:
            candidate = cand_dict[user]
            pred = self.pred(user, candidate).view(-1)
            pred_ix = argsort(-pred)
            predict[user] = [candidate[ix] for ix in pred_ix]

        return predict


class FunkSvd(_MF):
    def __init__(self, n_user, n_item, k_dim, mu_emb=0, sig_emb=1):
        super(_MF, self).__init__()
        self.items = list(range(n_item))
        # embeddings of interest
        self.embedding_user = nn.Embedding(n_user, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        self.embedding_user.weight.data.uniform_(mu_emb, sig_emb)
        self.embedding_item.weight.data.uniform_(mu_emb, sig_emb)

    def pred(self, user, item):
        """ Predict rating for user and a set of item based on their latent factors/ embeddings.

        :param user: int, list, or tensor
            The index of user(s) of interest.

        :param item: int, list, or tensor
            The index of item(s) of interest

        :return: prediction of score
        """
        from torch import LongTensor

        # Tensorize
        if isinstance(user, int): user = LongTensor([user])
        if isinstance(user, list): user = LongTensor(user)
        if isinstance(item, int): item = LongTensor([item])
        if isinstance(item, list): item = LongTensor(item)

        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)

        return user_emb @ item_emb.T # dot product

    def forward(self, user, item, rating):
        preds = self.pred(user, item)
        return rating - preds

    def fit(self, train_dict, opt_fn
            , batch_size=128
            , n_epochs=128
            , user_per_ep=128, item_per_ep=32
            , report_interval=10, use_cuda=False):

        import time
        from torch import autograd, LongTensor, device

        if use_cuda:
            compute_device = device('cuda')
            self.cuda()
        else:
            compute_device = device('cpu')

        losses = []

        t0 = time.time()
        for epoch in range(n_epochs):
            train_data = mf_data_loader(train_dict, user_per_ep, item_per_ep, batch_size)

            for i, batch in enumerate(train_data):
                user, item, rate = batch

                self.zero_grad()

                # variablize
                user = autograd.Variable(LongTensor(user)).to(compute_device)
                item = autograd.Variable(LongTensor(item)).to(compute_device)
                rate = autograd.Variable(rate).to(compute_device)

                preds = self(user, item, rate)
                loss = RMSE(preds)

                loss.backward()
                opt_fn.step()
                losses.append(loss.data.to(compute_device).tolist())
            if report_interval > 0 \
                    and ((epoch+1) % report_interval == 0):
                t1=time.time()
                print(f'Epoch: {epoch+1}, Time: {round(t1-t0,2)}, /Average loss {round(sum(losses[-report_interval:])/report_interval, 5)}')
                t0=time.time()

        self.last_train_loss = losses

        return self

    def pred_on_candidate_dict(self, cand_dict):
        from torch import argsort

        predict = {}

        for user in cand_dict:
            candidate = cand_dict[user]
            pred = self.pred(user, candidate).view(-1)
            pred_ix = argsort(-pred)
            predict[user] = [candidate[ix] for ix in pred_ix]

        return predict


