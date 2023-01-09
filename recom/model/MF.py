import sys, os
sys.path.append(os.getcwd())

# pack load
from torch import LongTensor, Tensor
import torch.nn as nn
import torch.nn.functional as F

# local load
from recom.utils.sampler import mf_data_loader
from recom.utils.util import rating_min_max_scalar


def RMSE(rate, preds):
    """ RMSE loss
        Mainly for evaluating the MF model

    :param rate: tensor of original rating

    :param preds: tensor of predictions

    :return: rmse value of the predictions
    """

    return ((rate - preds.view(-1))**2).mean()


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

                preds = self(user, item)
                loss = RMSE(rate, preds)

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
    def __init__(self, n_user, n_item, k_dim):
        super(_MF, self).__init__()
        self.items = list(range(n_item))
        # embeddings of interest
        self.embedding_user = nn.Embedding(n_user, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        nn.init.normal_(self.embedding_user.weight, mean=0, std=0)
        nn.init.normal_(self.embedding_item.weight, mean=0, std=0)

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

    def forward(self, user, item):
        return (self.embedding_user(user) \
                * self.embedding_item(item)).sum(1)


class BiasedFunkSvd(_MF):
    def __init__(self, n_user, n_item, k_dim, user_deviation, item_deviation, global_bias):
        super(_MF, self).__init__()
        self.items = list(range(n_item))
        # embeddings of interest
        self.embedding_user = nn.Embedding(n_user, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        nn.init.normal_(self.embedding_user.weight, mean=0, std=0)
        nn.init.normal_(self.embedding_item.weight, mean=0, std=0)
        # global bias, and user, item deviation from that bias
        self.global_bias = nn.Parameter(Tensor([global_bias]), requires_grad=False)
        self.dev_user = nn.Parameter(Tensor(list(user_deviation.values())), requires_grad=False)
        self.dev_item = nn.Parameter(Tensor(list(item_deviation.values())), requires_grad=False)

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

        # dot product + bias + deviation
        return (self.embedding_user(user) \
                @ self.embedding_item(item).T) \
            + self.dev_user[user] \
            + self.dev_item[item] \
            + self.global_bias

    def forward(self, user, item):
        """ Predict rating for user and a set of item based on their latent factors/ embeddings.

        :param user: int, list, or tensor
            The index of user(s) of interest.

        :param item: int, list, or tensor
            The index of item(s) of interest

        :return: prediction of score
        """
        # dot product + bias + deviation
        return (self.embedding_user(user) \
                * self.embedding_item(item)).sum(1) \
            + self.dev_user[user] \
            + self.dev_item[item] \
            + self.global_bias


class PMF(_MF):
    """ Probabilistic Matrix Factorization
        - [ref]: Mnih, A., & Salakhutdinov, R. R. (2007). Probabilistic matrix factorization. Advances in neural information processing systems, 20.
        - [url]: https://proceedings.neurips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf
    """
    def __init__(self, n_user, n_item, k_dim
                 , std_user=1, std_item=1):
        super(_MF, self).__init__()
        self.items = list(range(n_item))
        # embeddings of interest
        self.embedding_user = nn.Embedding(n_user, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        nn.init.normal_(self.embedding_user.weight, mean=0, std=std_user)
        nn.init.normal_(self.embedding_item.weight, mean=0, std=std_item)
        self.sigmoid = F.sigmoid

    def pred(self, user, item):
        from torch import LongTensor

        # Tensorize
        if isinstance(user, int): user = LongTensor([user])
        if isinstance(user, list): user = LongTensor(user)
        if isinstance(item, int): item = LongTensor([item])
        if isinstance(item, list): item = LongTensor(item)

        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)

        return user_emb @ item_emb.T # dot product

    def forward(self, user, item):
        return (self.embedding_user(user) \
                * self.embedding_item(item)).sum(1)

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

        proj_train_dict = rating_min_max_scalar(train_dict)

        losses = []

        t0 = time.time()
        for epoch in range(n_epochs):
            train_data = mf_data_loader(proj_train_dict, user_per_ep, item_per_ep, batch_size)

            for i, batch in enumerate(train_data):
                user, item, rate = batch

                self.zero_grad()

                # variablize
                user = autograd.Variable(LongTensor(user)).to(compute_device)
                item = autograd.Variable(LongTensor(item)).to(compute_device)
                rate = autograd.Variable(rate).to(compute_device)

                # use sigmoid as projection
                preds = self(user, item)
                loss = RMSE(rate, self.sigmoid(preds))

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