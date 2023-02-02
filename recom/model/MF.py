import sys, os
sys.path.append(os.getcwd())

# pack load
from torch import Tensor, row_stack
import torch.nn as nn
import torch.nn.functional as F

# local load
from recom.utils.sampler import mf_data_loader
from recom.utils.util import rating_min_max_scalar, rating_vectorize


def SE(input:Tensor, target:Tensor):
    """ Squared error (SE) for two tensor.
        SE serves as the major loss evaluation metric for MF model.
    """
    return (input-target)**2


def maskedSE(input:Tensor, target:Tensor):
    """ Compute SE between two tensors only on those have values.

        Usage: 

        >>> input =  Tensor([0, 1, 1, 1, 1])
        >>> target = Tensor([0, 1, 0, 2, 3])

        >>> maskedSE(input, target) # do not compute value on the thrid element
        
        tensor([0, 0, 0, 1, 4]) 

    """
    # Compute squared error for testing set
    mask = (target!=0)

    return ((input[mask]-target[mask])**2)


class _MF(nn.Module):
    """ Base class of Matrix Factorization.
        Latent factor based model for recommendation.
    """
    def __init__(self, n_user, n_item):
        super(_MF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item

    def pred(self, ):
        """ Predict rating for user and a set of item based on their latent factors/ embeddings.
        """
        pass

    def pred_all(self, ):
        """ Estimate the whole rating matrix.
        """
        pass

    def forward(self, user, item, rating):
        """ The way model train itself by inputs
        """
        pass

    def fit(self, train_dict, opt_fn
            , n_epochs=128 , batch_size=128     # training                 
            , method='all', user_per_ep=None, item_per_ep=None # sampling
            , report_interval=10                # train reporting
            , report_test=False, test_dict=None # testing 
            , use_cuda=False):

        assert method in ['all', 'sample'], f'Invalid method {method}'

        import time
        from torch import autograd, LongTensor, device
        from numpy import sqrt, mean

        if use_cuda:
            compute_device = device('cuda')
            self.cuda()
            test_mat = rating_vectorize(test_dict, self.n_user, self.n_item) if report_test else None
        else:
            compute_device = device('cpu')

        train_loss_by_ep = []
        test_rmse_by_ep = [] if report_test else None

        t0 = time.time()
        for epoch in range(n_epochs):
            train_data = mf_data_loader(train_dict, user_per_ep, item_per_ep, batch_size)

            ep_loss = []
            for i, batch in enumerate(train_data):
                user, item, rate = batch

                self.zero_grad()

                # variablize
                user = autograd.Variable(LongTensor(user)).to(compute_device)
                item = autograd.Variable(LongTensor(item)).to(compute_device)
                score = autograd.Variable(rate).to(compute_device)

                preds = self(user, item)
                loss = SE(input=preds, target=score)

                loss.mean().backward()
                opt_fn.step()
                ep_loss.extend(loss.data.to(compute_device).tolist())

            train_loss_by_ep.append(sqrt(mean(ep_loss)))

            # test
            if report_test:
                preds = self.pred_all()
                test_rmse = maskedSE(input=preds, target=test_mat)
                test_rmse_by_ep.append(test_rmse.data.to(compute_device).tolist())

            if report_interval > 0 \
                    and ((epoch+1) % report_interval == 0):
                t1=time.time()
                print(f'Epoch: {epoch+1}, Time: {round(t1-t0,2)}, /Average loss {round(sum(train_loss_by_ep[-report_interval:])/report_interval, 5)}')
                if report_test:
                    print(f'\t\t\t/Average test loss {round(sum(test_rmse_by_ep[-report_interval:])/report_interval, 5)}')
                t0=time.time()

        self.last_train_loss = train_loss_by_ep
        self.last_test_rmse = test_rmse_by_ep

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
        nn.init.normal_(self.embedding_user.weight, mean=1, std=0)
        nn.init.normal_(self.embedding_item.weight, mean=1, std=0)

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
        # dot product + bias + deviation
        return (self.embedding_user(user) \
                * self.embedding_item(item)).sum(1) \
            + self.dev_user[user] \
            + self.dev_item[item] \
            + self.global_bias


""" Probabilistic Matrix Factorization Related Methods
        - [ref]: Mnih, A., & Salakhutdinov, R. R. (2007). Probabilistic matrix factorization. Advances in neural information processing systems, 20.
        - [url]: https://proceedings.neurips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf
"""
class PMF(_MF):
    """ Probabilistic Matrix Factorization
        PMF has some normality assumptions over users' and items' latent factor,
        seen as a normally distributed prior. By adding these priors into param,
        we have the most classic pmf model.
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


class LogisticPMF(_MF):
    """ Logistic Probabilistic Matrix Factorization
        PMF has some normality assumptions over users' and items' latent factor,
        seen as a normally distributed prior. By adding these priors into param,
        we have the most classic pmf model.
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


class ConstrainedPMF(_MF):
    def __init__(self, n_user, n_item, k_dim
                 , std_user=1, std_item=1):
        super(_MF, self).__init__()
        self.n_user = n_user # placeholder
        # embeddings of interest
        self.user_offset = nn.Embedding(n_user, 1)
        self.latent_similarity = nn.Embedding(n_item, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        nn.init.normal_(self.user_offset.weight, mean=0, std=0)
        nn.init.normal_(self.latent_similarity.weight, mean=0, std=std_user)
        nn.init.normal_(self.embedding_item.weight, mean=0, std=std_item)
        # rating history
        self.rat_hist = None
        # layer
        self.sigmoid = F.sigmoid

    def pred(self, user, item):
        from torch import LongTensor

        # Tensorize
        if isinstance(user, int): user = [user]
        if isinstance(item, int): item = LongTensor([item])
        if isinstance(item, list): item = LongTensor(item)

        user_offset = self.user_offset(LongTensor(user))

        user_emb = row_stack([
            self.latent_similarity(self.rat_hist[int(u)]).mean(0)
            for u in user
        ])

        item_emb = self.embedding_item(item)

        # dot product
        return (user_emb + user_offset) @ item_emb.T

    def forward(self, user_ix, user_emb, item):
        user_offset = self.user_offset(user_ix)
        embedding_user = user_offset + user_emb

        return (embedding_user * self.embedding_item(item)).sum(1)

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
            self.latent_similarity.to('cpu')
        else:
            compute_device = device('cpu')

        self.rat_hist = self.get_rating_history(train_dict)
        proj_train_dict = rating_min_max_scalar(train_dict)

        losses = []

        t0 = time.time()
        for epoch in range(n_epochs):
            train_data = mf_data_loader(proj_train_dict, user_per_ep, item_per_ep, batch_size)

            for i, batch in enumerate(train_data):
                user, item, rate = batch

                self.zero_grad()

                # variablize
                user_emb = row_stack([
                    self.latent_similarity(self.rat_hist[int(u)]).mean(0) \
                    for u in user
                ]).to(compute_device)

                user_ix = autograd.Variable(LongTensor(user)).to(compute_device)

                item = autograd.Variable(LongTensor(item)).to(compute_device)
                rate = autograd.Variable(rate).to(compute_device)

                # use sigmoid as projection
                preds = self(user_ix, user_emb, item)
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

    def get_rating_history(self, rating):
        from torch import LongTensor

        return dict(
            (user, LongTensor(list(rating[user].keys())))
            for user in rating
        )

    def set_history(self, rating)->None:
        self.rat_hist = self.get_rating_history(rating)