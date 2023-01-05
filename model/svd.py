# modeling svd
import torch
import torch.nn as nn
from torch import tensor

class FunkSVD(nn.Module):
    """ Matrix Factorization based method, referring to FunkSVD
    """
    def __init__(self, n_user, n_item, k_dim, negative_sampling:bool):
        super(FunkSVD, self).__init__()
        self.k_dim=k_dim
        self.negative_sampling = negative_sampling # if training with negative sampling
        # embeddings of interest
        self.embedding_user = nn.Embedding(n_user, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        self.embedding_user.weight.data.uniform_(0, 1)
        self.embedding_item.weight.data.uniform_(0, 1)

    def pred_score(self, user, item):
        user = tensor([user])
        if not isinstance(item, torch.Tensor):
            item = tensor(item)
        predicted_score = self.embedding_user(user) \
                          @ self.embedding_item(item).T
        return predicted_score.view(-1)

    def forward(self, user, pos_item, neg_item=None):
        pos_rat = (self.embedding_user(user)*self.embedding_item(pos_item)).sum(1)

        if not self.negative_sampling:
            return -pos_rat

        neg_emb = self.embedding_item(neg_item)
        # if len(neg_emb.shape) == 2:
        #     r, c = neg_emb.shape
        #     neg_emb = neg_emb.view(r, 1, self.k_dim)

        user_emb = self.embedding_user(user).view(-1, self.k_dim, 1) # transform shape
        neg_rat = -(torch.bmm(neg_emb, user_emb)).view(-1) # use bmm to calcualte score

        return torch.cat((pos_rat, neg_rat), 0) # concat positive and negative ratings


class BiasedFunkSVD(nn.Module):
    """ Matrix Factorization based method, referring to FunkSVD
    """
    def __init__(self, n_user, n_item, k_dim, user_bias, item_bias, negative_sampling:bool):
        super(BiasedFunkSVD, self).__init__()
        self.k_dim=k_dim
        self.negative_sampling = negative_sampling # if training with negative sampling
        # bias
        self.bias_user = nn.Parameter(tensor(list(user_bias.values())))
        self.bias_item = nn.Parameter(tensor(list(item_bias.values())))
        self.bias = nn.Parameter(tensor([0.]), requires_grad=True)
        # embeddings of interest
        self.embedding_user = nn.Embedding(n_user, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        self.embedding_user.weight.data.uniform_(0, 1)
        self.embedding_item.weight.data.uniform_(0, 1)

    def pred_score(self, user, item):
        user = tensor([user])
        if not isinstance(item, torch.Tensor):
            item = tensor(item)
        predicted_score = self.embedding_user(user) \
                          @ self.embedding_item(item).T \
                          + self.bias_user[user] \
                          + self.bias_item[item] \
                          + self.bias
        return predicted_score.view(-1)

    def forward(self, user, pos_item, neg_item=None):
        pos_rat = (self.embedding_user(user)
                   * self.embedding_item(pos_item)).sum(1) \
                  + self.bias_user[user] \
                  + self.bias_item[pos_item] \
                  + self.bias

        if not self.negative_sampling:
            return -pos_rat

        neg_emb = self.embedding_item(neg_item)
        if len(neg_emb.shape) == 2:
            r, c = neg_emb.shape
            neg_emb = neg_emb.view(r, 1, self.k_dim)

        user_emb = self.embedding_user(user).view(-1, self.k_dim, 1) # transform shape
        neg_size = neg_emb.shape[1]

        neg_rat = -(
                torch.bmm(neg_emb, user_emb)[:,:,0]
                + self.bias_user[user].view(-1, 1).expand(-1, neg_size)
                + self.bias_item[neg_item]
        ).view(-1) # use bmm to calcualte score

        return torch.cat((pos_rat, neg_rat), 0) # concat positive and negative ratings