# modeling svd
import torch
import torch.nn as nn
import torch.nn.functional as F

class FunkSVD(nn.Module):
    """ Matrix Factorization based method, referring to FunkSVD
    """
    def __init__(self, n_user, n_item, k_dim, negative_sampling:bool):
        super(FunkSVD, self).__init__()
        self.k_dim = k_dim
        self.negative_sampling = negative_sampling # if training with negative sampling
        # embeddings of interest
        self.embedding_user = nn.Embedding(n_user, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        self.embedding_user.weight.data.uniform_(0, 1)
        self.embedding_item.weight.data.uniform_(0, 1)

    def forward(self, user, pos_item, neg_item=None):
        pos_rat = (self.embedding_user(user)*self.embedding_item(pos_item)).sum(1)

        if not self.negative_sampling:
            return -pos_rat
        neg_emb = self.embedding_item(neg_item)
        user_emb = self.embedding_user(user).view(-1, self.k_dim, 1)
        neg_rat = -(torch.bmm(neg_emb, user_emb)).view(-1)
        return torch.cat((pos_rat, neg_rat), 0)

    def get_user_embedding(self, user):
        return self.embedding_user(user).data

    def get_item_embedding(self, item):
        return self.embedding_item(item).data

def MF_loss(preds):
    return -F.logsigmoid(preds).sum()