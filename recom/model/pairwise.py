import sys, os
sys.path.append(os.getcwd())

# pack load
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# local load
from recom.utils.sampler import pairwise_loader
from recom.utils.util import tensorize

def BPRloss(preds):
    return -F.logsigmoid(preds.view(-1)).mean()


class BPR(nn.Module):
    def __init__(self, n_user, n_item, k_dim, std_user, std_item) -> None:
        super(BPR, self).__init__()
        self.items = list(range(n_item))
        self.embedding_user = nn.Embedding(n_user, k_dim)
        self.embedding_item = nn.Embedding(n_item, k_dim)
        # init param
        nn.init.normal_(self.embedding_user.weight, mean=0, std=std_user)
        nn.init.normal_(self.embedding_item.weight, mean=0, std=std_item)

    def pred(self, user, item):
        user, item  = tensorize(user, item)

        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)

        return user_emb @ item_emb.T

    def forward(self, user, pos_item, neg_item):
        user_emb = self.embedding_user(user)
        posi_emb = self.embedding_item(pos_item)
        negi_emb = self.embedding_item(neg_item)

        return (user_emb*posi_emb).sum(1) \
               - (user_emb*negi_emb).sum(1)

    def fit(self, train_dict, opt_fn
            , batch_size=128
            , n_epochs=128
            , user_per_ep=128
            , pos_item_per_ep=32
            , neg_sample_size=4
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
            train_data = pairwise_loader(
                train_dict
                , items=self.items
                , user_size=user_per_ep
                , pos_size=pos_item_per_ep
                , neg_size=neg_sample_size
                , batch_size=batch_size
            )

            for i, batch in enumerate(train_data):
                user, pos_item, neg_item = batch

                self.zero_grad()

                # variablize
                user = autograd.Variable(LongTensor(user)).to(compute_device)
                posi = autograd.Variable(LongTensor(pos_item)).to(compute_device)
                negi = autograd.Variable(LongTensor(neg_item)).to(compute_device)

                # use sigmoid as projection
                preds = self(user, posi, negi)
                loss = BPRloss(preds)

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