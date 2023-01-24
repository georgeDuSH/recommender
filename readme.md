# Recommendation Algo Notes

## 1 Background
The repository contains models in personalized recommendation
    , with simple implementation with PyTorch
    , toolkits for file parsing, sampling, and evaluation.

Detailed notes on models are available in my Notion attached.

## 2 Models Covered
### 2.1 Matrix Factorization based 
Models here mainly focus on classic matrix factorization, or latent factor models.

- **`Funk SVD`**, **`Biased SVD`**: Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.
  - <u>Funk SVD</u> model users and items into latent vectors, and estimate rating via inner product of each. Minimizing RMSE between estimated ratings and observed ones to fit the model.
  - On the top of that, <u>biased SVD</u> add user, item, and global bias, performing better on RMSE on testing items.
 
- **`PMF`**, **`Logistic PMF`**, **`Constrained PMF`**: Mnih, A., & Salakhutdinov, R. R. (2007). Probabilistic matrix factorization. Advances in neural information processing systems, 20.
  - <u>PMF</u> add priors over FunkSVD, supposing latent vectors are normally distributed.
  - <u>Logistic PMF</u> treats 

### 2.2 Pairwise Ranking based Model
- **`BPR`**: Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2012). BPR: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618.

### 2.3 Auto Encoder based 
- **`User-based AutoEncoder`**, **`Item-based AutoEncoder`**: Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015, May). Autorec: Autoencoders meet collaborative filtering. In Proceedings of the 24th international conference on World Wide Web (pp. 111-112).
  - Recommendation based on deep learning, with masked training only for observed inputs via MLP with encoder-decoder-structure.