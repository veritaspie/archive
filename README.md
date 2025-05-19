## Divergence estimation of continuous vector embeddings from `X(y=1)` and `X(y=0)`
- Implementation of Wang-Kulkarni-Verdu (2009) estimator of KL div. See Formula(25)
- Wang, Qing, Sanjeev R. Kulkarni, and Sergio Verdú. "Divergence estimation for multidimensional densities via $ k $-Nearest-Neighbor distances." IEEE Transactions on Information Theory 55.5 (2009): 2392-2405.
- `/snippets/mae_lit.py`: implementation in torch lightening format
- `/snippets/mae_kl_divergence.ipynb`: estimated KL div (k=3) during MTAE, measured on PTB-XL
