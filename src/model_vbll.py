"""Base model: MLP feature extractor + Variational Bayesian Last Layer (VBLL).

Architecture and training follow the official D3M code for tabular data
(github.com/teivng/d3m, d3m/models/mlp_model.py):
  * feature extractor FE: Linear -> ELU, then residual blocks
    x <- ELU(Linear(x) + x), dropout 0.2 after every activation;
  * head: vbll.DiscClassification with a DIAGONAL Gaussian over the last-layer
    weights, regularization weight = reg_factor / n_train (reg_factor = 100);
  * AdamW, lr 1e-3, weight decay 1e-4, batch 64.

What the VBLL head gives us (vbll 0.4.9, `logit_predictive`):
    z | x  ~  N( W_mean phi(x),  diag( sum_j phi_j(x)^2 s_cj^2 + s_noise_c^2 ) )
i.e. an independent Gaussian for every class logit c and every input x, whose
variance grows with the size of the feature vector phi(x).  Training maximises
the ELBO: E_q[log softmax(z)_y] (Jensen bound) - reg_weight * KL(q(W) || prior).
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vbll


class VBLLNet(nn.Module):
    def __init__(self, n_train, in_dim=2, hidden=32, n_mid=3, n_classes=2,
                 dropout=0.2, reg_factor=100.0, prior_scale=1.0, wishart_scale=1.0):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.mid = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_mid)])
        self.drop = nn.Dropout(dropout)
        self.head = vbll.DiscClassification(
            hidden, n_classes, reg_factor / n_train, parameterization="diagonal",
            prior_scale=prior_scale, wishart_scale=wishart_scale)

    def features(self, x):
        h = self.drop(F.elu(self.inp(x)))
        for layer in self.mid:
            h = self.drop(F.elu(layer(h) + h))   # residual (skip) connection
        return h

    def forward(self, x):
        return self.head(self.features(x))      # VBLLReturn (has train_loss_fn)

    @torch.no_grad()
    def logit_dist(self, X):
        """Posterior over logits for inputs X (numpy or tensor).
        Returns (mean [n, C], variance [n, C]) as float tensors."""
        self.eval()
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        d = self.head.logit_predictive(self.features(x))
        return d.mean.clone(), d.covariance_diagonal.clone()

    @torch.no_grad()
    def predict_proba(self, X):
        """Probabilities of the deployed (mean) model: softmax of the mean logits."""
        mu, _ = self.logit_dist(X)
        return torch.softmax(mu, -1).numpy()

    def predict(self, X):
        """Deployed model f(x) = argmax of the mean logits (= D3M's pseudo-label)."""
        return self.predict_proba(X).argmax(1)


def train_vbll(X, y, seed, epochs=50, batch_size=64, lr=1e-3, wd=1e-4, **model_kw):
    """Train the base model on labelled ID data. Returns (model, seconds)."""
    torch.manual_seed(seed)
    model = VBLLNet(n_train=len(X), **model_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            loss = model(Xt[idx]).train_loss_fn(yt[idx])   # negative ELBO
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, time.perf_counter() - t0
