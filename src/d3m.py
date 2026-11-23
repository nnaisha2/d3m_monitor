"""D3M: Disagreement-Driven Deterioration Monitoring (Nguyen et al., NeurIPS 2025).

This file implements the Calibrate and Deploy stages myself.  The only thing it
needs from the trained model is, for every input x, the Gaussian posterior over
the class logits that the VBLL head produces:

    q(z | x) = N( mu(x), diag(var(x)) ),   z in R^C.

Algorithm for ONE batch of m unlabelled inputs (used in both stages):

  1. Pseudo-labels of the deployed (mean) model:  ybar_i = argmax_c mu_c(x_i).
     (Official code: argmax of the predictive mean `ll_dist.loc`.)
  2. For k = 1..K draw logits  z_i^(k) ~ q(z | x_i), independently for every i
     (official code: `ll_dist.rsample([K])`, one Gaussian per input).
  3. Temperature-scaled softmax:  p_i^(k) = softmax(z_i^(k) / tau).
  4. Sample a label from it:      yhat_i^(k) ~ Categorical(p_i^(k)).
  5. Disagreement rate of draw k: DisRate(k) = (1/m) sum_i 1[yhat_i^(k) != ybar_i].
  6. Maximum disagreement rate:   phi = max_k DisRate(k).

Calibrate: repeat 1-6 on T bootstrap batches (size m, with replacement) of a
held-out ID validation set  ->  Phi = {phi_1, ..., phi_T}.
Deploy:    compute phi~ on the query batch and raise an alarm if
           phi~ >= Quantile_{1-alpha}(Phi)      (official code uses >=).

Phi depends on m (a max over averages of m terms), so we keep one Phi per m.

Two engines compute the SAME random variable phi:

* engine="literal": steps 1-6 exactly as written (K x m x C Gaussian draws).
  Slow on a laptop (minutes per calibration for K = 5000), used as reference.
* engine="exact" (default): uses one observation.  Because every logit draw
  and every sampled label is independent, the indicator 1[yhat_i^(k) != ybar_i]
  is a coin flip with probability
        p_i = E_z[ 1 - softmax(z / tau)_{ybar_i} ]       (disagreement prob.)
  So each DisRate(k) is an average of m independent coin flips with
  different p_i (a "Poisson-binomial" variable), and the K values DisRate(1..K)
  are i.i.d.  We (a) compute p_i once per point (numerical integration for
  two classes, Monte Carlo otherwise), (b) compute
  the exact distribution of m * DisRate with a short dynamic program, and
  (c) draw the maximum of K i.i.d. copies with the inverse CDF of the max,
  P(max <= j) = F(j)^K.  Cost O(m^2) per batch, independent of K.
  tests/test_d3m.py checks that both engines give the same distribution.

Shapes: mu, var [B, m, C] where B is a "batch of batches" (many bootstrap
rounds processed at once).
"""
import numpy as np
import torch

# Upper bound on B*K*m*C elements per tensor operation (~160 MB in float32).
_MAX_ELEMS = 40_000_000
# Monte Carlo draws used to estimate each point's disagreement probability p_i.
P_DRAWS = 2000


def sample_labels(probs, gen):
    """Draw one label per row of `probs` (shape [..., C]) by inverse-CDF sampling.

    u ~ Uniform(0,1); the label is the number of cumulative probabilities below u.
    Same distribution as torch.distributions.Categorical, but explicit.
    """
    u = torch.rand(probs.shape[:-1] + (1,), generator=gen)
    cdf = probs.cumsum(-1)
    return (u > cdf).sum(-1).clamp(max=probs.shape[-1] - 1)


def disagreement_rates(mu, var, K, temperature, gen, sample=True):
    """Steps 1-5. mu, var: [B, m, C]. Returns DisRate(k): tensor [B, K].

    sample=False is an ablation: use argmax of the sampled logits instead of
    sampling a label (the paper's App. C.5 says this collapses with VBLL).
    """
    ybar = mu.argmax(-1).unsqueeze(1)                          # [B, 1, m]
    eps = torch.randn((mu.shape[0], K) + mu.shape[1:], generator=gen)
    z = mu.unsqueeze(1) + var.sqrt().unsqueeze(1) * eps        # [B, K, m, C]
    if sample:
        yhat = sample_labels(torch.softmax(z / temperature, dim=-1), gen)
    else:
        yhat = z.argmax(-1)                                    # [B, K, m]
    # count disagreements as integers, divide in float64 so that phi values are
    # exactly j/m (the alarm rule phi >= threshold is sensitive to rounding)
    return (yhat != ybar).sum(-1).double() / mu.shape[1]       # [B, K]


def disagreement_prob(mu, var, n_draws, temperature, gen, sample=True):
    """Per-point probability that ONE draw disagrees with ybar.  mu, var: [n, C]
    -> p: np.ndarray [n].

        sample=True :  p_i = E_z[ 1 - softmax(z/tau)_{ybar_i} ]
        sample=False:  p_i = P_z[ argmax z != ybar_i ]       (ablation)

    mean_i p_i is the EXPECTED DisRate(k); D3M's max over K draws sits above it.
    Two classes: computed by numerical integration (exact, fast).  More classes:
    Monte Carlo with n_draws logit draws.
    """
    if mu.shape[-1] == 2:
        return _disagreement_prob_binary(mu, var, temperature, sample)
    ybar = mu.argmax(-1)                                       # [n]
    chunk = max(1, _MAX_ELEMS // (n_draws * mu.shape[-1]))
    out = []
    for s in range(0, len(mu), chunk):
        m_, v_, y_ = mu[s:s + chunk], var[s:s + chunk], ybar[s:s + chunk]
        eps = torch.randn((n_draws,) + m_.shape, generator=gen)   # [D, n, C]
        z = m_ + v_.sqrt() * eps
        if sample:
            p_agree = torch.softmax(z / temperature, -1).gather(
                -1, y_.expand(n_draws, -1).unsqueeze(-1)).squeeze(-1)
        else:
            p_agree = (z.argmax(-1) == y_).float()
        out.append(1.0 - p_agree.mean(0))
    return torch.cat(out).numpy().astype(np.float64)


def _disagreement_prob_binary(mu, var, temperature, sample):
    """Two classes.  The logit gap D = z_ybar - z_other is Gaussian:
        D ~ N(a, b),  a = |mu_1 - mu_0| >= 0,  b = var_0 + var_1
    (the two logits are independent Gaussians in VBLL's diagonal posterior).
    One sampled label disagrees with ybar with probability sigmoid(-D / tau),
    so p_i = E[sigmoid(-D / tau)]: a 1-D Gaussian integral, evaluated with
    64-point Gauss-Hermite quadrature.  Without label sampling (ablation) the
    draw disagrees iff D < 0, so p_i = Phi(-a / sqrt(b)).
    """
    from scipy.special import expit
    from scipy.stats import norm
    a = (mu[:, 1] - mu[:, 0]).abs().double().numpy()
    b = var.sum(-1).double().numpy()
    if not sample:
        return norm.cdf(-a / np.sqrt(b))
    x, w = np.polynomial.hermite_e.hermegauss(64)   # nodes/weights for N(0, 1)
    D = a[:, None] + np.sqrt(b)[:, None] * x[None, :]
    return (expit(-D / temperature) * w).sum(1) / np.sqrt(2 * np.pi)


def poisson_binomial_pmf(p):
    """Exact distribution of the number of disagreements S = sum_i Bernoulli(p_i).

    p: [B, m] -> pmf [B, m+1] with pmf[b, j] = P(S = j).  Dynamic programme:
    add one point at a time; the count either stays (prob 1 - p_i) or goes up
    by one (prob p_i).
    """
    B, m = p.shape
    pmf = np.zeros((B, m + 1))
    pmf[:, 0] = 1.0
    for i in range(m):
        pi = p[:, i:i + 1]
        old = pmf[:, :i + 2].copy()
        pmf[:, :i + 2] = old * (1.0 - pi)
        pmf[:, 1:i + 2] += old[:, :i + 1] * pi
    return pmf


def sample_max_of_K(pmf, K, rng):
    """Draw max_k DisRate(k) for K i.i.d. copies of S/m, one value per row.

    P(max <= j) = F(j)^K, so with u ~ U(0,1) the max is the smallest j with
    F(j) >= u^(1/K)  (inverse-CDF sampling of the maximum).
    """
    B, m1 = pmf.shape
    cdf = np.cumsum(pmf, axis=1)
    target = rng.random(B) ** (1.0 / K)
    j = np.minimum((cdf < target[:, None]).sum(1), m1 - 1)
    return j / (m1 - 1)


class D3M:
    """Calibrate once per batch size m, then test query batches.

    logit_fn(X: np.ndarray) -> (mu, var) torch tensors [n, C].
    statistic: "max" (the paper) or "mean" (ablation: E[DisRate], no max over K).
    sample:    True (the paper: categorical label sampling) or False (ablation:
               argmax of the sampled logits).
    engine:    "exact" (fast, same distribution) or "literal" (steps 1-6 as written).
    """

    def __init__(self, logit_fn, K=5000, temperature=1.0, T=1000, seed=0,
                 statistic="max", sample=True, engine="exact"):
        self.logit_fn = logit_fn
        self.K, self.temperature, self.T = K, temperature, T
        self.statistic, self.sample, self.engine = statistic, sample, engine
        self.gen = torch.Generator().manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.Phi = {}  # m -> np.ndarray of T calibration values

    # ---- building blocks ----------------------------------------------
    def point_probs(self, X):
        """Disagreement probability p_i of every input in X (np.ndarray [n])."""
        mu, var = self.logit_fn(X)
        return disagreement_prob(mu, var, P_DRAWS, self.temperature, self.gen, self.sample)

    def _phi_from_p(self, p):
        """p: [B, m] disagreement probabilities -> phi for each batch [B]."""
        if self.statistic == "mean":
            return p.mean(1)
        return sample_max_of_K(poisson_binomial_pmf(p), self.K, self.rng)

    def _phi_literal(self, mu, var):
        """mu, var: [B, m, C] -> phi [B] by literally sampling (steps 2-6)."""
        B, m, C = mu.shape
        chunk = max(1, _MAX_ELEMS // (self.K * m * C))
        out = []
        for i in range(0, B, chunk):
            rates = disagreement_rates(mu[i:i + chunk], var[i:i + chunk], self.K,
                                       self.temperature, self.gen, self.sample)
            out.append(rates.max(-1).values.numpy() if self.statistic == "max"
                       else rates.mean(-1).numpy())
        return np.concatenate(out)

    # ---- phi for query batches ------------------------------------------
    @torch.no_grad()
    def score_many(self, Xs):
        """phi~ for R query batches at once. Xs: np.ndarray [R, m, d] -> [R]."""
        R, m, d = Xs.shape
        flat = Xs.reshape(R * m, d)
        if self.engine == "literal":
            mu, var = self.logit_fn(flat)
            return self._phi_literal(mu.reshape(R, m, -1), var.reshape(R, m, -1))
        return self._phi_from_p(self.point_probs(flat).reshape(R, m))

    def score(self, X):
        """phi~ for one query batch X (shape [m, d])."""
        return float(self.score_many(np.asarray(X)[None])[0])

    # ---- Calibrate ------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, X_val, m):
        """Bootstrap T batches of size m from the ID validation set -> Phi[m].

        Per-point quantities of the validation set are computed once; a
        bootstrap batch just picks rows (exactly equivalent, much faster).
        """
        idx = self.rng.integers(0, len(X_val), size=(self.T, m))   # [T, m]
        if self.engine == "literal":
            mu, var = self.logit_fn(X_val)
            t = torch.as_tensor(idx)
            self.Phi[m] = self._phi_literal(mu[t], var[t])
        else:
            if getattr(self, "_val_X", None) is not X_val:   # cache p_i of X_val
                self._val_X, self._val_p = X_val, self.point_probs(X_val)
            self.Phi[m] = self._phi_from_p(self._val_p[idx])
        return self.Phi[m]

    # ---- Deploy ---------------------------------------------------------
    def threshold(self, m, alpha):
        return float(np.quantile(self.Phi[m], 1.0 - alpha))

    def p_value(self, phi, m):
        """Randomised p-value of phi~ against Phi[m] (our proposed fix, NOT the paper).

        phi only takes values j/m, so many calibration rounds share the exact
        threshold value; the official rule 'phi >= quantile' then fires more
        often than alpha.  Breaking ties at random,
            p = (#{Phi > phi} + U * (#{Phi == phi} + 1)) / (T + 1),  U ~ U(0,1),
        gives P(p <= alpha) = alpha when phi~ is exchangeable with Phi.
        Alarm if p <= alpha.  Works on a scalar or an array of phi values.
        """
        Phi = self.Phi[m]
        phi = np.asarray(phi, dtype=float)
        gt = (Phi[None, :] > phi.reshape(-1, 1)).sum(1)
        eq = (Phi[None, :] == phi.reshape(-1, 1)).sum(1)
        u = self.rng.random(len(gt))
        return ((gt + u * (eq + 1)) / (len(Phi) + 1)).reshape(phi.shape)

    def test(self, X, alpha):
        """Returns (phi~, threshold, alarm). Requires calibrate(X_val, len(X))."""
        m = len(X)
        if m not in self.Phi:
            raise KeyError(f"calibrate for m={m} first")
        phi = self.score(X)
        thr = self.threshold(m, alpha)
        return phi, thr, bool(phi >= thr)
