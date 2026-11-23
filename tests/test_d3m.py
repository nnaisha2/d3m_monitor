"""Fast checks (no long training).  Run:  python -m pytest -q tests

Most tests use a tiny hand-made "model" whose logit posterior is known:
    mu(x) = [0, 3 * x2],  var(x) = 0.3 + 0.05 * x1^2
so the decision boundary is x2 = 0 and the uncertainty grows with |x1|.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import baselines, data  # noqa: E402
from src.d3m import (D3M, _disagreement_prob_binary, disagreement_prob,  # noqa: E402
                     disagreement_rates, poisson_binomial_pmf, sample_max_of_K)


def toy_logit_fn(X):
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    mu = torch.stack([torch.zeros(len(X)), 3.0 * X[:, 1]], 1)
    v = (0.3 + 0.05 * X[:, 0] ** 2).unsqueeze(1).expand(-1, 2).contiguous()
    return mu, v


# ------------------------------------------------------------ shapes
def test_shapes():
    rng = np.random.default_rng(0)
    X_tr, y_tr, X_val, y_val = data.make_splits(0, n_train=50, n_val=40)
    assert X_tr.shape == (50, 2) and y_tr.shape == (50,) and X_val.shape == (40, 2)
    for kind in data.SHIFT_KINDS:
        X, y = data.sample_query(7, kind, 0.5, rng)
        assert X.shape == (7, 2) and y.shape == (7,)
    mu, var = toy_logit_fn(X_val)
    g = torch.Generator().manual_seed(0)
    assert disagreement_rates(mu[None], var[None], 11, 1.0, g).shape == (1, 11)
    assert disagreement_prob(mu, var, 0, 1.0, g).shape == (40,)
    mon = D3M(toy_logit_fn, K=50, T=30, seed=0)
    assert mon.calibrate(X_val, 10).shape == (30,)
    assert mon.score_many(np.stack([X_val[:10]] * 4)).shape == (4,)


# ------------------------------------------------------------ the exact engine is exact
def test_poisson_binomial_matches_brute_force():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=6)
    pmf = poisson_binomial_pmf(p[None])[0]
    flips = rng.random((200_000, 6)) < p
    emp = np.bincount(flips.sum(1), minlength=7) / 200_000
    assert abs(pmf.sum() - 1) < 1e-12
    assert np.max(np.abs(pmf - emp)) < 0.005


def test_max_sampler_matches_brute_force():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 0.8, size=15)
    K, R = 50, 20_000
    brute = (rng.random((R, K, 15)) < p).mean(2).max(1)
    exact = sample_max_of_K(poisson_binomial_pmf(np.tile(p, (R, 1))), K, rng)
    assert abs(brute.mean() - exact.mean()) < 0.003
    assert abs(np.quantile(brute, 0.9) - np.quantile(exact, 0.9)) < 1e-9 + 1 / 15


def test_quadrature_matches_monte_carlo():
    X = data.sample_id(200, np.random.default_rng(3)) * 2
    mu, var = toy_logit_fn(X)
    g = torch.Generator().manual_seed(0)
    mu3 = torch.cat([mu, torch.full((200, 1), -1e4)], 1)     # 3rd class never wins
    var3 = torch.cat([var, torch.full((200, 1), 1e-8)], 1)   # -> forces the MC path
    for tau, sample in [(1.0, True), (3.0, True), (1.0, False)]:
        q = _disagreement_prob_binary(mu, var, tau, sample)
        mc = disagreement_prob(mu3, var3, 100_000, tau, g, sample)
        assert np.max(np.abs(q - mc)) < 0.01


def test_literal_and_exact_engines_agree():
    X_val = data.sample_id(500, np.random.default_rng(4))
    lit = D3M(toy_logit_fn, K=200, T=1500, seed=1, engine="literal").calibrate(X_val, 20)
    exa = D3M(toy_logit_fn, K=200, T=1500, seed=2, engine="exact").calibrate(X_val, 20)
    vals = np.union1d(lit, exa)
    f_lit = np.array([np.mean(lit == v) for v in vals])
    f_exa = np.array([np.mean(exa == v) for v in vals])
    assert abs(lit.mean() - exa.mean()) < 0.01
    assert np.max(np.abs(f_lit - f_exa)) < 0.05


# ------------------------------------------------------------ calibration
def test_fpr_on_id_close_to_alpha():
    """With random tie-breaking the FPR on fresh ID batches must be ~alpha.
    The official '>=' rule is allowed to exceed alpha (ties), never to be far below."""
    rng = np.random.default_rng(5)
    X_val = data.sample_id(3000, rng)
    m, alpha, R = 50, 0.1, 1500
    mon = D3M(toy_logit_fn, K=1000, T=2000, seed=0)
    mon.calibrate(X_val, m)
    phis = mon.score_many(np.stack([data.sample_id(m, rng) for _ in range(R)]))
    fpr_fix = np.mean(mon.p_value(phis, m) <= alpha)
    fpr_official = np.mean(phis >= mon.threshold(m, alpha))
    tol = 3 * np.sqrt(alpha * (1 - alpha) / R) + 0.02   # sampling noise + finite val set
    assert abs(fpr_fix - alpha) < tol
    assert fpr_official >= alpha - tol


def test_ks_fpr_close_to_alpha():
    rng = np.random.default_rng(6)
    X_ref = data.sample_id(2000, rng)
    alarms = [baselines.ks_features(X_ref, data.sample_id(100, rng), 0.1)[1] for _ in range(600)]
    assert np.mean(alarms) < 0.1 + 3 * np.sqrt(0.09 / 600)   # Bonferroni: at most alpha


# ------------------------------------------------------------ the shifts do what they claim
@pytest.mark.parametrize("kind,harmful", [
    ("harmful_boundary", True), ("harmful_extrapolation", True),
    ("harmless_easier", False), ("harmless_support", False), ("harmless_luck", False)])
def test_shift_labels_with_flat_classifier(kind, harmful):
    """The model learns ~ the flat rule f(x) = 1[x2 > 0]; check the shift's
    effect on ITS accuracy empirically (not by assumption)."""
    rng = np.random.default_rng(7)
    X0 = data.sample_id(20000, rng)
    flat = lambda X: (X[:, 1] > 0).astype(int)
    acc0 = data.expected_accuracy(X0, flat(X0))
    X1 = data.apply_shift(X0, kind, 1.0)
    drop = acc0 - data.expected_accuracy(X1, flat(X1))
    assert (drop > 0.05) == harmful


# ------------------------------------------------------------ determinism
def test_determinism_fixed_seed():
    X_val = data.sample_id(300, np.random.default_rng(8))
    a = D3M(toy_logit_fn, K=100, T=50, seed=3).calibrate(X_val, 25)
    b = D3M(toy_logit_fn, K=100, T=50, seed=3).calibrate(X_val, 25)
    assert np.array_equal(a, b)
    assert np.array_equal(data.make_splits(9)[0], data.make_splits(9)[0])


def test_training_is_deterministic():
    from src.model_vbll import train_vbll
    X, y, X_val, _ = data.make_splits(0, n_train=200, n_val=50)
    m1, _ = train_vbll(X, y, seed=0, epochs=2)
    m2, _ = train_vbll(X, y, seed=0, epochs=2)
    assert torch.equal(m1.logit_dist(X_val)[0], m2.logit_dist(X_val)[0])
