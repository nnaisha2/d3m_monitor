"""Synthetic 2-D classification problem with a KNOWN true labelling rule.

Why synthetic?  We need the *true* accuracy of the deployed model on any shifted
batch, so we can check whether a shift is really harmful.  With a known
labelling rule we can compute that accuracy exactly.

Geometry (all numbers are module constants so you can quote them in the Q&A):

* Inputs during training:  x = (x1, x2) ~ N(0, I)   (standard 2-D Gaussian).
* True boundary:           x2 = b(x1),  b(x1) = BEND * max(0, x1 - BEND_START)^2.
  Flat line x2 = 0 for x1 <= 2, bending upward (a parabola) for x1 > 2.
  Only ~2% of training points have x1 > 2, so the model never really sees
  the bend: it learns "x2 = 0" and extrapolates it.
* True labelling rule:     P(y = 1 | x) = sigmoid( (x2 - b(x1)) / NOISE_W ).
  Far from the boundary the label is certain; within ~NOISE_W of it the
  label is partly random (borderline cases).  The rule is fixed and known,
  so the true accuracy of any model on any batch can be computed exactly:
      acc(x) = P(y = f(x) | x).
  (A fully deterministic rule makes the model confident almost everywhere,
  see FEASIBILITY_REPORT.md, section 1.)

Deployment shifts (strength s in [0, 1]; s = 0 means no shift).  Only the
inputs change; the labelling rule stays the same (covariate shift, the
setting of the paper's theory).

* "harmful_boundary":      x2 -> (1 - 0.85 s) x2.  More borderline cases near
                           the boundary -> true accuracy drops.
* "harmful_extrapolation": x1 += 4 s.  The batch slides right into the bend,
                           where the flat extrapolation is CONFIDENTLY wrong.
* "harmless_easier":       x2 += 2.5 s * sign(x2).  Points move AWAY from the
                           boundary (clear-cut cases) -> accuracy goes up.
* "harmless_support":      x1 -> (1 - 0.5 s) x1 - s.  The batch concentrates
                           on the left half of the training region.
* "harmless_luck":         x1 -= 4 s.  The batch slides left, OUTSIDE the
                           training data, where the flat extrapolation happens
                           to be right.  Nothing in the unlabelled data tells
                           you that: a stress test for any label-free monitor.
"""
import numpy as np

BEND_START = 2.0   # x1 where the true boundary starts to bend
BEND = 1.0         # curvature of the bend
NOISE_W = 0.3      # width of the noisy band around the true boundary

SHIFT_KINDS = ("none", "harmful_boundary", "harmful_extrapolation",
               "harmless_easier", "harmless_support", "harmless_luck")


def true_boundary(x1):
    """Height of the true decision boundary b(x1)."""
    return BEND * np.clip(x1 - BEND_START, 0.0, None) ** 2


def true_prob(X):
    """P(y = 1 | x) under the known labelling rule."""
    return 1.0 / (1.0 + np.exp(-(X[:, 1] - true_boundary(X[:, 0])) / NOISE_W))


def sample_labels(X, rng):
    """Draw labels y ~ Bernoulli(P(y = 1 | x))."""
    return (rng.random(len(X)) < true_prob(X)).astype(np.int64)


def expected_accuracy(X, y_pred):
    """Exact expected accuracy of predictions y_pred on inputs X:
    mean over points of P(y = y_pred | x).  No label noise in this number."""
    p1 = true_prob(X)
    return float(np.mean(np.where(y_pred == 1, p1, 1.0 - p1)))


def sample_id(n, rng):
    """n in-distribution inputs, x ~ N(0, I)."""
    return rng.normal(size=(n, 2))


def apply_shift(X, kind, s):
    """Return a shifted copy of X. `kind` is one of SHIFT_KINDS, s in [0, 1]."""
    X = X.copy()
    if kind == "none" or s == 0:
        return X
    if kind == "harmful_boundary":
        X[:, 1] *= 1.0 - 0.85 * s
    elif kind == "harmful_extrapolation":
        X[:, 0] += 4.0 * s
    elif kind == "harmless_easier":
        X[:, 1] += 2.5 * s * np.sign(X[:, 1])
    elif kind == "harmless_support":
        X[:, 0] = (1.0 - 0.5 * s) * X[:, 0] - 1.0 * s
    elif kind == "harmless_luck":
        X[:, 0] -= 4.0 * s
    else:
        raise ValueError(f"unknown shift kind {kind!r}; choose from {SHIFT_KINDS}")
    return X


def sample_query(m, kind, s, rng):
    """One deployment batch of m points: inputs AND sampled true labels.

    The monitor only ever sees X; y is used for the oracle only.
    """
    X = apply_shift(sample_id(m, rng), kind, s)
    return X, sample_labels(X, rng)


def make_splits(seed, n_train=2000, n_val=2000):
    """Training set (labelled) and held-out ID validation set (calibration).

    The validation inputs are what D3M bootstraps from during Calibrate, and
    what the KS / BBSD baselines use as their reference sample.
    """
    rng = np.random.default_rng(seed)
    X_tr = sample_id(n_train, rng)
    X_val = sample_id(n_val, rng)
    return X_tr, sample_labels(X_tr, rng), X_val, sample_labels(X_val, rng)
