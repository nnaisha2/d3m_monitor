"""Label-free drift detectors that react to ANY change, plus a label-using oracle.

All detectors compare the query batch with a reference sample from the
training distribution (we use the held-out ID validation set).  Note that this
means they need training-distribution data at deployment time; D3M does not.
"""
import numpy as np
from scipy.stats import ks_2samp


def ks_features(X_ref, X, alpha):
    """Per-feature two-sample Kolmogorov-Smirnov test with Bonferroni correction.

    One KS test per input feature; alarm if any p-value < alpha / d.
    Returns (smallest p-value, alarm).
    """
    d = X.shape[1]
    pvals = [ks_2samp(X_ref[:, j], X[:, j]).pvalue for j in range(d)]
    p_min = float(min(pvals))
    return p_min, p_min < alpha / d


def bbsd(P_ref, P, alpha):
    """Black-Box Shift Detection (Lipton et al. 2018; Rabanser et al. 2019).

    KS test on the model's predicted class probabilities instead of the raw
    inputs.  For C = 2 the two columns carry the same information, so we test
    only P[:, 1]; for C > 2 we test every column with Bonferroni.
    Returns (smallest p-value, alarm).
    """
    C = P.shape[1]
    cols = [1] if C == 2 else list(range(C))
    pvals = [ks_2samp(P_ref[:, c], P[:, c]).pvalue for c in cols]
    p_min = float(min(pvals))
    return p_min, p_min < alpha / len(cols)


def oracle(y_true, y_pred, id_acc, drop=0.05):
    """Uses the TRUE labels (unavailable in practice): alarm if the batch
    accuracy is more than `drop` below the ID accuracy.  Ground-truth reference
    only.  Returns (batch accuracy, alarm)."""
    acc = float(np.mean(y_true == y_pred))
    return acc, acc < id_acc - drop
