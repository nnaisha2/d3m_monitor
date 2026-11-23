"""Feasibility experiments E1-E7 (called from run_spike.py and the notebook).

Every experiment takes a `Setup` (one trained model = one seed) and returns
plain dicts/lists of numbers, so results can be saved as JSON and plotted.

Detection rates are "fraction of R independent query batches that raised an
alarm".  Under a harmful shift that fraction is the true positive rate (TPR);
under no shift or a harmless shift it is the false positive rate (FPR).
Whether a condition is harmful is decided by its MEASURED population accuracy
(drop of more than HARM_DROP below the ID accuracy), not by its name.
"""
import io
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import baselines, data
from .d3m import D3M
from .model_vbll import train_vbll

M_GRID = (10, 25, 50, 100, 250, 500)
STRENGTHS = (0.25, 0.5, 0.75, 1.0)
ALPHAS = (0.01, 0.05, 0.1, 0.2)
D3M_DEFAULTS = dict(K=5000, T=1000, temperature=1.0)   # paper: K=5000, |Phi|=1000, tau=1 (UCI)
HARM_DROP = 0.05       # accuracy drop (absolute) that counts as deterioration
POP_N = 20000          # points used to measure the population accuracy of a condition


@dataclass
class Setup:
    seed: int
    model: object
    train_secs: float
    X_val: np.ndarray        # held-out ID inputs (D3M calibration, KS/BBSD reference)
    P_val: np.ndarray        # model probabilities on X_val (BBSD reference)
    id_acc: float            # expected accuracy on ID data


def build_setup(seed, **model_kw):
    torch.manual_seed(seed)
    X_tr, y_tr, X_val, _ = data.make_splits(seed)
    model, secs = train_vbll(X_tr, y_tr, seed, **model_kw)
    X_id = data.sample_id(POP_N, np.random.default_rng(seed + 10_000))
    return Setup(seed, model, secs, X_val, model.predict_proba(X_val),
                 data.expected_accuracy(X_id, model.predict(X_id)))


def make_monitor(setup, **kw):
    return D3M(setup.model.logit_dist, **{"seed": setup.seed, **D3M_DEFAULTS, **kw})


def population_accuracy(setup, kind, s):
    """Expected accuracy of the deployed model under shift (kind, s)."""
    rng = np.random.default_rng(setup.seed + 20_000)
    X = data.apply_shift(data.sample_id(POP_N, rng), kind, s)
    return data.expected_accuracy(X, setup.model.predict(X))


def draw_batches(kind, s, m, R, rng):
    """R query batches: inputs [R, m, 2] and sampled true labels [R, m]."""
    Xs = np.stack([data.apply_shift(data.sample_id(m, rng), kind, s) for _ in range(R)])
    ys = np.stack([data.sample_labels(X, rng) for X in Xs])
    return Xs, ys


def detect(setup, mon, Xs, ys, alpha, baselines_too=True):
    """Alarm rates of every detector on the batches Xs (mon must be calibrated)."""
    m = Xs.shape[1]
    phis = mon.score_many(Xs)
    thr = mon.threshold(m, alpha)
    out = {"d3m": float(np.mean(phis >= thr)), "phi_mean": float(phis.mean()),
           "threshold": thr,
           "d3m_tiefix": float(np.mean(mon.p_value(phis, m) <= alpha))}
    if baselines_too:
        ks, bb, orc = [], [], []
        for X, y in zip(Xs, ys):
            ks.append(baselines.ks_features(setup.X_val, X, alpha)[1])
            P = setup.model.predict_proba(X)
            bb.append(baselines.bbsd(setup.P_val, P, alpha)[1])
            orc.append(baselines.oracle(y, P.argmax(1), setup.id_acc, HARM_DROP)[1])
        out.update(ks=float(np.mean(ks)), bbsd=float(np.mean(bb)), oracle=float(np.mean(orc)))
    return out


# --------------------------------------------------------------------- E1
def e1_sanity(setup, temperature=1.0):
    """ID accuracy and how the VBLL posterior behaves on each shift (s = 1).

    For each condition: mean |logit gap| (how far from the model's boundary),
    mean posterior std of the gap (epistemic + learned noise), and the mean
    disagreement probability with and without categorical label sampling.
    """
    from .d3m import disagreement_prob
    out = {"id_acc": setup.id_acc, "train_secs": setup.train_secs, "conditions": {}}
    rng = np.random.default_rng(setup.seed + 1)
    for kind in data.SHIFT_KINDS:
        s = 0.0 if kind == "none" else 1.0
        X = data.apply_shift(data.sample_id(5000, rng), kind, s)
        mu, var = setup.model.logit_dist(X)
        out["conditions"][kind] = {
            "acc": population_accuracy(setup, kind, s),
            "mean_abs_gap": float((mu[:, 1] - mu[:, 0]).abs().mean()),
            "mean_gap_std": float(var.sum(1).sqrt().mean()),
            "p_dis_sampled": float(disagreement_prob(mu, var, 0, temperature, None, True).mean()),
            "p_dis_argmax": float(disagreement_prob(mu, var, 0, temperature, None, False).mean()),
        }
    return out


# --------------------------------------------------------------------- E2
def e2_no_shift(setup, mon, R=500):
    """FPR on fresh ID batches for every m and alpha.  Also how much of Phi
    sits exactly ON the threshold (ties: phi only takes values j/m)."""
    rng = np.random.default_rng(setup.seed + 2)
    out = {}
    for m in M_GRID:
        if m not in mon.Phi:
            mon.calibrate(setup.X_val, m)
        Xs, ys = draw_batches("none", 0.0, m, R, rng)
        phis = mon.score_many(Xs)
        row = {"distinct_phi_values": int(len(np.unique(mon.Phi[m])))}
        for a in ALPHAS:
            thr = mon.threshold(m, a)
            row[f"fpr_{a}"] = float(np.mean(phis >= thr))
            row[f"fpr_strict_{a}"] = float(np.mean(phis > thr))     # with '>' instead of '>='
            row[f"fpr_tiefix_{a}"] = float(np.mean(mon.p_value(phis, m) <= a))
            row[f"calib_mass_at_or_above_{a}"] = float(np.mean(mon.Phi[m] >= thr))
        if m == 100:   # baselines at the default m, for comparison
            for a in ALPHAS:
                r = detect(setup, mon, Xs[:200], ys[:200], a)
                row[f"ks_fpr_{a}"], row[f"bbsd_fpr_{a}"] = r["ks"], r["bbsd"]
        out[m] = row
    return out


# --------------------------------------------------------------------- E3
def e3_shift_strength(setup, mon, m=100, alpha=0.1, R=200):
    """Alarm rates vs shift strength for every shift kind (fixed m, alpha)."""
    rng = np.random.default_rng(setup.seed + 3)
    if m not in mon.Phi:
        mon.calibrate(setup.X_val, m)
    out = {}
    for kind in data.SHIFT_KINDS:
        strengths = (0.0,) if kind == "none" else STRENGTHS
        for s in strengths:
            Xs, ys = draw_batches(kind, s, m, R, rng)
            r = detect(setup, mon, Xs, ys, alpha)
            r["acc"] = population_accuracy(setup, kind, s)
            out[f"{kind}|{s}"] = r
    return out


# --------------------------------------------------------------------- E4
def e4_batch_size(setup, mon, s=0.75, alpha=0.1, R=200):
    """Alarm rates vs batch size m for a fixed shift strength."""
    rng = np.random.default_rng(setup.seed + 4)
    out = {}
    for m in M_GRID:
        if m not in mon.Phi:
            mon.calibrate(setup.X_val, m)
        for kind in data.SHIFT_KINDS:
            s_k = 0.0 if kind == "none" else s
            Xs, ys = draw_batches(kind, s_k, m, R, rng)
            r = detect(setup, mon, Xs, ys, alpha)
            r["acc"] = population_accuracy(setup, kind, s_k)
            out[f"{kind}|{m}"] = r
    return out


# --------------------------------------------------------------------- E5
E5_KINDS = ("none", "harmful_boundary", "harmful_extrapolation", "harmless_easier", "harmless_luck")


def _e5_batches(setup, m, s, R):
    rng = np.random.default_rng(setup.seed + 5)   # same batches for every variant
    return {k: draw_batches(k, 0.0 if k == "none" else s, m, R, rng) for k in E5_KINDS}


def _e5_rates(setup, mon, batches, m, alpha):
    mon.calibrate(setup.X_val, m)
    return {k: detect(setup, mon, Xs, ys, alpha, baselines_too=False)
            for k, (Xs, ys) in batches.items()}


def e5_sensitivity(setup, m=100, s=0.75, alpha=0.1, R=200):
    """D3M alarm rates when K, tau, the statistic, or label sampling change;
    and how stable the threshold is for different T."""
    batches = _e5_batches(setup, m, s, R)
    out = {"K": {}, "temperature": {}, "variant": {}, "T": {}}
    for K in (1, 10, 100, 1000, 5000, 50000):
        out["K"][K] = _e5_rates(setup, make_monitor(setup, K=K), batches, m, alpha)
    for tau in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        out["temperature"][tau] = _e5_rates(setup, make_monitor(setup, temperature=tau), batches, m, alpha)
    out["variant"]["paper (max, sampled labels)"] = out["K"][5000]
    out["variant"]["mean instead of max"] = _e5_rates(
        setup, make_monitor(setup, statistic="mean"), batches, m, alpha)
    out["variant"]["argmax instead of sampled labels"] = _e5_rates(
        setup, make_monitor(setup, sample=False), batches, m, alpha)
    for T in (20, 50, 100, 500, 1000):
        thrs, fprs = [], []
        Xs_id = batches["none"][0]
        for rep in range(20):   # 20 independent calibrations
            mon = make_monitor(setup, T=T, seed=setup.seed * 100 + rep)
            mon.calibrate(setup.X_val, m)
            thrs.append(mon.threshold(m, alpha))
            fprs.append(float(np.mean(mon.score_many(Xs_id) >= thrs[-1])))
        out["T"][T] = {"thr_mean": float(np.mean(thrs)), "thr_std": float(np.std(thrs)),
                       "fpr_mean": float(np.mean(fprs)), "fpr_std": float(np.std(fprs))}
    return out


def e5_reg_factor(base, factors=(1.0, 100.0, 1000.0), m=100, s=0.75, alpha=0.1, R=200):
    """Retrain with a different VBLL regularization factor (KL weight x n).
    `base` is the default setup (factor 100), reused instead of retrained."""
    out = {}
    for f in factors:
        st = base if f == 100.0 else build_setup(base.seed, reg_factor=f)
        r = _e5_rates(st, make_monitor(st), _e5_batches(st, m, s, R), m, alpha)
        r["id_acc"] = st.id_acc
        out[f] = r
    return out


# --------------------------------------------------------------------- E6
def e6_timing(setup, panel_fn=None):
    """Wall-clock cost of the pieces of one live-demo interaction."""
    out = {"train_secs": setup.train_secs, "calibrate_secs": {}}
    mon = make_monitor(setup)
    for m in M_GRID:
        t0 = time.perf_counter()
        mon.calibrate(setup.X_val, m)
        out["calibrate_secs"][m] = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    X, y = data.sample_query(500, "harmful_boundary", 0.5, rng)
    phi, thr, alarm = mon.test(X, 0.1)
    baselines.ks_features(setup.X_val, X, 0.1)
    baselines.bbsd(setup.P_val, setup.model.predict_proba(X), 0.1)
    population_accuracy(setup, "harmful_boundary", 0.5)
    out["interaction_compute_secs_m500"] = time.perf_counter() - t0
    if panel_fn is not None:
        import matplotlib.pyplot as plt
        t0 = time.perf_counter()
        fig = panel_fn(setup, mon, "harmful_boundary", 0.5, 500, 0.1, seed=0)
        fig.savefig(io.BytesIO(), format="png", dpi=100)
        plt.close(fig)
        out["interaction_with_plot_secs_m500"] = time.perf_counter() - t0
    lit = make_monitor(setup, engine="literal", T=50)
    t0 = time.perf_counter()
    lit.calibrate(setup.X_val, 100)
    out["literal_calibrate_T50_m100_secs"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    lit.score(X)
    out["literal_score_m500_secs"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------- E7
TIMELINES = {
    "A: easier cases, then borderline cases": (("harmless_easier", 20), ("harmful_boundary", 40)),
    "B: within-support drift, then extrapolation": (("harmless_support", 20), ("harmful_extrapolation", 40)),
}


def timeline_schedule(name, days=60):
    """(kind, s) for every day: no shift until the first change, then each
    shift ramps from s = 1/20 to s = 1 over 20 days."""
    (k1, d1), (k2, d2) = TIMELINES[name]
    sched = []
    for d in range(days):
        if d < d1:
            sched.append(("none", 0.0))
        elif d < d2:
            sched.append((k1, (d - d1 + 1) / (d2 - d1)))
        else:
            sched.append((k2, min(1.0, (d - d2 + 1) / 20)))
    return sched


def e7_timeline(setup, mon, name, m=100, alpha=0.1):
    """One batch of m points per day.  Returns per-day accuracy and alarms,
    plus false alarms (days without deterioration) and detection delay
    (days from deterioration onset to the first alarm)."""
    if m not in mon.Phi:
        mon.calibrate(setup.X_val, m)
    rng = np.random.default_rng(setup.seed + 7)
    days = []
    for kind, s in timeline_schedule(name):
        X, y = data.sample_query(m, kind, s, rng)
        P = setup.model.predict_proba(X)
        days.append({
            "kind": kind, "s": s, "acc": population_accuracy(setup, kind, s),
            "d3m": mon.test(X, alpha)[2],
            "ks": baselines.ks_features(setup.X_val, X, alpha)[1],
            "bbsd": baselines.bbsd(setup.P_val, P, alpha)[1],
            "oracle": baselines.oracle(y, P.argmax(1), setup.id_acc, HARM_DROP)[1],
        })
    harmful = [d["acc"] < setup.id_acc - HARM_DROP for d in days]
    onset = harmful.index(True) if any(harmful) else None
    summary = {}
    for det in ("d3m", "ks", "bbsd", "oracle"):
        false_alarms = sum(d[det] and not h for d, h in zip(days, harmful))
        hits = [i for i, (d, h) in enumerate(zip(days, harmful)) if d[det] and h]
        summary[det] = {"false_alarms": false_alarms,
                        "n_ok_days": int(sum(not h for h in harmful)),
                        "delay": (hits[0] - onset) if (hits and onset is not None) else None,
                        "hit_rate_after_onset": float(np.mean([d[det] for d, h in zip(days, harmful) if h])) if onset is not None else None}
    return {"days": days, "onset": onset, "summary": summary}


# --------------------------------------------------------------------- live Panel B
def batch_size_sweep(setup, mon, kinds, s, alpha=0.1, R=100, m_grid=M_GRID, seed=0):
    """Panel B in the notebook: alarm rates vs m for a few shift kinds.
    Returns a dict in the same format as e4 (plus '_id_acc') for plots.fig_rates."""
    rng = np.random.default_rng(seed)
    out = {"_id_acc": setup.id_acc}
    for m in m_grid:
        if m not in mon.Phi:
            mon.calibrate(setup.X_val, m)
        for kind in kinds:
            s_k = 0.0 if kind == "none" else s
            Xs, ys = draw_batches(kind, s_k, m, R, rng)
            r = detect(setup, mon, Xs, ys, alpha)
            r["acc"] = population_accuracy(setup, kind, s_k)
            out[f"{kind}|{m}"] = r
    return out


def load_or_build_setup(seed, ckpt_path=None):
    """Notebook start-up: load the trained model saved by run_spike.py if it
    exists (instant), otherwise train it (~15-60 s on a laptop CPU)."""
    import os
    from .model_vbll import VBLLNet
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return build_setup(seed)
    X_tr, _, X_val, _ = data.make_splits(seed)
    model = VBLLNet(n_train=len(X_tr))
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    X_id = data.sample_id(POP_N, np.random.default_rng(seed + 10_000))
    return Setup(seed, model, 0.0, X_val, model.predict_proba(X_val),
                 data.expected_accuracy(X_id, model.predict(X_id)))
