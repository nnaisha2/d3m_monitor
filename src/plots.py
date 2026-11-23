"""All figures (matplotlib).  Colours follow one fixed scheme everywhere:

    D3M = blue (circles), KS = orange (squares), BBSD = aqua (triangles),
    oracle (uses true labels) = grey (x).   Thresholds/references are dashed.

Every figure with several series has a legend, and the series also differ by
marker shape, so they can be told apart without colour.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import data

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DET = {  # detector -> (label, colour, marker)
    "d3m": ("D3M", "#2a78d6", "o"),
    "ks": ("KS on inputs", "#eb6834", "s"),
    "bbsd": ("BBSD", "#1baf7a", "^"),
    "oracle": ("oracle (true labels)", MUTED, "x"),
}
QUERY = "#eb6834"   # deployment batch points in the scatter
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
KIND_LABEL = {
    "none": "no shift",
    "harmful_boundary": "harmful: borderline cases",
    "harmful_extrapolation": "harmful: extrapolation (bend)",
    "harmless_easier": "harmless: easier cases",
    "harmless_support": "harmless: within training region",
    "harmless_luck": "harmless: extrapolation, lucky",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
})


def _det_line(ax, x, y, det, err=None):
    label, color, marker = DET[det]
    ax.plot(x, y, color=color, marker=marker, label=label, ms=5.5,
            ls="-" if det != "oracle" else ":", lw=2 if det != "oracle" else 1.5)
    if err is not None:
        ax.fill_between(x, np.clip(y - err, 0, 1), np.clip(y + err, 0, 1), color=color, alpha=0.12, lw=0)


# ------------------------------------------------------------ grid helpers
def _grid(lim=(-6, 6.5, -4, 5), n=160):
    x1 = np.linspace(lim[0], lim[1], n)
    x2 = np.linspace(lim[2], lim[3], n)
    G1, G2 = np.meshgrid(x1, x2)
    return G1, G2, np.c_[G1.ravel(), G2.ravel()]


def disagreement_map(ax, model, temperature=1.0, lim=(-6, 6.5, -4, 5)):
    """Background: probability that one D3M draw disagrees with the deployed
    model at x (p(x)); plus the true and the model's decision boundaries."""
    from .d3m import disagreement_prob
    G1, G2, P = _grid(lim)
    mu, var = model.logit_dist(P)
    p = disagreement_prob(mu, var, 0, temperature, None, True).reshape(G1.shape)
    im = ax.pcolormesh(G1, G2, p, cmap=BLUES, vmin=0, vmax=0.5, shading="auto", rasterized=True)
    gap = (mu[:, 1] - mu[:, 0]).numpy().reshape(G1.shape)
    ax.contour(G1, G2, gap, levels=[0], colors=INK2, linewidths=1.5, linestyles="--")
    xs = np.linspace(lim[0], lim[1], 300)
    ax.plot(xs, data.true_boundary(xs), color=INK, lw=1.8)
    ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(False)
    return im


def _boundary_legend(ax):
    ax.plot([], [], color=INK, lw=1.8, label="true boundary")
    ax.plot([], [], color=INK2, lw=1.5, ls="--", label="model boundary")


# ------------------------------------------------------------ Panel A
def panel_a(setup, mon, kind, s, m, alpha, seed=0, temperature=None):
    """The live-demo figure: data + boundaries + query batch | Phi histogram."""
    from . import baselines
    from .experiments import population_accuracy, HARM_DROP
    rng = np.random.default_rng(seed)
    X, y = data.sample_query(m, kind, s, rng)
    if m not in mon.Phi:
        mon.calibrate(setup.X_val, m)
    phi, thr, alarm = mon.test(X, alpha)
    P = setup.model.predict_proba(X)
    ks = baselines.ks_features(setup.X_val, X, alpha)[1]
    bb = baselines.bbsd(setup.P_val, P, alpha)[1]
    acc = population_accuracy(setup, kind, s)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    im = disagreement_map(a1, setup.model, temperature or mon.temperature)
    tr = setup.X_val[:300]
    a1.scatter(tr[:, 0], tr[:, 1], s=4, color=MUTED, alpha=0.6, lw=0, label="training-like data")
    wrong = P.argmax(1) != (data.true_prob(X) > 0.5)
    a1.scatter(X[~wrong, 0], X[~wrong, 1], s=16, color=QUERY, edgecolor=SURFACE, lw=0.6,
               label="query batch (model right)")
    a1.scatter(X[wrong, 0], X[wrong, 1], s=26, color=QUERY, marker="x", lw=1.4,
               label="query batch (model wrong)")
    _boundary_legend(a1)
    a1.legend(loc="lower left", fontsize=7.5, ncol=2)
    cb = fig.colorbar(im, ax=a1, fraction=0.04, pad=0.02)
    cb.set_label("D3M disagreement prob. p(x)", color=INK2); cb.outline.set_visible(False)
    a1.set_title(f"{KIND_LABEL[kind]}, strength s = {s:g}")

    Phi = mon.Phi[m]
    bins = np.arange(Phi.min() - 0.5 / m, max(Phi.max(), phi) + 1.5 / m, 1.0 / m)
    a2.hist(Phi, bins=bins, color="#86b6ef", edgecolor=SURFACE, lw=0.8,
            label=f"Phi: ID calibration ({len(Phi)} rounds)")
    a2.axvline(thr, color=INK, ls="--", lw=1.5, label=f"threshold = {1 - alpha:g}-quantile = {thr:.3f}")
    a2.axvline(phi, color=QUERY, lw=2.5, label=f"query phi~ = {phi:.3f}")
    a2.set_xlabel(f"maximum disagreement rate phi  (m = {m}, K = {mon.K})")
    a2.set_ylabel("count"); a2.legend(fontsize=7.5, loc="upper left")
    harmful = acc < setup.id_acc - HARM_DROP
    verdict = lambda b: "ALARM" if b else "quiet"
    a2.set_title(f"true accuracy {acc:.3f} (ID {setup.id_acc:.3f}) -> {'HARMFUL' if harmful else 'not harmful'}\n"
                 f"D3M: {verdict(alarm)}    KS: {verdict(ks)}    BBSD: {verdict(bb)}   (alpha = {alpha:g})")
    fig.tight_layout()
    fig.info = dict(phi=phi, threshold=thr, d3m=alarm, ks=ks, bbsd=bb, acc=acc, id_acc=setup.id_acc,
                    p_value_tiefix=float(mon.p_value(phi, m)),
                    batch_acc=float(np.mean(P.argmax(1) == y)))
    return fig


# ------------------------------------------------------------ E1
def fig_e1(setup, temperature=1.0):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = disagreement_map(ax, setup.model, temperature)
    tr = setup.X_val[:400]
    ax.scatter(tr[:, 0], tr[:, 1], s=4, color=MUTED, alpha=0.7, lw=0)
    # label each shift (s = 1) next to where it moves the data
    spots = {"harmful_boundary": (0.0, -1.3), "harmful_extrapolation": (4.6, -1.6),
             "harmless_easier": (0.0, 3.4), "harmless_support": (-1.3, 2.0),
             "harmless_luck": (-4.2, 1.6)}
    for kind, (cx, cy) in spots.items():
        ax.annotate(KIND_LABEL[kind].replace(": ", ":\n"), (cx, cy), fontsize=7, color=INK,
                    ha="center", bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=GRID, alpha=0.9))
    _boundary_legend(ax)
    ax.legend(loc="lower right", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("p(x) = P(one D3M draw disagrees with model)", color=INK2); cb.outline.set_visible(False)
    ax.set_title(f"Where D3M sees disagreement (seed {setup.seed}, tau = {temperature:g}); "
                 "labels mark where each shift (s = 1) moves the data")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ helpers for seed aggregation
def _agg(results, key_fn):
    """Mean and std over seeds of key_fn(seed_result)."""
    v = np.array([key_fn(r) for r in results], dtype=float)
    return v.mean(0), v.std(0)


# ------------------------------------------------------------ E2
def fig_e2(e2_list, m_grid, alphas):
    fig, axes = plt.subplots(1, len(alphas), figsize=(12, 3.2), sharey=False)
    x = np.array(m_grid)
    for ax, a in zip(axes, alphas):
        mean, sd = _agg(e2_list, lambda r: [r[m][f"fpr_{a}"] for m in m_grid])
        _det_line(ax, x, mean, "d3m", sd)
        ax.lines[-1].set_label("D3M, official rule (phi >= threshold)")
        mean2, _ = _agg(e2_list, lambda r: [r[m][f"fpr_strict_{a}"] for m in m_grid])
        ax.plot(x, mean2, color=MUTED, marker="o", mfc=SURFACE, lw=1.5, label="D3M with '>' instead of '>='")
        mean3, _ = _agg(e2_list, lambda r: [r[m][f"fpr_tiefix_{a}"] for m in m_grid])
        ax.plot(x, mean3, color=INK2, marker="D", ms=4.5, lw=1.5, label="D3M, random tie-break (our fix)")
        ax.axhline(a, color=INK, ls="--", lw=1.2, label="target = alpha")
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels(x)
        ax.set_title(f"alpha = {a:g}"); ax.set_xlabel("batch size m")
        ax.set_ylim(0, max(0.05, 3 * a))
    axes[0].set_ylabel("false positive rate (no shift)")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("E2: is D3M calibrated on fresh ID batches?  (mean over seeds; band = +/- 1 sd)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ E3 / E4
def fig_rates(res_list, kinds, xs, key, xlabel, title, alpha=None, logx=False, show_acc=True):
    n = len(kinds)
    rows = 2 if show_acc else 1
    fig, axes = plt.subplots(rows, n, figsize=(2.55 * n, 2.6 * rows + 0.6), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1] if show_acc else [1]})
    for j, kind in enumerate(kinds):
        ax = axes[0, j]
        for det in ("d3m", "ks", "bbsd", "oracle"):
            mean, sd = _agg(res_list, lambda r: [r[key(kind, x)][det] for x in xs])
            _det_line(ax, np.array(xs), mean, det, sd if det == "d3m" else None)
        if alpha is not None:
            ax.axhline(alpha, color=INK, ls="--", lw=1, label="alpha")
        ax.set_ylim(-0.03, 1.03); ax.set_title(KIND_LABEL[kind], fontsize=8.5)
        if logx:
            ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(xs, fontsize=7)
        if j == 0:
            ax.set_ylabel("alarm rate")
        if show_acc:
            axa = axes[1, j]
            mean, sd = _agg(res_list, lambda r: [r[key(kind, x)]["acc"] for x in xs])
            axa.plot(xs, mean, color=INK2, marker="o", ms=4, lw=1.5)
            idm = np.mean([r["_id_acc"] for r in res_list])
            axa.axhline(idm, color=MUTED, ls="--", lw=1)
            axa.axhline(idm - 0.05, color="#e34948", ls="--", lw=1)
            axa.set_ylim(0.5, 1.02); axa.set_xlabel(xlabel)
            if logx:
                axa.set_xscale("log"); axa.set_xticks(xs); axa.set_xticklabels(xs, fontsize=7)
            if j == 0:
                axa.set_ylabel("true accuracy")
                axa.plot([], [], color=MUTED, ls="--", lw=1, label="ID accuracy")
                axa.plot([], [], color="#e34948", ls="--", lw=1, label="ID - 5 pts (harm line)")
                axa.legend(fontsize=6.5, loc="lower left")
        else:
            ax.set_xlabel(xlabel)
    axes[0, 0].legend(fontsize=7, loc="center left")
    fig.suptitle(title, color=INK, fontsize=10)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ E5
def fig_e5(e5_list, kinds):
    params = [("K", "posterior draws K", True), ("temperature", "temperature tau", True)]
    fig, axes = plt.subplots(3, len(kinds), figsize=(2.55 * len(kinds), 7.4), squeeze=False)
    _, color, marker = DET["d3m"]
    for i, (p, xlabel, logx) in enumerate(params):
        xs = sorted(e5_list[0][p].keys(), key=float)
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            mean, sd = _agg(e5_list, lambda r: [r[p][x][kind]["d3m"] for x in xs])
            xv = np.array(xs, dtype=float)
            ax.plot(xv, mean, color=color, marker=marker)
            ax.fill_between(xv, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=color, alpha=0.12, lw=0)
            ax.axhline(0.1, color=INK, ls="--", lw=1)
            if logx:
                ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03); ax.set_xlabel(xlabel, fontsize=8)
            if i == 0:
                ax.set_title(KIND_LABEL[kind], fontsize=8.5)
            if j == 0:
                ax.set_ylabel("D3M alarm rate")
    variants = list(e5_list[0]["variant"].keys())
    short = ["paper", "mean", "argmax"]
    for j, kind in enumerate(kinds):
        ax = axes[2, j]
        mean, sd = _agg(e5_list, lambda r: [r["variant"][v][kind]["d3m"] for v in variants])
        ax.bar(range(len(variants)), mean, yerr=sd, color=color, width=0.6,
               error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        for k, v in enumerate(mean):
            ax.text(k + 0.33, min(v + 0.03, 0.95), f"{v:.2f}", ha="left", fontsize=7, color=INK)
        ax.axhline(0.1, color=INK, ls="--", lw=1)
        ax.set_xticks(range(len(variants))); ax.set_xticklabels(short, fontsize=7.5)
        ax.set_ylim(0, 1.05)
        if j == 0:
            ax.set_ylabel("D3M alarm rate")
    fig.suptitle("E5: D3M sensitivity (m = 100, alpha = 0.1 dashed, s = 0.75; mean over seeds, band = +/- 1 sd)\n"
                 "bottom row: paper = max over K sampled labels; mean = expected disagreement (no max); "
                 "argmax = no label sampling", color=INK, fontsize=9.5)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ E7
def fig_e7(timelines, id_acc):
    """timelines: {name: e7 result for one seed}."""
    names = list(timelines)
    fig, axes = plt.subplots(2, len(names), figsize=(6.2 * len(names), 4.6), squeeze=False,
                             gridspec_kw={"height_ratios": [1.3, 1]}, sharex=True)
    for j, name in enumerate(names):
        res = timelines[name]
        days = res["days"]
        d = np.arange(len(days))
        ax = axes[0, j]
        ax.plot(d, [x["acc"] for x in days], color=INK2, lw=1.8)
        ax.axhline(id_acc, color=MUTED, ls="--", lw=1)
        ax.axhline(id_acc - 0.05, color="#e34948", ls="--", lw=1)
        ax.set_ylim(0.5, 1.02); ax.set_title(name)
        if j == 0:
            ax.set_ylabel("true accuracy")
        axr = axes[1, j]
        for i, det in enumerate(("d3m", "ks", "bbsd", "oracle")):
            label, color, marker = DET[det]
            hit = [k for k, x in enumerate(days) if x[det]]
            axr.scatter(hit, np.full(len(hit), -i), color=color, marker=marker, s=22, label=label)
        axr.set_yticks([0, -1, -2, -3]); axr.set_yticklabels(["D3M", "KS", "BBSD", "oracle"])
        axr.set_ylim(-3.6, 0.6); axr.set_xlabel("day (one batch of m = 100 per day)")
        for a in (ax, axr):
            if res["onset"] is not None:
                a.axvspan(res["onset"] - 0.5, len(days) - 0.5, color="#e34948", alpha=0.06, lw=0)
            for b in (20, 40):
                a.axvline(b - 0.5, color=MUTED, lw=0.8)
    axes[1, 0].set_ylabel("alarm days")
    fig.suptitle("E7: monitoring timeline (seed 0). Days 0-19 no shift, 20-39 first drift, 40-59 second drift. "
                 "Red band = true accuracy more than 5 pts below ID.", color=INK, fontsize=9.5)
    fig.tight_layout()
    return fig
