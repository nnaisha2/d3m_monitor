"""Run all feasibility experiments (E1-E7), time them, save JSON + figures.

    python run_spike.py            # full run: 5 seeds (~see FEASIBILITY_REPORT.md)
    python run_spike.py --quick    # smoke test: 2 seeds, few repetitions

Outputs: results/spike_results.json, results/figures/*.png,
         results/checkpoints/model_seed0.pt (used by the notebook; git-ignored).
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import experiments as E, plots, data   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=130)
    plt.close(fig)
    print(f"  saved figures/{name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--replot", action="store_true",
                    help="only redraw the aggregated figures from results/spike_results.json")
    args = ap.parse_args()
    if args.replot:
        with open(os.path.join(RES, "spike_results.json")) as f:
            out = json.load(f)
        for r in out["per_seed"]:   # JSON turned integer keys into strings
            r["e2"] = {int(m): v for m, v in r["e2"].items()}
        aggregated_figures(out["per_seed"])
        return
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(os.path.join(RES, "checkpoints"), exist_ok=True)
    seeds = list(range(2 if args.quick else args.seeds))
    R = 40 if args.quick else 200
    R_id = 100 if args.quick else 500
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    out = {"config": {"seeds": seeds, "R": R, "R_id": R_id, "d3m": E.D3M_DEFAULTS,
                      "m_grid": E.M_GRID, "strengths": E.STRENGTHS, "alphas": E.ALPHAS,
                      "harm_drop": E.HARM_DROP, "n_train": 2000, "n_val": 2000,
                      "noise_w": data.NOISE_W, "torch_threads": torch.get_num_threads()},
           "timing_secs": {}, "per_seed": []}
    t_all = time.perf_counter()
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        rs, tm = {"seed": seed}, {}

        t0 = time.perf_counter(); st = E.build_setup(seed); tm["train"] = time.perf_counter() - t0
        rs["_id_acc"] = st.id_acc
        if seed == 0:
            torch.save(st.model.state_dict(), os.path.join(RES, "checkpoints", "model_seed0.pt"))
        mon = E.make_monitor(st)

        steps = [
            ("e1", lambda: E.e1_sanity(st)),
            ("e2", lambda: E.e2_no_shift(st, mon, R=R_id)),
            ("e3", lambda: E.e3_shift_strength(st, mon, R=R)),
            ("e4", lambda: E.e4_batch_size(st, mon, R=R)),
            ("e5", lambda: E.e5_sensitivity(st, R=R)),
            ("e5_reg", lambda: E.e5_reg_factor(st, R=R)),
            ("e7", lambda: {n: E.e7_timeline(st, mon, n) for n in E.TIMELINES}),
        ]
        if seed == 0:
            steps.insert(0, ("e6", lambda: E.e6_timing(st, plots.panel_a)))
        for name, fn in steps:
            t0 = time.perf_counter()
            rs[name] = fn()
            tm[name] = time.perf_counter() - t0
            print(f"  {name}: {tm[name]:.1f}s", flush=True)
        rs["timing_secs"] = tm
        out["per_seed"].append(rs)

        if seed == 0:   # single-seed illustrative figures
            save(plots.fig_e1(st), "e1_disagreement_map.png")
            for kind, s in [("harmful_boundary", 0.5), ("harmful_extrapolation", 0.75),
                            ("harmless_easier", 0.75), ("harmless_luck", 1.0)]:
                save(plots.panel_a(st, mon, kind, s, 100, 0.1, seed=1), f"panelA_{kind}.png")
            save(plots.fig_e7(rs["e7"], st.id_acc), "e7_timeline.png")

    out["timing_secs"]["total"] = time.perf_counter() - t_all
    aggregated_figures(out["per_seed"])
    with open(os.path.join(RES, "spike_results.json"), "w") as f:
        json.dump(jsonable(out), f, indent=1)
    print(f"TOTAL {out['timing_secs']['total']:.1f}s", flush=True)


def aggregated_figures(P):
    """Figures that average over seeds (E2-E5)."""
    save(plots.fig_e2([r["e2"] for r in P], E.M_GRID, E.ALPHAS), "e2_fpr_no_shift.png")
    e3 = [{**r["e3"], "_id_acc": r["_id_acc"]} for r in P]
    kinds = data.SHIFT_KINDS[1:]
    save(plots.fig_rates(e3, kinds, E.STRENGTHS, lambda k, s: f"{k}|{s}", "shift strength s",
                         "E3: alarm rate vs shift strength (m = 100, alpha = 0.1; mean over seeds)", alpha=0.1),
         "e3_shift_strength.png")
    e4 = [{**r["e4"], "_id_acc": r["_id_acc"]} for r in P]
    save(plots.fig_rates(e4, data.SHIFT_KINDS, E.M_GRID, lambda k, m: f"{k}|{m}", "batch size m",
                         "E4: alarm rate vs batch size (s = 0.75, alpha = 0.1; mean over seeds)",
                         alpha=0.1, logx=True), "e4_batch_size.png")
    save(plots.fig_e5([r["e5"] for r in P], E.E5_KINDS), "e5_sensitivity.png")


if __name__ == "__main__":
    main()
