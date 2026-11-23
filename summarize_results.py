"""Print the Markdown tables used in FEASIBILITY_REPORT.md from results/spike_results.json.

    python summarize_results.py > results/tables.md

Every cell is "mean ± sd over seeds" unless stated otherwise.
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "spike_results.json")))
P = R["per_seed"]
KINDS = ["none", "harmful_boundary", "harmful_extrapolation", "harmless_easier",
         "harmless_support", "harmless_luck"]
DETS = [("d3m", "D3M"), ("d3m_tiefix", "D3M tie-fix"), ("ks", "KS"), ("bbsd", "BBSD"), ("oracle", "oracle")]


def ms(vals, nd=2):
    v = np.array(vals, dtype=float)
    return f"{v.mean():.{nd}f} ± {v.std():.{nd}f}"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


cfg = R["config"]
print(f"Seeds: {cfg['seeds']}; R = {cfg['R']} query batches per condition (E2: {cfg['R_id']}); "
      f"D3M: {cfg['d3m']}; total wall-clock {R['timing_secs']['total'] / 60:.1f} min\n")

# ---------------- E1
print("### E1: base model and posterior (s = 1 for every shift)\n")
print(f"ID accuracy {ms([p['_id_acc'] for p in P], 3)}; "
      f"training time {ms([p['timing_secs']['train'] for p in P], 1)} s per model\n")
rows = []
for k in KINDS:
    c = [p["e1"]["conditions"][k] for p in P]
    rows.append([k, ms([x["acc"] for x in c], 3), ms([x["mean_abs_gap"] for x in c]),
                 ms([x["mean_gap_std"] for x in c]), ms([x["p_dis_sampled"] for x in c], 3),
                 ms([x["p_dis_argmax"] for x in c], 3)])
table(["condition", "true acc", "mean abs logit gap", "mean posterior sd of gap",
       "mean p_i (sampled labels, tau=1)", "mean p_i (argmax only)"], rows)

# ---------------- E2
print("### E2: false positive rate on fresh ID batches\n")
for a in ["0.1", "0.05"]:
    rows = []
    for m in map(str, cfg["m_grid"]):
        e = [p["e2"][m] for p in P]
        rows.append([m, ms([x[f"fpr_{a}"] for x in e]), ms([x[f"fpr_strict_{a}"] for x in e]),
                     ms([x[f"fpr_tiefix_{a}"] for x in e]),
                     ms([x[f"calib_mass_at_or_above_{a}"] for x in e]),
                     ms([x["distinct_phi_values"] for x in e], 1)])
    print(f"alpha = {a}\n")
    table(["m", "D3M official (>=)", "D3M with >", "D3M tie-fix", "share of Phi >= threshold",
           "distinct values in Phi"], rows)
e = [p["e2"]["100"] for p in P]
print("Baselines at m = 100: " + ", ".join(
    f"alpha {a}: KS {ms([x[f'ks_fpr_{a}'] for x in e])}, BBSD {ms([x[f'bbsd_fpr_{a}'] for x in e])}"
    for a in ["0.05", "0.1"]) + "\n")

# ---------------- E3
print("### E3: alarm rate vs shift strength (m = 100, alpha = 0.1)\n")
rows = []
for key in P[0]["e3"]:
    k, s = key.split("|")
    e = [p["e3"][key] for p in P]
    rows.append([k, s, ms([x["acc"] for x in e], 3)] + [ms([x[d] for x in e]) for d, _ in DETS])
table(["shift", "s", "true acc"] + [n for _, n in DETS], rows)

# ---------------- E4
print("### E4: alarm rate vs batch size m (s = 0.75, alpha = 0.1)\n")
for k in KINDS:
    rows = []
    for m in map(str, cfg["m_grid"]):
        e = [p["e4"][f"{k}|{m}"] for p in P]
        rows.append([m] + [ms([x[d] for x in e]) for d, _ in DETS])
    acc = ms([p["e4"][f"{k}|100"]["acc"] for p in P], 3)
    print(f"**{k}** (true acc {acc})\n")
    table(["m"] + [n for _, n in DETS], rows)

# ---------------- E5
print("### E5: D3M sensitivity (m = 100, alpha = 0.1, s = 0.75)\n")
E5K = ["none", "harmful_boundary", "harmful_extrapolation", "harmless_easier", "harmless_luck"]
for par in ["K", "temperature", "variant"]:
    rows = []
    for v in P[0]["e5"][par]:
        rows.append([v] + [ms([p["e5"][par][v][k]["d3m"] for p in P]) for k in E5K]
                    + [ms([p["e5"][par][v]["none"]["phi_mean"] for p in P], 3),
                       ms([p["e5"][par][v]["none"]["threshold"] for p in P], 3)])
    table([par] + E5K + ["mean phi (ID)", "threshold"], rows)
rows = [[T] + [ms([p["e5"]["T"][T][f] for p in P], 4) for f in ["thr_mean", "thr_std", "fpr_mean", "fpr_std"]]
        for T in P[0]["e5"]["T"]]
print("Threshold stability over 20 independent calibrations:\n")
table(["T", "threshold mean", "threshold sd", "ID FPR mean", "ID FPR sd"], rows)
rows = []
for f in P[0]["e5_reg"]:
    rows.append([f, ms([p["e5_reg"][f]["id_acc"] for p in P], 3)]
                + [ms([p["e5_reg"][f][k]["d3m"] for p in P]) for k in E5K])
print("VBLL regularization factor (KL weight = factor / n_train):\n")
table(["factor", "ID acc"] + E5K, rows)

# ---------------- E6
t = P[0]["e6"]
print("### E6: timing (seed 0)\n")
print(f"- train base model: {P[0]['timing_secs']['train']:.1f} s")
print("- calibrate Phi (T = 1000, K = 5000, exact engine): " +
      ", ".join(f"m={m}: {v:.2f} s" for m, v in t["calibrate_secs"].items()))
print(f"- one interaction at m = 500 (sample, D3M test, KS, BBSD, true accuracy): {t['interaction_compute_secs_m500']:.2f} s")
if "interaction_with_plot_secs_m500" in t:
    print(f"- same plus drawing Panel A: {t['interaction_with_plot_secs_m500']:.2f} s")
print(f"- literal engine: calibrate T = 50, m = 100: {t['literal_calibrate_T50_m100_secs']:.1f} s; "
      f"score one batch m = 500: {t['literal_score_m500_secs']:.2f} s")
print("- per-seed experiment times (s): " + "; ".join(
    f"seed {p['seed']}: " + ", ".join(f"{k} {v:.0f}" for k, v in p["timing_secs"].items()) for p in P) + "\n")

# ---------------- E7
print("### E7: timelines (60 days, m = 100, alpha = 0.1)\n")
for name in P[0]["e7"]:
    rows = []
    for d in ["d3m", "ks", "bbsd", "oracle"]:
        s = [p["e7"][name]["summary"][d] for p in P]
        delays = [x["delay"] for x in s if x["delay"] is not None]
        rows.append([d, ms([x["false_alarms"] for x in s], 1), s[0]["n_ok_days"],
                     (ms(delays, 1) if delays else "never") + f" ({len(delays)}/{len(s)} seeds fired)",
                     ms([x["hit_rate_after_onset"] for x in s])])
    print(f"**{name}** (onset day per seed: {[p['e7'][name]['onset'] for p in P]})\n")
    table(["detector", "false alarms", "ok days (seed 0)", "delay (days)", "alarm rate after onset"], rows)
