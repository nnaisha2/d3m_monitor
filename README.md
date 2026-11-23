# d3m_monitor: does a label-free monitor alarm only when the model gets worse?

A small, CPU-only study and interactive demo of

> Nguyen, Shui, Giri, Arya, Verma, Razak, Krishnan. *Reliably Detecting Model Failures in Deployment Without Labels.* NeurIPS 2025. [arXiv:2506.05047](https://arxiv.org/abs/2506.05047) · [official code](https://github.com/teivng/d3m)

It was built for the MLOps slot of the Applied Deep Learning course (TH Ingolstadt).

**The idea in one chain.**
1. A deployed model meets shifted data and has no labels to check itself against.
2. Classic drift tests (a KS test on inputs, or BBSD on predicted probabilities) alarm on *every* change.
3. D3M (Disagreement-Driven Deterioration Monitoring) instead asks: "do plausible alternative models disagree with the deployed model on this batch more than they usually do?"
4. If yes, it raises an alarm. The goal is to alarm on harmful shifts and stay quiet on harmless ones.

This project re-implements D3M's Calibrate and Deploy stages in about 150 lines of PyTorch and NumPy, on top of a VBLL base model. It tests D3M on a 2-D problem whose true labelling rule is known, so every shift can be checked against the model's *true* accuracy. It compares D3M with a KS test, BBSD and a label-using oracle.

The verdict, results and critical assessment are in **[FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md)**.

## Setup (Windows PowerShell)

Keep the virtual environment **outside OneDrive**, because sync breaks venvs.

```powershell
py -3.12 -m venv C:\Users\aisha\venvs\d3m
C:\Users\aisha\venvs\d3m\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ipykernel install --user --name d3m --display-name "Python (d3m)"
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. On Colab, `pip install vbll` is enough, because torch, numpy, scipy, matplotlib and ipywidgets are preinstalled.

## How to run

All commands are run from the `mlops/d3m_monitor` folder with the venv active (`cd mlops\d3m_monitor` from the repo root).

| What | Command | Time on the test laptop* |
|---|---|---|
| Unit tests | `python -m pytest -q tests` | 22–35 s |
| All experiments E1–E7 (5 seeds) | `python run_spike.py` | 21.9 min (measured) |
| Quick smoke run (2 seeds, small R) | `python run_spike.py --quick` (overwrites `results/`) | about 4 min |
| Redraw the averaged figures from the saved JSON | `python run_spike.py --replot` | a few seconds |
| Tables used in the report | `python summarize_results.py > results/tables.md` | < 1 s |
| Interactive demo | `jupyter lab demo.ipynb` (kernel "Python (d3m)") | start-up about 4 s; each Panel A interaction 1.1–1.3 s; Panel B about 20 s; the whole notebook 65 s |

\*Intel i5-7200U (2 cores, 2017 laptop), PyTorch CPU build, 2 threads.

`run_spike.py` writes `results/spike_results.json` (all numbers), `results/figures/*.png` and `results/checkpoints/model_seed0.pt`. The checkpoint is git-ignored; the notebook loads it if it exists and otherwise trains the model in about 15–60 s.

## Project structure

```
d3m_monitor/
├── FEASIBILITY_REPORT.md   verdict, corrections, results, Q&A preparation
├── README.md
├── demo.ipynb              Panels A (live harmful vs harmless), B (batch size), C (timeline)
├── run_spike.py            runs E1–E7, times them, saves JSON + figures
├── summarize_results.py    turns the JSON into the Markdown tables of the report
├── requirements.txt
├── src/
│   ├── data.py             2-D problem, known labelling rule, 5 shift types
│   ├── model_vbll.py       MLP feature extractor + vbll.DiscClassification (as in the official code)
│   ├── d3m.py              D3M Calibrate/Deploy, written from scratch (literal + exact engine)
│   ├── baselines.py        KS (Bonferroni), BBSD, oracle
│   ├── experiments.py      E1–E7 and the notebook helpers
│   └── plots.py            all figures
├── tests/test_d3m.py       shapes, exactness of the fast engine, FPR ≈ alpha, determinism
└── results/
    ├── spike_results.json, tables.md, run_spike.log
    ├── figures/            all plots used in the report
    └── checkpoints/        (git-ignored) trained model for the notebook
```

## What is computed by hand and what comes from a library

- **From a library:** the VBLL layer (`vbll.DiscClassification`, the same package and settings as the official code), the KS test (`scipy.stats.ks_2samp`), and PyTorch and NumPy basics.
- **Written here:** the D3M algorithm itself. That covers pseudo-labels, posterior logit draws, the temperature-scaled softmax, categorical label sampling, disagreement rates, the max over K, bootstrap calibration of Φ and the quantile test. It also covers the exact fast engine (Poisson-binomial distribution and max-of-K sampling), the BBSD wrapper, the data generator and all experiments.

## AI use statement

*(Placeholder: to be completed by the author.)* Describe which parts were produced with AI assistance (e.g. code drafting, reading the official code, report drafting), what you checked or changed yourself, and how you verified the results.
