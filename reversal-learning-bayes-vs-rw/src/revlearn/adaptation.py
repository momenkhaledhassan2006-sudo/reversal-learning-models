"""Speed-stability benchmark: how well can each learning rule do *in principle*?

For a grid of learning-rule parameters (alpha for RW, reversal hazard h for the Bayesian
observer with the true reward probability p = 0.8) and a grid of softmax inverse
temperatures, we simulate fresh sessions and measure

* acc        : P(choose better option), overall
* early_acc  : P(choose better option) in trials 1-5 after a reversal      (speed)
* late_acc   : P(choose better option) in trials 16-25 after a reversal    (stability)
* t50        : first post-reversal trial at which P(choose new better option) >= 0.5

For each learning-rule setting the inverse temperature is then chosen to maximise overall
accuracy, so each learning rule is compared at its best-case decision noise rather than
at an arbitrary one.  This gives a fair "frontier" for each rule.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .analysis import CURVE_X, behavior_stats, reversal_curve
from .models import MODELS
from .task import TaskConfig, make_schedule


def _t50(curve: np.ndarray) -> float:
    post = CURVE_X >= 0
    hit = np.flatnonzero(curve[post] >= 0.5)
    return float(CURVE_X[post][hit[0]] + 1) if len(hit) else np.nan   # 1 = first trial after reversal


def evaluate(model_name: str, par: float, beta: float, cfg: TaskConfig, n_sessions: int, seed: int):
    model = MODELS[model_name]
    theta = np.array([par, beta])
    stats, curves = [], []
    for i in range(n_sessions):
        sched = make_schedule(cfg, np.random.default_rng([seed, i]))   # identical schedules for every setting
        c, r, _ = model.simulate(theta, sched, np.random.default_rng([seed, i, 1]))
        stats.append(behavior_stats(c, r, sched))
        curves.append(reversal_curve(c, sched))
    out = {k: float(np.nanmean([s[k] for s in stats])) for k in ("acc", "early_acc", "late_acc")}
    out["t50"] = _t50(np.nanmean(curves, axis=0))
    return out


def sweep(cfg: TaskConfig, n_sessions: int = 100, seed: int = 7,
          alphas=(0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0),
          hazards=(0.005, 0.01, 0.02, 0.035, 0.05, 0.1, 0.2),
          betas=(2, 4, 6, 10, 15, 20, 30)) -> pd.DataFrame:
    rows = []
    for name, grid in (("RW", alphas), ("Bayes-fixedP", hazards)):
        for par, beta in itertools.product(grid, betas):
            rows.append({"model": name, "param": par, "beta": beta,
                         **evaluate(name, par, beta, cfg, n_sessions, seed)})
    return pd.DataFrame(rows)


def best_beta_frontier(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["model", "param"])["acc"].idxmax()
    return df.loc[idx].reset_index(drop=True)
