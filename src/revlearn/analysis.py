"""Behavioural summary statistics, reversal-aligned curves and posterior-predictive checks."""
from __future__ import annotations

import numpy as np

from .task import Schedule

CURVE_PRE, CURVE_POST = 5, 20          # trials shown before / after a reversal
CURVE_X = np.arange(-CURVE_PRE, CURVE_POST)
STAT_NAMES = ("acc", "early_acc", "late_acc", "win_stay", "lose_shift", "switch_rate")
STAT_LABELS = {
    "acc": "P(choose better option)",
    "early_acc": "Accuracy, trials 1-5 after reversal",
    "late_acc": "Accuracy, trials 16-25 after reversal",
    "win_stay": "P(stay | rewarded)",
    "lose_shift": "P(shift | unrewarded)",
    "switch_rate": "P(switch)",
}


def behavior_stats(choices, rewards, sched: Schedule) -> dict:
    c, r = np.asarray(choices), np.asarray(rewards)
    correct = (c == sched.best).astype(float)
    tib = sched.trial_in_block
    rev = sched.reversal_trials
    after_rev = np.zeros(len(c), bool)
    if len(rev):
        after_rev[rev[0]:] = True                     # excludes the first (reversal-free) block
    stay = (c[1:] == c[:-1]).astype(float)
    prev_r = r[:-1]
    return {
        "acc": correct.mean(),
        "early_acc": correct[(tib < 5) & after_rev].mean() if np.any((tib < 5) & after_rev) else np.nan,
        "late_acc": correct[(tib >= 15) & (tib < 25)].mean(),
        "win_stay": stay[prev_r == 1].mean() if np.any(prev_r == 1) else np.nan,
        "lose_shift": 1.0 - stay[prev_r == 0].mean() if np.any(prev_r == 0) else np.nan,
        "switch_rate": 1.0 - stay.mean(),
    }


def reversal_curve(choices, sched: Schedule) -> np.ndarray:
    """P(choose the *new* better option) aligned to each reversal (x = CURVE_X)."""
    c = np.asarray(choices)
    T = len(c)
    hits = np.zeros(len(CURVE_X))
    counts = np.zeros(len(CURVE_X))
    for t_r in sched.reversal_trials:
        new_best = sched.best[t_r]
        for j, k in enumerate(CURVE_X):
            idx = t_r + k
            if 0 <= idx < T:
                hits[j] += float(c[idx] == new_best)
                counts[j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return hits / counts


def posterior_predictive(model, samples: np.ndarray, sched: Schedule, rng: np.random.Generator,
                         n_draws: int = 100) -> dict:
    """Simulate ``n_draws`` replicated sessions, each with a fresh posterior draw of theta.

    The synthetic agent runs "free" in the same task schedule (it generates its own
    choices and receives its own, newly sampled, rewards).
    """
    idx = rng.integers(0, len(samples), n_draws)
    stats = {s: np.empty(n_draws) for s in STAT_NAMES}
    curves = np.empty((n_draws, len(CURVE_X)))
    for d, j in enumerate(idx):
        c, r, _ = model.simulate(samples[j], sched, rng)
        st = behavior_stats(c, r, sched)
        for s in STAT_NAMES:
            stats[s][d] = st[s]
        curves[d] = reversal_curve(c, sched)
    return {"stats": stats, "curves": curves}


def mid_p(rep: np.ndarray, obs: float) -> float:
    """Posterior-predictive p-value P(T_rep >= T_obs), with mid-p handling of ties."""
    rep = rep[~np.isnan(rep)]
    return float(np.mean(rep > obs) + 0.5 * np.mean(rep == obs))


def is_extreme(p: float, tail: float = 0.025) -> bool:
    return p < tail or p > 1 - tail
