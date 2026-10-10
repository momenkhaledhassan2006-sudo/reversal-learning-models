"""Probabilistic reversal-learning task environment.

Two options (0 and 1). On every trial one option is the "better" option and pays
off with probability ``p_high``; the other pays off with ``1 - p_high``. The
identity of the better option reverses at the end of every block; block lengths
are drawn uniformly from ``[block_min, block_max]`` so reversals are not
perfectly predictable from the trial counter alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    n_trials: int = 400
    p_high: float = 0.8
    block_min: int = 25
    block_max: int = 45


@dataclass
class Schedule:
    best: np.ndarray            # (T,) int   index of the better option on each trial
    reward_prob: np.ndarray     # (T, 2) float  P(reward | option) on each trial
    trial_in_block: np.ndarray  # (T,) int   0 on the first trial after a reversal

    @property
    def n_trials(self) -> int:
        return len(self.best)

    @property
    def reversal_trials(self) -> np.ndarray:
        """Indices of the first trial of every block after the first."""
        return np.flatnonzero(self.trial_in_block == 0)[1:]


def schedule_from_best(best: np.ndarray, p_high: float = 0.8) -> Schedule:
    """Build a Schedule from the per-trial index of the better option.

    A reversal is any trial on which ``best`` differs from the previous trial.
    Used both by the simulator and when loading empirical data.
    """
    best = np.asarray(best, dtype=int)
    T = len(best)
    tib = np.zeros(T, dtype=int)
    for t in range(1, T):
        tib[t] = 0 if best[t] != best[t - 1] else tib[t - 1] + 1
    reward_prob = np.where(best[:, None] == np.arange(2)[None, :], p_high, 1.0 - p_high)
    return Schedule(best=best, reward_prob=reward_prob, trial_in_block=tib)


def make_schedule(cfg: TaskConfig, rng: np.random.Generator) -> Schedule:
    best = np.empty(cfg.n_trials, dtype=int)
    t, current = 0, int(rng.integers(2))
    while t < cfg.n_trials:
        length = int(rng.integers(cfg.block_min, cfg.block_max + 1))
        end = min(t + length, cfg.n_trials)
        best[t:end] = current
        t, current = end, 1 - current
    return schedule_from_best(best, cfg.p_high)
