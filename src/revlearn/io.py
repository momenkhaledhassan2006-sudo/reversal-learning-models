"""Data containers and CSV input/output.

Expected long-format CSV (one row per trial):

    subject, trial, choice, reward [, best_option] [, true_model]

* choice      : 0 / 1 (which of the two options was chosen)
* reward      : 0 / 1
* best_option : (optional) 0 / 1, the objectively better option on that trial.
                Needed only for posterior-predictive checks / reversal-aligned plots.
* true_model  : (optional) ground-truth generating model for simulated cohorts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .task import Schedule, schedule_from_best


@dataclass
class SubjectData:
    subject: str
    choices: np.ndarray
    rewards: np.ndarray
    schedule: Schedule | None = None
    true_model: str | None = None


def load_csv(path: str, p_high: float = 0.8) -> list[SubjectData]:
    df = pd.read_csv(path)
    need = {"subject", "trial", "choice", "reward"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    out = []
    for sid, g in df.sort_values(["subject", "trial"]).groupby("subject", sort=False):
        sched = schedule_from_best(g["best_option"].to_numpy(), p_high) if "best_option" in g else None
        tm = str(g["true_model"].iloc[0]) if "true_model" in g else None
        out.append(SubjectData(str(sid), g["choice"].to_numpy(int), g["reward"].to_numpy(int), sched, tm))
    return out


def to_dataframe(subjects: list[SubjectData]) -> pd.DataFrame:
    rows = []
    for s in subjects:
        T = len(s.choices)
        d = pd.DataFrame({"subject": s.subject, "trial": np.arange(T), "choice": s.choices, "reward": s.rewards})
        if s.schedule is not None:
            d["best_option"] = s.schedule.best
        if s.true_model is not None:
            d["true_model"] = s.true_model
        rows.append(d)
    return pd.concat(rows, ignore_index=True)
