"""Matplotlib figures (all saved as PNG)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import CURVE_X, STAT_LABELS, STAT_NAMES

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10})
COLORS = {"RW": "#d95f02", "Bayes": "#1b9e77", "Bayes-fixedP": "#7570b3", "data": "black"}


def task_schematic(sched, choices_by_model: dict, path):
    T = sched.n_trials
    fig, axes = plt.subplots(1 + len(choices_by_model), 1, figsize=(9, 1.2 + 1.1 * len(choices_by_model)), sharex=True)
    axes[0].step(np.arange(T), sched.best, where="post", color="k")
    axes[0].set_yticks([0, 1]); axes[0].set_ylabel("better\noption")
    axes[0].set_title("Reversal-learning schedule and example simulated agents")
    for ax, (name, c) in zip(axes[1:], choices_by_model.items()):
        ax.scatter(np.arange(T), c, s=4, color=COLORS.get(name, "gray"))
        ax.set_yticks([0, 1]); ax.set_ylabel(name)
    axes[-1].set_xlabel("trial")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def parameter_recovery(model, true, est, path):
    k = len(model.param_names)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.2))
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        x, y = true[:, j], est[:, j]
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=12, alpha=0.7, color=COLORS.get(model.name, "gray"))
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"true {model.param_names[j]}"); ax.set_ylabel("recovered")
        ax.set_title(f"{model.param_names[j]}: r = {r:.2f}")
    fig.suptitle(f"Parameter recovery: {model.name}", y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def confusion_matrices(mats: dict, gen_names, model_names, path):
    fig, axes = plt.subplots(1, len(mats), figsize=(3.6 * len(mats), 3.3))
    axes = np.atleast_1d(axes)
    for ax, (title, M) in zip(axes, mats.items()):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(model_names))); ax.set_xticklabels(model_names, rotation=30, ha="right")
        ax.set_yticks(range(len(gen_names))); ax.set_yticklabels(gen_names)
        ax.set_xlabel("best-fitting model"); ax.set_ylabel("generating model"); ax.set_title(title)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > 0.55 else "black")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def reversal_ppc(obs_curve, rep_curves: dict, title, path):
    """obs_curve: (K,) group mean; rep_curves: model -> (n_draws, K) group means."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, C in rep_curves.items():
        lo, hi = np.nanpercentile(C, [2.5, 97.5], axis=0)
        ax.fill_between(CURVE_X, lo, hi, alpha=0.25, color=COLORS.get(name, "gray"), label=f"{name} 95% PPC band")
        ax.plot(CURVE_X, np.nanmean(C, axis=0), color=COLORS.get(name, "gray"), lw=1)
    ax.plot(CURVE_X, obs_curve, "k-o", ms=3, lw=1.3, label="data")
    ax.axvline(-0.5, color="gray", ls=":")
    ax.set_xlabel("trial relative to reversal"); ax.set_ylabel("P(choose new better option)")
    ax.set_ylim(0, 1); ax.set_title(title); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def ppc_stats_panel(obs: dict, rep: dict, title, path):
    """obs: stat -> group mean; rep: model -> stat -> (n_draws,) group means."""
    n = len(STAT_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 3.0))
    for ax, s in zip(axes, STAT_NAMES):
        for i, (name, d) in enumerate(rep.items()):
            lo, hi = np.nanpercentile(d[s], [2.5, 97.5])
            ax.plot([i, i], [lo, hi], color=COLORS.get(name, "gray"), lw=5, alpha=0.5, solid_capstyle="butt")
            ax.plot(i, np.nanmean(d[s]), "o", color=COLORS.get(name, "gray"), ms=4)
        ax.axhline(obs[s], color="k", lw=1.2)
        ax.set_xticks(range(len(rep))); ax.set_xticklabels(list(rep), rotation=40, ha="right", fontsize=7)
        ax.set_title(STAT_LABELS[s], fontsize=7)
    fig.suptitle(title, y=1.03, fontsize=9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def model_comparison(delta_bic: np.ndarray, delta_cv: np.ndarray, labels, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, d, lab in zip(axes, (delta_bic, delta_cv), ("BIC(RW) - BIC(Bayes-fixedP)", "held-out LL/trial: Bayes-fixedP - RW")):
        order = np.argsort(d)
        ax.bar(range(len(d)), d[order], color=[COLORS["Bayes-fixedP"] if v > 0 else COLORS["RW"] for v in d[order]])
        ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("subject (sorted)"); ax.set_ylabel(lab)
    fig.suptitle(title, y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def adaptation_frontier(frontier, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for name, g in frontier.groupby("model"):
        g = g.sort_values("param")
        axes[0].plot(g["early_acc"], g["late_acc"], "-o", ms=4, color=COLORS.get(name, "gray"), label=name)
        axes[1].plot(g["param"], g["acc"], "-o", ms=4, color=COLORS.get(name, "gray"), label=name)
    axes[0].set_xlabel("early accuracy (trials 1-5 after reversal)  [speed]")
    axes[0].set_ylabel("late accuracy (trials 16-25)  [stability]")
    axes[0].set_title("Speed-stability frontier (best-case softmax beta)"); axes[0].legend(frameon=False)
    axes[1].set_xscale("log"); axes[1].set_xlabel("alpha (RW) / hazard h (Bayes-fixedP)"); axes[1].set_ylabel("overall accuracy")
    axes[1].set_title("Overall accuracy vs learning-rule parameter")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
