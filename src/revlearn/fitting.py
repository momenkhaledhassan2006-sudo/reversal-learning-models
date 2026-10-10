"""Parameter estimation: multi-start MLE / MAP, block cross-validation, and MCMC.

* ``fit``            -- maximum-likelihood (default) or MAP estimate via L-BFGS-B with
                        multiple random starts (likelihood surfaces are multimodal
                        near the parameter bounds).
* ``cross_validate`` -- K-fold CV over *contiguous trial blocks*. Parameters are fit on
                        the training trials only; the held-out log-likelihood is the
                        one-step-ahead predictive log-probability of the test-block
                        choices (the learner's latent state is filtered through all
                        earlier trials, exactly as a real agent's would be).
* ``sample_posterior``-- adaptive random-walk Metropolis in a logit-transformed
                        parameter space, used for posterior-predictive checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass
class FitResult:
    model: str
    params: np.ndarray
    loglik: float          # log-likelihood of ALL trials at ``params``
    log_post: float        # objective value that was maximised (train mask only)
    n_trials: int
    k: int
    success: bool
    param_names: tuple = field(default_factory=tuple)

    @property
    def aic(self) -> float:
        return 2 * self.k - 2 * self.loglik

    @property
    def bic(self) -> float:
        return self.k * np.log(self.n_trials) - 2 * self.loglik


def _objective(theta, model, choices, rewards, mask, use_prior):
    ll = model.trial_loglik(theta, choices, rewards)
    total = ll.sum() if mask is None else ll[mask].sum()
    if use_prior:
        total += model.log_prior(theta)
    return -total


def fit(model, choices, rewards, method: str = "mle", n_starts: int = 8,
        rng: np.random.Generator | None = None, mask: np.ndarray | None = None) -> FitResult:
    """Fit ``model`` to one agent's data.

    method : "mle" or "map".  ``mask`` (bool, length T) restricts the objective to the
    training trials (used by cross-validation); ``loglik`` in the result is always the
    full-sequence log-likelihood at the fitted parameters.
    """
    if method not in ("mle", "map"):
        raise ValueError("method must be 'mle' or 'map'")
    rng = np.random.default_rng() if rng is None else rng
    choices, rewards = np.asarray(choices, int), np.asarray(rewards, int)
    best_x, best_f, ok = None, np.inf, False
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in model.start_box])
        res = minimize(_objective, x0, args=(model, choices, rewards, mask, method == "map"),
                       method="L-BFGS-B", bounds=model.bounds)
        if res.fun < best_f:
            best_x, best_f, ok = res.x, res.fun, bool(res.success)
    ll_all = float(model.trial_loglik(best_x, choices, rewards).sum())
    return FitResult(model.name, best_x, ll_all, -best_f, len(choices), model.n_params, ok,
                     tuple(model.param_names))


def contiguous_folds(n_trials: int, k: int):
    return np.array_split(np.arange(n_trials), k)


def cross_validate(model, choices, rewards, k_folds: int = 5, method: str = "mle",
                   n_starts: int = 5, rng: np.random.Generator | None = None) -> float:
    """Mean held-out log-likelihood per trial under contiguous K-fold CV."""
    rng = np.random.default_rng() if rng is None else rng
    T = len(choices)
    total = 0.0
    for test_idx in contiguous_folds(T, k_folds):
        train = np.ones(T, bool)
        train[test_idx] = False
        res = fit(model, choices, rewards, method=method, n_starts=n_starts, rng=rng, mask=train)
        ll = model.trial_loglik(res.params, choices, rewards)
        total += ll[test_idx].sum()
    return total / T


def sample_posterior(model, choices, rewards, theta0, n_warmup: int = 600, n_samples: int = 1000,
                     rng: np.random.Generator | None = None):
    """Adaptive random-walk Metropolis over the bounded parameter box.

    Parameters are mapped to R with a scaled logit; the posterior density in that space
    includes the Jacobian.  The proposal covariance is re-estimated from the chain at
    1/3, 2/3 and the end of warm-up.  Returns (samples [n_samples, k], accept_rate).
    """
    rng = np.random.default_rng() if rng is None else rng
    lo = np.array([b[0] for b in model.bounds])
    hi = np.array([b[1] for b in model.bounds])
    d = len(lo)

    def to_u(x):
        s = np.clip((x - lo) / (hi - lo), 1e-4, 1 - 1e-4)
        return np.log(s / (1 - s))

    def to_x(u):
        return lo + (hi - lo) / (1.0 + np.exp(-u))

    def log_post(u):
        x = to_x(u)
        s = (x - lo) / (hi - lo)
        ll = model.trial_loglik(x, choices, rewards).sum()
        return ll + model.log_prior(x) + float(np.sum(np.log(s) + np.log1p(-s)))

    u = to_u(np.asarray(theta0, float))
    lp = log_post(u)
    cov = np.eye(d) * 0.1
    scale = 2.38 ** 2 / d
    chain, n_acc = np.empty((n_warmup + n_samples, d)), 0
    adapt_at = {n_warmup // 3, 2 * n_warmup // 3, n_warmup}
    L = np.linalg.cholesky(scale * cov)
    acc_after_warmup = 0
    for i in range(n_warmup + n_samples):
        prop = u + L @ rng.standard_normal(d)
        lp_prop = log_post(prop)
        if np.log(rng.random()) < lp_prop - lp:
            u, lp = prop, lp_prop
            n_acc += 1
            if i >= n_warmup:
                acc_after_warmup += 1
        chain[i] = u
        if (i + 1) in adapt_at:
            seg = chain[max(0, (i + 1) // 2): i + 1]
            if len(seg) > 20:
                cov = np.cov(seg.T).reshape(d, d) + 1e-6 * np.eye(d)
                L = np.linalg.cholesky(scale * cov)
    samples = np.array([to_x(c) for c in chain[n_warmup:]])
    return samples, acc_after_warmup / max(n_samples, 1)
