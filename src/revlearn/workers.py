"""Top-level (picklable) worker functions and a tiny parallel map helper."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .analysis import STAT_NAMES, behavior_stats, mid_p, posterior_predictive, reversal_curve
from .fitting import cross_validate, fit, sample_posterior
from .models import MODELS
from .task import TaskConfig, make_schedule


def pmap(fn, tasks, n_jobs: int = 1):
    """Order-preserving map; ``n_jobs <= 0`` means 'all cores minus one'."""
    if n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    if n_jobs == 1:
        return [fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(fn, tasks, chunksize=1))


def spawn_seeds(seed: int, n: int):
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]


# --------------------------------------------------------------------------- stage 1
def parameter_recovery_task(args):
    model_name, seed, cfg, n_starts = args
    rng = np.random.default_rng(seed)
    model = MODELS[model_name]
    sched = make_schedule(cfg, rng)
    theta = model.sample_params(rng)
    c, r, _ = model.simulate(theta, sched, rng)
    res = fit(model, c, r, method="mle", n_starts=n_starts, rng=rng)
    true_ll = float(model.trial_loglik(theta, c, r).sum())
    return {"model": model_name, "true": theta, "est": res.params, "ll_fit": res.loglik, "ll_true": true_ll}


# --------------------------------------------------------------------------- stage 2
def model_recovery_task(args):
    gen_name, seed, cfg, model_names, n_starts, k_folds = args
    rng = np.random.default_rng(seed)
    gen = MODELS[gen_name]
    sched = make_schedule(cfg, rng)
    theta = gen.sample_params(rng)
    c, r, _ = gen.simulate(theta, sched, rng)
    out = {"generator": gen_name}
    for name in model_names:
        m = MODELS[name]
        res = fit(m, c, r, method="mle", n_starts=n_starts, rng=rng)
        out[name] = {"ll": res.loglik, "aic": res.aic, "bic": res.bic,
                     "cv_ll": cross_validate(m, c, r, k_folds=k_folds, n_starts=max(3, n_starts // 2), rng=rng)}
    return out


# --------------------------------------------------------------------------- stage 3
def subject_task(args):
    """Fit all models, cross-validate, sample posteriors and run posterior-predictive checks."""
    subj, model_names, seed, n_starts, k_folds, n_warmup, n_samples, n_draws = args
    rng = np.random.default_rng(seed)
    out = {"subject": subj.subject, "true_model": subj.true_model, "models": {}}
    if subj.schedule is not None:
        out["obs_stats"] = behavior_stats(subj.choices, subj.rewards, subj.schedule)
        out["obs_curve"] = reversal_curve(subj.choices, subj.schedule)
    for name in model_names:
        m = MODELS[name]
        res = fit(m, subj.choices, subj.rewards, method="mle", n_starts=n_starts, rng=rng)
        rec = {"params": res.params, "ll": res.loglik, "aic": res.aic, "bic": res.bic, "success": res.success,
               "cv_ll": cross_validate(m, subj.choices, subj.rewards, k_folds=k_folds,
                                       n_starts=max(3, n_starts // 2), rng=rng)}
        if subj.schedule is not None and n_draws > 0:
            samples, acc = sample_posterior(m, subj.choices, subj.rewards, res.params,
                                            n_warmup=n_warmup, n_samples=n_samples, rng=rng)
            ppc = posterior_predictive(m, samples, subj.schedule, rng, n_draws=n_draws)
            rec["post_mean"] = samples.mean(axis=0)
            rec["mcmc_accept"] = acc
            rec["ppc_stats"] = ppc["stats"]
            rec["ppc_curves"] = ppc["curves"]
            rec["ppc_p"] = {s: mid_p(ppc["stats"][s], out["obs_stats"][s]) for s in STAT_NAMES}
        out["models"][name] = rec
    return out
