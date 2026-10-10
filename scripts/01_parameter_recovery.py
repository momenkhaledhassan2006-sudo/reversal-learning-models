"""Stage 1 -- parameter recovery.

Simulate agents with known parameters, fit them with MLE, and ask: do the recovered
parameters match the generating ones?  A model whose parameters cannot be recovered
cannot be interpreted, however well it fits.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import base_parser, out_dirs, save_json
from revlearn import MODELS, TaskConfig, make_schedule
from revlearn import plotting
from revlearn.workers import parameter_recovery_task, pmap, spawn_seeds

ap = base_parser(__doc__)
ap.add_argument("--n-agents", type=int, default=100, help="simulated agents per model")
ap.add_argument("--n-starts", type=int, default=8, help="optimiser restarts per fit")
args = ap.parse_args()
n = 12 if args.quick else args.n_agents
fig_dir, tab_dir = out_dirs(args.out)
cfg = TaskConfig()
names = list(MODELS)

seeds = spawn_seeds(args.seed, len(names) * n)
tasks = [(m, seeds[i * n + j], cfg, args.n_starts) for i, m in enumerate(names) for j in range(n)]
res = pmap(parameter_recovery_task, tasks, args.n_jobs)

rows, summary = [], {}
for m in names:
    model = MODELS[m]
    R = [r for r in res if r["model"] == m]
    true, est = np.array([r["true"] for r in R]), np.array([r["est"] for r in R])
    plotting.parameter_recovery(model, true, est, fig_dir / f"parameter_recovery_{m}.png")
    summary[m] = {"n_agents": len(R), "params": {},
                  "optimiser_ok_frac": float(np.mean([r["ll_fit"] >= r["ll_true"] - 1e-6 for r in R]))}
    for j, pn in enumerate(model.param_names):
        lo, hi = model.bounds[j]
        at_bound = np.mean((est[:, j] <= lo * 1.02 + 1e-9) | (est[:, j] >= hi * 0.99))
        summary[m]["params"][pn] = {
            "pearson_r": float(np.corrcoef(true[:, j], est[:, j])[0, 1]),
            "spearman_rho": float(spearmanr(true[:, j], est[:, j])[0]),
            "bias": float(np.mean(est[:, j] - true[:, j])),
            "rmse": float(np.sqrt(np.mean((est[:, j] - true[:, j]) ** 2))),
            "frac_at_bound": float(at_bound),
        }
        for t, e in zip(true[:, j], est[:, j]):
            rows.append({"model": m, "param": pn, "true": t, "recovered": e})
    if m == "Bayes":      # identifiable composite: effective inverse temperature k = beta * (2p - 1)
        kt, ke = true[:, 2] * (2 * true[:, 1] - 1), est[:, 2] * (2 * est[:, 1] - 1)
        summary[m]["params"]["k=beta*(2p-1)"] = {"pearson_r": float(np.corrcoef(kt, ke)[0, 1]),
                                                 "spearman_rho": float(spearmanr(kt, ke)[0])}
pd.DataFrame(rows).to_csv(tab_dir / "parameter_recovery.csv", index=False)
save_json(summary, tab_dir / "stage1_summary.json")

# example agents (schematic)
rng = np.random.default_rng(args.seed)
sched = make_schedule(cfg, rng)
ex = {"RW": [0.35, 6.0], "Bayes": [0.04, 0.8, 8.0], "Bayes-fixedP": [0.04, 8.0]}
plotting.task_schematic(sched, {m: MODELS[m].simulate(np.array(th), sched, rng)[0][:] for m, th in ex.items()},
                        fig_dir / "task_schematic.png")

print(f"{'model':14s}{'param':16s}{'r':>7s}{'rho':>7s}{'bias':>8s}{'rmse':>8s}{'@bound':>8s}")
for m, s in summary.items():
    for pn, d in s["params"].items():
        print(f"{m:14s}{pn:16s}{d['pearson_r']:7.2f}{d['spearman_rho']:7.2f}{d.get('bias', float('nan')):8.2f}"
              f"{d.get('rmse', float('nan')):8.2f}{d.get('frac_at_bound', float('nan')):8.2f}")
    print(f"   optimiser reached >= true-parameter likelihood in {s['optimiser_ok_frac']:.0%} of fits")
