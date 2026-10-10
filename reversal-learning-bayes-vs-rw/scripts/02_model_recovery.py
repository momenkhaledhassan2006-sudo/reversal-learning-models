"""Stage 2 -- model recovery (confusion matrix).

Simulate data from each model, fit *all* candidate models, and record which one wins by
BIC and by held-out log-likelihood.  If the comparison pipeline cannot identify the true
generating model on data where the answer is known, its verdict on real data is uninformative.
"""
import numpy as np
import pandas as pd

from _common import base_parser, out_dirs, save_json
from revlearn import MODELS, TaskConfig
from revlearn import plotting
from revlearn.workers import model_recovery_task, pmap, spawn_seeds

ap = base_parser(__doc__)
ap.add_argument("--n-agents", type=int, default=40, help="simulated agents per generating model")
ap.add_argument("--n-starts", type=int, default=6)
ap.add_argument("--k-folds", type=int, default=5)
args = ap.parse_args()
n = 5 if args.quick else args.n_agents
fig_dir, tab_dir = out_dirs(args.out)
cfg, names = TaskConfig(), list(MODELS)

seeds = spawn_seeds(args.seed + 1, len(names) * n)
tasks = [(g, seeds[i * n + j], cfg, names, args.n_starts, args.k_folds) for i, g in enumerate(names) for j in range(n)]
res = pmap(model_recovery_task, tasks, args.n_jobs)

rows = []
for r in res:
    row = {"generator": r["generator"]}
    for m in names:
        row.update({f"{k}_{m}": r[m][k] for k in ("ll", "aic", "bic", "cv_ll")})
    row["winner_bic"] = names[int(np.argmin([r[m]["bic"] for m in names]))]
    row["winner_aic"] = names[int(np.argmin([r[m]["aic"] for m in names]))]
    row["winner_cv"] = names[int(np.argmax([r[m]["cv_ll"] for m in names]))]
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(tab_dir / "model_recovery_raw.csv", index=False)

mats, summary = {}, {}
for crit, col in (("BIC", "winner_bic"), ("AIC", "winner_aic"), ("held-out LL", "winner_cv")):
    M = np.array([[np.mean(df[df.generator == g][col] == m) for m in names] for g in names])
    mats[crit] = M
    summary[crit] = {"confusion": {g: dict(zip(names, M[i].round(3))) for i, g in enumerate(names)},
                     "mean_diagonal": float(np.mean(np.diag(M)))}
    pd.DataFrame(M, index=names, columns=names).round(3).to_csv(tab_dir / f"confusion_{crit.replace(' ', '_')}.csv")
plotting.confusion_matrices(mats, names, names, fig_dir / "model_recovery_confusion.png")
save_json(summary, tab_dir / "stage2_summary.json")

for crit, M in mats.items():
    print(f"\n{crit}: P(best-fitting model | generating model); mean diagonal = {np.mean(np.diag(M)):.2f}")
    print(pd.DataFrame(M, index=[f"gen={g}" for g in names], columns=names).round(2).to_string())
