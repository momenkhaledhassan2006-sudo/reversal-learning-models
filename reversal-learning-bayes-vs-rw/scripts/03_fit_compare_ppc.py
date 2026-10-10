"""Stage 3 -- fit, compare and check the models on a cohort.

Default: simulate a labelled cohort (RW agents + Bayesian agents) so that the ground truth is
known.  Real data: pass ``--data path/to/trials.csv`` (see README for the format); the pipeline is
identical, only the ground-truth cross-tabulation is skipped.

Per subject and model:  MLE fit -> AIC/BIC -> 5-fold contiguous-block held-out log-likelihood ->
MCMC posterior -> posterior-predictive replications of six behavioural statistics and the
reversal-aligned learning curve.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from _common import ROOT, base_parser, out_dirs, save_json
from revlearn import MODELS, TaskConfig, make_schedule
from revlearn import plotting
from revlearn.analysis import STAT_NAMES, is_extreme, mid_p
from revlearn.io import SubjectData, load_csv, to_dataframe
from revlearn.workers import pmap, spawn_seeds, subject_task

ap = base_parser(__doc__)
ap.add_argument("--data", default=None, help="CSV of real/pre-simulated trials")
ap.add_argument("--p-high", type=float, default=0.8, help="reward probability of the better option (real data)")
ap.add_argument("--n-per-model", type=int, default=20, help="simulated cohort: agents per generating model")
ap.add_argument("--models", nargs="+", default=list(MODELS))
ap.add_argument("--n-starts", type=int, default=8)
ap.add_argument("--k-folds", type=int, default=5)
ap.add_argument("--n-warmup", type=int, default=600)
ap.add_argument("--n-samples", type=int, default=1000)
ap.add_argument("--n-draws", type=int, default=100, help="posterior-predictive replications per subject")
args = ap.parse_args()
fig_dir, tab_dir = out_dirs(args.out)
names = args.models

# ------------------------------------------------------------------ data
if args.data:
    subjects = load_csv(args.data, p_high=args.p_high)
else:
    n = 4 if args.quick else args.n_per_model
    cfg, rng = TaskConfig(), np.random.default_rng(args.seed + 2)
    subjects, truth = [], []
    for gen in ("RW", "Bayes"):
        for i in range(n):
            sched = make_schedule(cfg, rng)
            theta = MODELS[gen].sample_params(rng)
            c, r, _ = MODELS[gen].simulate(theta, sched, rng)
            subjects.append(SubjectData(f"{gen}_{i:02d}", c, r, sched, gen))
            truth.append({"subject": f"{gen}_{i:02d}", "generator": gen,
                          **dict(zip(MODELS[gen].param_names, theta))})
    (ROOT / "data").mkdir(exist_ok=True)
    to_dataframe(subjects).to_csv(ROOT / "data" / "simulated_cohort.csv", index=False)
    pd.DataFrame(truth).to_csv(ROOT / "data" / "simulated_cohort_true_params.csv", index=False)

has_ppc = all(s.schedule is not None for s in subjects)
n_draws = args.n_draws if has_ppc else 0
if args.quick:
    args.n_warmup, args.n_samples, n_draws = 150, 200, min(n_draws, 20)

seeds = spawn_seeds(args.seed + 3, len(subjects))
tasks = [(s, names, seeds[i], args.n_starts, args.k_folds, args.n_warmup, args.n_samples, n_draws)
         for i, s in enumerate(subjects)]
res = pmap(subject_task, tasks, args.n_jobs)

# ------------------------------------------------------------------ model comparison tables
rows = []
for r in res:
    row = {"subject": r["subject"], "true_model": r["true_model"]}
    aic = np.array([r["models"][m]["aic"] for m in names])
    w = np.exp(-(aic - aic.min()) / 2); w /= w.sum()
    for k, m in enumerate(names):
        d = r["models"][m]
        row.update({f"ll_{m}": d["ll"], f"aic_{m}": d["aic"], f"bic_{m}": d["bic"], f"cv_ll_{m}": d["cv_ll"],
                    f"akaike_w_{m}": w[k], f"params_{m}": np.round(d["params"], 4).tolist()})
    row["best_bic"] = names[int(np.argmin([r["models"][m]["bic"] for m in names]))]
    row["best_aic"] = names[int(np.argmin(aic))]
    row["best_cv"] = names[int(np.argmax([r["models"][m]["cv_ll"] for m in names]))]
    rows.append(row)
sub = pd.DataFrame(rows)
sub.to_csv(tab_dir / "subject_fits.csv", index=False)

n_trials = len(subjects[0].choices)
group = []
for m in names:
    group.append({"model": m, "k": MODELS[m].n_params, "sum_LL": sub[f"ll_{m}"].sum(), "sum_AIC": sub[f"aic_{m}"].sum(),
                  "sum_BIC": sub[f"bic_{m}"].sum(), "mean_heldout_LL_per_trial": sub[f"cv_ll_{m}"].mean(),
                  "mean_akaike_weight": sub[f"akaike_w_{m}"].mean(),
                  "n_best_BIC": int((sub.best_bic == m).sum()), "n_best_heldout": int((sub.best_cv == m).sum())})
gdf = pd.DataFrame(group)
gdf.to_csv(tab_dir / "model_comparison_group.csv", index=False)

summary = {"n_subjects": len(subjects), "n_trials": n_trials, "group": gdf.round(4).to_dict("records")}
if "RW" in names and "Bayes-fixedP" in names:
    dcv = sub["cv_ll_Bayes-fixedP"] - sub["cv_ll_RW"]
    dbic = sub["bic_RW"] - sub["bic_Bayes-fixedP"]
    summary["paired_RW_vs_BayesFixedP"] = {
        "wilcoxon_p_heldout": float(wilcoxon(dcv).pvalue) if np.any(dcv != 0) else 1.0,
        "n_subjects_bayes_better_heldout": int((dcv > 0).sum()),
        "median_delta_heldout_per_trial": float(np.median(dcv)), "median_delta_bic": float(np.median(dbic))}
    plotting.model_comparison(dbic.to_numpy(), dcv.to_numpy(), sub.subject, "RW vs Bayesian (fixed p), per subject",
                              fig_dir / "model_comparison_subjects.png")
if sub.true_model.notna().all():
    summary["ground_truth_identification"] = {}
    for crit in ("best_bic", "best_aic", "best_cv"):
        ct = pd.crosstab(sub.true_model, sub[crit])
        ct.to_csv(tab_dir / f"crosstab_true_vs_{crit}.csv")
        summary["ground_truth_identification"][crit] = ct.to_dict()
    # per-generating-model group comparison
    per_gen = {}
    for g, d in sub.groupby("true_model"):
        per_gen[g] = {m: {"sum_BIC": float(d[f"bic_{m}"].sum()), "mean_heldout_LL": float(d[f"cv_ll_{m}"].mean())}
                      for m in names}
    summary["per_generating_model"] = per_gen

# ------------------------------------------------------------------ posterior-predictive checks
if has_ppc and n_draws > 0:
    groups = ({g: [i for i, s in enumerate(subjects) if s.true_model == g] for g in sorted({s.true_model for s in subjects})}
              if all(s.true_model for s in subjects) else {"all": list(range(len(subjects)))})
    ppc_rows, ppc_summary = [], {}
    for label, idx in groups.items():
        obs = {s: float(np.nanmean([res[i]["obs_stats"][s] for i in idx])) for s in STAT_NAMES}
        obs_curve = np.nanmean([res[i]["obs_curve"] for i in idx], axis=0)
        rep_stats, rep_curves = {}, {}
        ppc_summary[label] = {}
        for m in names:
            rep_stats[m] = {s: np.nanmean([res[i]["models"][m]["ppc_stats"][s] for i in idx], axis=0) for s in STAT_NAMES}
            rep_curves[m] = np.nanmean([res[i]["models"][m]["ppc_curves"] for i in idx], axis=0)
            n_fail, fr_ext = 0, []
            for s in STAT_NAMES:
                ppp = mid_p(rep_stats[m][s], obs[s])
                ext_frac = float(np.mean([is_extreme(res[i]["models"][m]["ppc_p"][s]) for i in idx]))
                fr_ext.append(ext_frac)
                n_fail += int(is_extreme(ppp))
                lo, hi = np.nanpercentile(rep_stats[m][s], [2.5, 97.5])
                ppc_rows.append({"group": label, "model": m, "stat": s, "observed": obs[s],
                                 "replicated_mean": float(np.nanmean(rep_stats[m][s])), "rep_2.5": lo, "rep_97.5": hi,
                                 "group_ppp": ppp, "extreme_at_5pct": is_extreme(ppp), "frac_subjects_extreme": ext_frac})
            # RMSE between the observed and the mean replicated reversal curve
            rmse = float(np.sqrt(np.nanmean((np.nanmean(rep_curves[m], axis=0) - obs_curve) ** 2)))
            ppc_summary[label][m] = {"n_stats_failed_of_6": n_fail, "mean_frac_subject_stats_extreme": float(np.mean(fr_ext)),
                                     "reversal_curve_rmse": rmse,
                                     "mean_mcmc_accept": float(np.mean([res[i]["models"][m]["mcmc_accept"] for i in idx]))}
        plotting.reversal_ppc(obs_curve, rep_curves, f"Posterior-predictive check: {label}", fig_dir / f"ppc_reversal_curve_{label}.png")
        plotting.ppc_stats_panel(obs, rep_stats, f"Posterior-predictive check, summary statistics: {label}",
                                 fig_dir / f"ppc_statistics_{label}.png")
    pd.DataFrame(ppc_rows).to_csv(tab_dir / "ppc_group.csv", index=False)
    summary["ppc"] = ppc_summary

# parameter estimates vs truth for the simulated cohort
if not args.data:
    truth_df = pd.read_csv(ROOT / "data" / "simulated_cohort_true_params.csv")
    truth_df.to_csv(tab_dir / "cohort_true_params.csv", index=False)

save_json(summary, tab_dir / "stage3_summary.json")
print(gdf.round(3).to_string(index=False))
if "paired_RW_vs_BayesFixedP" in summary:
    print("\nPaired RW vs Bayes-fixedP:", summary["paired_RW_vs_BayesFixedP"])
for crit, ct in (summary.get("ground_truth_identification") or {}).items():
    print(f"\nTrue model x winner ({crit}):")
    print(pd.DataFrame(ct).fillna(0).astype(int).T.to_string())
if "ppc" in summary:
    print("\nPosterior-predictive failures (of 6 statistics, group-level, 95% two-sided):")
    for g, d in summary["ppc"].items():
        for m, v in d.items():
            print(f"  data={g:6s} model={m:13s} failed={v['n_stats_failed_of_6']}/6  curve RMSE={v['reversal_curve_rmse']:.3f}")
