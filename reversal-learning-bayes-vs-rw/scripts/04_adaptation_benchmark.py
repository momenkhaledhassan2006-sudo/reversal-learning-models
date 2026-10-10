"""Stage 4 -- speed/stability benchmark of the two learning rules (best-case decision noise)."""
from _common import base_parser, out_dirs, save_json
from revlearn import TaskConfig
from revlearn import plotting
from revlearn.adaptation import best_beta_frontier, sweep

ap = base_parser(__doc__)
ap.add_argument("--n-sessions", type=int, default=100)
args = ap.parse_args()
fig_dir, tab_dir = out_dirs(args.out)
df = sweep(TaskConfig(), n_sessions=20 if args.quick else args.n_sessions, seed=args.seed)
fr = best_beta_frontier(df)
df.to_csv(tab_dir / "adaptation_full_grid.csv", index=False)
fr.to_csv(tab_dir / "adaptation_frontier.csv", index=False)
plotting.adaptation_frontier(fr, fig_dir / "adaptation_frontier.png")
best = fr.loc[fr.groupby("model")["acc"].idxmax()]
save_json({r.model: {"best_param": r.param, "best_beta": r.beta, "acc": r.acc, "early_acc": r.early_acc,
                     "late_acc": r.late_acc, "t50": r.t50} for r in best.itertuples()}, tab_dir / "stage4_summary.json")
print(fr.round(3).to_string(index=False))
print("\nBest-case operating point per learning rule:")
print(best.round(3).to_string(index=False))
