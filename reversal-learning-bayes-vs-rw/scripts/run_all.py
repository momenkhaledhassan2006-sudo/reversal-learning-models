"""Run the full pipeline (stages 1-4).  ``python scripts/run_all.py --n-jobs 0`` uses all cores."""
import subprocess
import sys
import time
from pathlib import Path

from _common import base_parser

ap = base_parser(__doc__)
args = ap.parse_args()
here = Path(__file__).parent
common = ["--seed", str(args.seed), "--n-jobs", str(args.n_jobs), "--out", str(args.out)] + (["--quick"] if args.quick else [])
for script in ("01_parameter_recovery.py", "02_model_recovery.py", "03_fit_compare_ppc.py", "04_adaptation_benchmark.py"):
    print(f"\n{'=' * 80}\n{script}\n{'=' * 80}", flush=True)
    t = time.time()
    subprocess.run([sys.executable, str(here / script), *common], check=True)
    print(f"[{script} finished in {time.time() - t:.0f}s]", flush=True)
