"""Shared helpers for the pipeline scripts (path setup, CLI args, output dirs)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))      # lets the scripts run without `pip install -e .`


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=2024, help="master RNG seed (default 2024)")
    p.add_argument("--n-jobs", type=int, default=1, help="worker processes; 0 = all cores minus one")
    p.add_argument("--out", type=Path, default=ROOT / "results", help="output directory")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test run (not for reporting)")
    return p


def out_dirs(out: Path):
    fig, tab = out / "figures", out / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    return fig, tab


def save_json(obj, path: Path):
    def default(o):
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=default))
