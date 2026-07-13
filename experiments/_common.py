"""Boilerplate shared by the experiment scripts."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402,F401
import numpy as np                # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def banner(name, cfg):
    """Print every parameter, so the paper can be linked to the code."""
    print(f"=== {name} ===")
    for k, v in vars(cfg).items():
        print(f"  {k:12s} = {v}")
    print()


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    print(f"wrote results/{stem}.png|pdf")
