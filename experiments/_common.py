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
    # print every parameter so results can be traced back to the paper
    print(f"=== {name} ===")
    for k, v in vars(cfg).items():
        print(f"  {k:12s} = {v}")
    print()


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    print(f"wrote results/{stem}.png|pdf")


def mean_std(runs):
    # runs: one array per seed -> (mean, std) over seeds
    a = np.stack(runs)
    return a.mean(0), a.std(0)


def band(ax, x, mean, std, label, **kw):
    # line with a +/- 1 std shaded band
    (ln,) = ax.plot(x, mean, "o-", ms=4, label=label, **kw)
    ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=ln.get_color(), lw=0)
    return ln
