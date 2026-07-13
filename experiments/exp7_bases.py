"""Experiment 7 (OWN CONTRIBUTION) -- does the filter basis matter?

Exponential binning is only ONE way to reduce the number of filter unknowns.
The paper's conclusion explicitly flags this as future work:

    "Other bases for reducing the number of unknowns in the linear system can be
     used however, which might enable us to reduce computation time even further,
     or improve reconstruction quality."

Here we compare, at MATCHED numbers of unknowns N_b:
    exponential : the paper's piecewise-constant exponential bins
    equidistant : piecewise-constant bins of equal width  (no exponential prior)
    gaussian    : Gaussian RBFs at the exponential bin centres (smooth)
    dct         : low-frequency cosine modes (band-limited rather than localised)

    python experiments/exp7_bases.py
"""
import argparse
import time

from _common import RESULTS, banner, np, plt, save

from src.bases import dct_basis, equidistant_basis, exponential_basis, gaussian_basis
from src.metrics import mae, residual, ssim
from src.mrfbp import mrfbp
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256)
P.add_argument("--angles", type=int, default=64)
P.add_argument("--phantom", default="ellipses")
P.add_argument("--n-b", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16, 24, 32],
               help="numbers of filter unknowns to compare across bases")
P.add_argument("--seed", type=int, default=0)
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp7: alternative filter bases", cfg)

geom, p, gt = simulate(cfg.phantom, cfg.n, cfg.n, cfg.angles, seed=cfg.seed,
                       oversample=cfg.oversample, use_gpu=not cfg.cpu)
nd = geom.n_det


def basis(name, nb):
    """Return a basis with (approximately) nb columns, or None if not achievable."""
    if name == "equidistant":
        return equidistant_basis(nd, nb)
    if name == "dct":
        return dct_basis(nd, nb)
    # The exponential/gaussian bases have their column count set by n_l, so we
    # search for the n_l whose basis has nb columns.
    for n_l in range(1, nd):
        Q = exponential_basis(nd, n_l) if name == "exponential" else gaussian_basis(nd, n_l)
        if Q.shape[1] == nb:
            return Q
        if Q.shape[1] > nb:
            return None
    return None


names = ["exponential", "equidistant", "gaussian", "dct"]
out = {n: {"nb": [], "mae": [], "ssim": [], "res": [], "time": []} for n in names}
for name in names:
    for nb in cfg.n_b:
        Q = basis(name, nb)
        if Q is None:
            continue
        t0 = time.perf_counter()
        x, _ = mrfbp(geom, p, Q=Q)
        dt = time.perf_counter() - t0
        out[name]["nb"].append(Q.shape[1])
        out[name]["mae"].append(mae(x, gt)); out[name]["ssim"].append(ssim(x, gt))
        out[name]["res"].append(residual(geom, x, p)); out[name]["time"].append(dt)
        print(f"  {name:12s} N_b={Q.shape[1]:3d}  MAE={out[name]['mae'][-1]:.4f}  "
              f"SSIM={out[name]['ssim'][-1]:.4f}  t={dt:.3f}s")

np.savez(RESULTS / "exp7_bases.npz",
         **{f"{n}|{k}": np.array(v[k]) for n, v in out.items() for k in v})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for name in names:
    a1.plot(out[name]["nb"], out[name]["mae"], "o-", ms=4, label=name)
    a2.plot(out[name]["nb"], out[name]["ssim"], "o-", ms=4, label=name)
a1.set_xlabel(r"number of filter unknowns $N_b$"); a1.set_ylabel("mean absolute error")
a2.set_xlabel(r"number of filter unknowns $N_b$"); a2.set_ylabel("SSIM index")
for a in (a1, a2):
    a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle(rf"Filter basis comparison at matched cost ({cfg.phantom}, "
             rf"$N_\theta$={cfg.angles}, $N_d$={nd})")
save(fig, "exp7_bases")
