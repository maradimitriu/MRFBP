"""Experiment 4 -- exponential binning: quality and cost (paper Sec. VII-D)
   + OWN EXTENSION: sweep the binning parameter n_l.

The paper fixes n_l = 2 and reports a single with/without comparison. Here we
sweep n_l, which interpolates between coarse binning (few unknowns, fast) and
no binning at all (n_l >= N_d: one unknown per detector, slow), and record both
error and time as a function of the resulting number of unknowns N_b.

    python experiments/exp4_binning.py
"""
import argparse
import time

from _common import RESULTS, banner, np, plt, save

from src.bases import exponential_basis
from src.metrics import mae, residual, ssim
from src.mrfbp import mrfbp
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256)
P.add_argument("--angles", type=int, default=64)
P.add_argument("--n-l", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
P.add_argument("--phantom", default="ellipses")
P.add_argument("--seed", type=int, default=0)
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp4: exponential binning / n_l sweep", cfg)

geom, p, gt = simulate(cfg.phantom, cfg.n, cfg.n, cfg.angles, seed=cfg.seed,
                       oversample=cfg.oversample, use_gpu=not cfg.cpu)

rows = []
for n_l in cfg.n_l:
    nb = exponential_basis(geom.n_det, n_l=n_l).shape[1]
    t0 = time.perf_counter()
    x, _ = mrfbp(geom, p, n_l=n_l)
    dt = time.perf_counter() - t0
    rows.append((n_l, nb, mae(x, gt), ssim(x, gt), residual(geom, x, p), dt))
    print(f"  n_l={n_l:4d}  N_b={nb:4d}  MAE={rows[-1][2]:.4f}  "
          f"SSIM={rows[-1][3]:.4f}  t={dt:.3f}s")

r = np.array(rows)
np.savez(RESULTS / "exp4_binning.npz",
         n_l=r[:, 0], n_bins=r[:, 1], mae=r[:, 2], ssim=r[:, 3], res=r[:, 4], time=r[:, 5])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
a1.plot(r[:, 1], r[:, 2], "o-", color="tab:blue")
a1.set_xlabel(r"number of filter unknowns $N_b$"); a1.set_ylabel("mean absolute error")
a1.set_xscale("log")
a2.plot(r[:, 1], r[:, 5], "o-", color="tab:red")
a2.set_xlabel(r"number of filter unknowns $N_b$"); a2.set_ylabel("reconstruction time (s)")
a2.set_xscale("log"); a2.set_yscale("log")
for a in (a1, a2):
    a.grid(alpha=.3, which="both")
fig.suptitle(rf"Effect of exponential binning ({cfg.phantom}, $N_\theta$={cfg.angles}, "
             rf"$N_d$={geom.n_det})")
save(fig, "exp4_binning")
