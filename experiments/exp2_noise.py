"""Experiment 2 -- robustness to Poisson noise (paper Figs. 10, 11).

Fixed projection count; sweep the photon count I0 (lower I0 = more noise).

    python experiments/exp2_noise.py
"""
import argparse

from _common import RESULTS, banner, np, plt, save

from src.methods import LABELS, reconstruct
from src.metrics import mae, ssim
from src.noise import add_poisson_noise
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256)
P.add_argument("--n-det", type=int, default=None)
P.add_argument("--angles", type=int, default=64, help="fixed number of projections")
P.add_argument("--phantom", default="ellipses")
P.add_argument("--i0", type=float, nargs="+",
               default=[2 ** k for k in (6, 8, 10, 12, 14, 16, 18, 20)])
P.add_argument("--mu-max", type=float, default=3.0, help="max line integral in the noise model")
P.add_argument("--seed", type=int, default=0)
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
cfg.n_det = cfg.n_det or cfg.n
banner("exp2: Poisson noise", cfg)

methods = ["fbp-ram-lak", "fbp-hann", "sirt-200", "mrfbp"]
geom, p_clean, gt = simulate(cfg.phantom, cfg.n, cfg.n_det, cfg.angles,
                             seed=cfg.seed, oversample=cfg.oversample, use_gpu=not cfg.cpu)

res = {m: {"mae": [], "ssim": []} for m in methods}
images = {}
for i0 in cfg.i0:
    p = add_poisson_noise(p_clean, i0, seed=cfg.seed, mu_max=cfg.mu_max)
    for m in methods:
        x = reconstruct(m, geom, p)
        res[m]["mae"].append(mae(x, gt))
        res[m]["ssim"].append(ssim(x, gt))
        if i0 in (min(cfg.i0), 2 ** 10):
            images[(i0, m)] = x
    print(f"  I0={i0:>9.0f}  " + "  ".join(f"{LABELS[m]}:{res[m]['mae'][-1]:.4f}" for m in methods))

np.savez(RESULTS / "exp2_noise.npz", i0=cfg.i0, methods=methods,
         **{f"{m}|{k}": np.array(v[k]) for m, v in res.items() for k in v})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for m in methods:
    a1.plot(cfg.i0, res[m]["mae"], "o-", ms=4, label=LABELS[m])
    a2.plot(cfg.i0, res[m]["ssim"], "o-", ms=4, label=LABELS[m])
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel(r"photon count $I_0$ (lower = noisier)"); a1.set_ylabel("mean absolute error")
a2.set_xscale("log")
a2.set_xlabel(r"photon count $I_0$ (lower = noisier)"); a2.set_ylabel("SSIM index")
for a in (a1, a2):
    a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle(rf"Noise robustness, {cfg.phantom} phantom, $N_\theta$ = {cfg.angles}")
save(fig, "exp2_noise")

keys = sorted({k[0] for k in images})
fig, ax = plt.subplots(len(keys), len(methods), figsize=(3 * len(methods), 3 * len(keys)))
ax = np.atleast_2d(ax)
for i, i0 in enumerate(keys):
    for j, m in enumerate(methods):
        ax[i, j].imshow(images[(i0, m)], cmap="gray")
        ax[i, j].set_title(f"{LABELS[m]}, $I_0$={i0:.0f}", fontsize=9); ax[i, j].axis("off")
save(fig, "exp2_images")
