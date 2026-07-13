"""Experiment 1 -- reconstruction quality vs number of projections (paper Figs. 7, 8).

Compares MR-FBP against FBP with three static filters and SIRT, over a range of
projection counts, on all three phantom families.

    python experiments/exp1_projections.py                  # quick (N=256)
    python experiments/exp1_projections.py --n 1024 --oversample 4   # paper scale
"""
import argparse

from _common import RESULTS, banner, np, plt, save

from src.methods import LABELS, METHODS, reconstruct
from src.metrics import mae, residual, ssim
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256, help="reconstruction grid is n x n")
P.add_argument("--n-det", type=int, default=None, help="detectors (default: n)")
P.add_argument("--angles", type=int, nargs="+",
               default=[16, 24, 32, 48, 64, 96, 128, 192, 256])
P.add_argument("--phantoms", nargs="+", default=["ellipses", "blocks", "mixed"])
P.add_argument("--seed", type=int, default=0)
P.add_argument("--oversample", type=int, default=4, help="anti-inverse-crime factor")
P.add_argument("--show-at", type=int, default=32, help="n_angles for the image grid")
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
cfg.n_det = cfg.n_det or cfg.n
banner("exp1: quality vs number of projections", cfg)

out = {m: {ph: {k: [] for k in ("mae", "ssim", "res")} for ph in cfg.phantoms}
       for m in METHODS}
images = {}

for ph in cfg.phantoms:
    for na in cfg.angles:
        geom, p, gt = simulate(ph, cfg.n, cfg.n_det, na, seed=cfg.seed,
                               oversample=cfg.oversample, use_gpu=not cfg.cpu)
        for m in METHODS:
            x = reconstruct(m, geom, p)
            out[m][ph]["mae"].append(mae(x, gt))
            out[m][ph]["ssim"].append(ssim(x, gt))
            out[m][ph]["res"].append(residual(geom, x, p))
            if na == cfg.show_at:
                images[(ph, m)] = x
        print(f"  {ph:9s} N_theta={na:4d}  " +
              "  ".join(f"{LABELS[m]}:{out[m][ph]['mae'][-1]:.4f}" for m in METHODS))
    images[(ph, "gt")] = gt

np.savez(RESULTS / "exp1_projections.npz",
         angles=cfg.angles, phantoms=cfg.phantoms, methods=METHODS,
         **{f"{m}|{ph}|{k}": np.array(v[k])
            for m, d in out.items() for ph, v in d.items() for k in v})

# --- curves ------------------------------------------------------------------
fig, axes = plt.subplots(1, len(cfg.phantoms), figsize=(5 * len(cfg.phantoms), 3.8))
axes = np.atleast_1d(axes)
for ax, ph in zip(axes, cfg.phantoms):
    for m in METHODS:
        ax.plot(cfg.angles, out[m][ph]["mae"], "o-", ms=4, label=LABELS[m])
    ax.set_xlabel(r"number of projection angles $N_\theta$")
    ax.set_ylabel("mean absolute error (Eq. 22)")
    ax.set_title(ph); ax.set_yscale("log"); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
save(fig, "exp1_mae")

fig, axes = plt.subplots(1, len(cfg.phantoms), figsize=(5 * len(cfg.phantoms), 3.8))
axes = np.atleast_1d(axes)
for ax, ph in zip(axes, cfg.phantoms):
    for m in METHODS:
        ax.plot(cfg.angles, out[m][ph]["ssim"], "o-", ms=4, label=LABELS[m])
    ax.set_xlabel(r"number of projection angles $N_\theta$")
    ax.set_ylabel("SSIM index")
    ax.set_title(ph); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
save(fig, "exp1_ssim")

# --- image grid at a fixed, low projection count ------------------------------
cols = ["gt", "fbp-ram-lak", "sirt-200", "mrfbp"]
fig, ax = plt.subplots(len(cfg.phantoms), len(cols),
                       figsize=(3 * len(cols), 3 * len(cfg.phantoms)))
ax = np.atleast_2d(ax)
for i, ph in enumerate(cfg.phantoms):
    for j, c in enumerate(cols):
        ax[i, j].imshow(images[(ph, c)], cmap="gray")
        ax[i, j].set_title(("ground truth" if c == "gt" else LABELS[c]), fontsize=9)
        ax[i, j].axis("off")
    ax[i, 0].text(-0.06, 0.5, ph, transform=ax[i, 0].transAxes,
                  rotation=90, va="center", ha="center", fontsize=10)
fig.suptitle(rf"Reconstructions from $N_\theta$ = {cfg.show_at} projections", y=1.0)
save(fig, "exp1_images")
