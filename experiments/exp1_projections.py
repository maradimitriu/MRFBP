"""Experiment 1 -- reconstruction quality vs number of projections (paper Figs. 7, 8).

Compares MR-FBP against FBP with three static filters and SIRT, over a range of
projection counts, on all three phantom families.

Results are averaged over independent draws of the phantom generator (--seeds);
shaded bands are +/- 1 standard deviation across seeds.

    python experiments/exp1_projections.py                        # quick, one seed
    python experiments/exp1_projections.py --seeds 0 1 2 3 4      # with error bands
    python experiments/exp1_projections.py --n 1024 --seeds 0 --methods fbp-ram-lak fbp-hann sirt-200 mrfbp
"""
import argparse

from _common import RESULTS, band, banner, mean_std, np, plt, save

from src.methods import LABELS, METHODS, reconstruct
from src.metrics import mae, residual, ssim
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256, help="reconstruction grid is n x n")
P.add_argument("--n-det", type=int, default=None, help="detectors (default: n)")
P.add_argument("--angles", type=int, nargs="+",
               default=[16, 24, 32, 48, 64, 96, 128, 192, 256])
P.add_argument("--phantoms", nargs="+", default=["ellipses", "blocks", "mixed"])
P.add_argument("--methods", nargs="+", default=METHODS)
P.add_argument("--seeds", type=int, nargs="+", default=[0])
P.add_argument("--oversample", type=int, default=4, help="anti-inverse-crime factor")
P.add_argument("--show-at", type=int, default=32, help="n_angles for the image grid")
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
cfg.n_det = cfg.n_det or cfg.n
banner("exp1: quality vs number of projections", cfg)

# runs[metric][method][phantom] = list over seeds of arrays over angles
runs = {k: {m: {ph: [] for ph in cfg.phantoms} for m in cfg.methods}
        for k in ("mae", "ssim", "res")}
images = {}

for seed in cfg.seeds:
    for ph in cfg.phantoms:
        curve = {k: {m: [] for m in cfg.methods} for k in runs}
        for na in cfg.angles:
            geom, p, gt = simulate(ph, cfg.n, cfg.n_det, na, seed=seed,
                                   oversample=cfg.oversample, use_gpu=not cfg.cpu)
            for m in cfg.methods:
                x = reconstruct(m, geom, p)
                curve["mae"][m].append(mae(x, gt))
                curve["ssim"][m].append(ssim(x, gt))
                curve["res"][m].append(residual(geom, x, p))
                if na == cfg.show_at and seed == cfg.seeds[0]:
                    images[(ph, m)] = x
            print(f"  seed={seed} {ph:9s} N_theta={na:4d}  " +
                  "  ".join(f"{LABELS[m]}:{curve['mae'][m][-1]:.4f}" for m in cfg.methods))
        for k in runs:
            for m in cfg.methods:
                runs[k][m][ph].append(np.array(curve[k][m]))
        if seed == cfg.seeds[0]:
            images[(ph, "gt")] = gt

stats = {k: {m: {ph: mean_std(runs[k][m][ph]) for ph in cfg.phantoms} for m in cfg.methods}
         for k in runs}
np.savez(RESULTS / "exp1_projections.npz", angles=cfg.angles, seeds=cfg.seeds,
         phantoms=cfg.phantoms, methods=cfg.methods,
         **{f"{k}|{m}|{ph}|{s}": v
            for k in runs for m in cfg.methods for ph in cfg.phantoms
            for s, v in zip(("mean", "std"), stats[k][m][ph])})

for key, ylab, logy in [("mae", "mean absolute error (Eq. 22)", True),
                        ("ssim", "SSIM index", False)]:
    fig, axes = plt.subplots(1, len(cfg.phantoms),
                             figsize=(5 * len(cfg.phantoms), 3.8))
    axes = np.atleast_1d(axes)
    for ax, ph in zip(axes, cfg.phantoms):
        for m in cfg.methods:
            band(ax, cfg.angles, *stats[key][m][ph], LABELS[m])
        ax.set_xlabel(r"number of projection angles $N_\theta$")
        ax.set_ylabel(ylab); ax.set_title(ph); ax.grid(alpha=.3)
        if logy:
            ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"mean over {len(cfg.seeds)} seed(s); shading = $\\pm$1 std", fontsize=9)
    save(fig, f"exp1_{key}")

cols = ["gt"] + [m for m in ("fbp-ram-lak", "sirt-200", "mrfbp") if m in cfg.methods]
fig, ax = plt.subplots(len(cfg.phantoms), len(cols),
                       figsize=(3 * len(cols), 3 * len(cfg.phantoms)))
ax = np.atleast_2d(ax)
for i, ph in enumerate(cfg.phantoms):
    for j, c in enumerate(cols):
        ax[i, j].imshow(images[(ph, c)], cmap="gray")
        ax[i, j].set_title("ground truth" if c == "gt" else LABELS[c], fontsize=9)
        ax[i, j].axis("off")
    ax[i, 0].text(-0.06, 0.5, ph, transform=ax[i, 0].transAxes,
                  rotation=90, va="center", ha="center", fontsize=10)
fig.suptitle(rf"Reconstructions from $N_\theta$ = {cfg.show_at} projections "
             rf"(seed {cfg.seeds[0]})", y=1.0)
save(fig, "exp1_images")
