"""Experiment 6 -- MR-FBP_GM: adding a gradient prior (paper Sec. V, Figs. 14, 15).

The `blocks` phantom is piecewise constant, so its gradient is sparse. MR-FBP_GM
adds a penalty on the Sobel gradients of the reconstruction to the SAME linear
system (costing no extra projections), which should pay off most under noise.

OWN EXTENSION -- lambda is NOT scale-invariant. The paper uses the heuristic
lam = 27 + 1600/I0, "experimentally verified to be a reasonable choice". But lam
weights ||D W^T C_p h|| against ||p - W W^T C_p h||, and the relative magnitude of
those two terms depends on the image size, the detector count and the data
scaling. At N=256 the paper's constants over-regularise badly: MR-FBP_GM's error
saturates and stops responding to the data at all.

So we do two things:
  (a) sweep lambda at each noise level and find the true optimum, and
  (b) compare that optimum against the paper's heuristic.

    python experiments/exp6_gradmin.py
    python experiments/exp6_gradmin.py --no-sweep        # heuristic only (paper)
"""
import argparse

from _common import RESULTS, banner, np, plt, save

from src.methods import LABELS, reconstruct
from src.metrics import mae, ssim
from src.mrfbp import mrfbp_gm
from src.noise import add_poisson_noise
from src.pipeline import simulate

P = argparse.ArgumentParser()
P.add_argument("--n", type=int, default=256)
P.add_argument("--angles", type=int, default=64)
P.add_argument("--phantom", default="blocks", help="sparse-gradient phantom")
P.add_argument("--i0", type=float, nargs="+", default=[2 ** k for k in (6, 8, 10, 12, 14, 16)])
P.add_argument("--lam", type=float, nargs="+",
               default=[0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 27.0, 100.0],
               help="lambda values to sweep (0 = plain MR-FBP)")
P.add_argument("--lam-a", type=float, default=27.0, help="paper heuristic: lam = a + b / I0")
P.add_argument("--lam-b", type=float, default=1600.0)
P.add_argument("--no-sweep", action="store_true", help="use the paper heuristic only")
P.add_argument("--seed", type=int, default=0)
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp6: MR-FBP with gradient minimisation", cfg)

methods = ["fbp-ram-lak", "sirt-200", "mrfbp"]
geom, p_clean, gt = simulate(cfg.phantom, cfg.n, cfg.n, cfg.angles, seed=cfg.seed,
                             oversample=cfg.oversample, use_gpu=not cfg.cpu)

res = {m: {"mae": [], "ssim": []} for m in methods + ["mrfbp-gm", "mrfbp-gm-paper"]}
lam_grid = np.zeros((len(cfg.i0), len(cfg.lam)))   # MAE(I0, lambda)
lam_best, images = [], {}

for i, i0 in enumerate(cfg.i0):
    p = add_poisson_noise(p_clean, i0, seed=cfg.seed)

    for m in methods:
        x = reconstruct(m, geom, p)
        res[m]["mae"].append(mae(x, gt)); res[m]["ssim"].append(ssim(x, gt))
        if i0 == min(cfg.i0):
            images[m] = x

    # (a) sweep lambda -> find the optimum for this noise level
    if not cfg.no_sweep:
        best = (np.inf, None, None)
        for j, lam in enumerate(cfg.lam):
            x, _ = mrfbp_gm(geom, p, lam=lam)
            lam_grid[i, j] = mae(x, gt)
            if lam_grid[i, j] < best[0]:
                best = (lam_grid[i, j], lam, x)
        lam_best.append(best[1])
        res["mrfbp-gm"]["mae"].append(best[0])
        res["mrfbp-gm"]["ssim"].append(ssim(best[2], gt))
        if i0 == min(cfg.i0):
            images["mrfbp-gm"] = best[2]

    # (b) the paper's heuristic, for comparison
    lam_paper = cfg.lam_a + cfg.lam_b / i0
    x, _ = mrfbp_gm(geom, p, lam=lam_paper)
    res["mrfbp-gm-paper"]["mae"].append(mae(x, gt))
    res["mrfbp-gm-paper"]["ssim"].append(ssim(x, gt))

    print(f"  I0={i0:>9.0f}  " +
          "  ".join(f"{m}:{res[m]['mae'][-1]:.4f}" for m in methods) +
          (f"  GM(lam*={lam_best[-1]:g}):{res['mrfbp-gm']['mae'][-1]:.4f}" if not cfg.no_sweep else "") +
          f"  GM(paper lam={lam_paper:.1f}):{res['mrfbp-gm-paper']['mae'][-1]:.4f}")

np.savez(RESULTS / "exp6_gradmin.npz", i0=cfg.i0, lam=cfg.lam, lam_grid=lam_grid,
         lam_best=lam_best or [np.nan],
         **{f"{m}|{k}": np.array(v[k]) for m, v in res.items() for k in v if v[k]})

shown = methods + (["mrfbp-gm"] if not cfg.no_sweep else []) + ["mrfbp-gm-paper"]
lbl = dict(LABELS, **{"mrfbp-gm": r"MR-FBP$_{GM}$ ($\lambda^*$)",
                      "mrfbp-gm-paper": r"MR-FBP$_{GM}$ (paper $\lambda$)"})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for m in shown:
    a1.plot(cfg.i0, res[m]["mae"], "o-", ms=4, label=lbl[m])
    a2.plot(cfg.i0, res[m]["ssim"], "o-", ms=4, label=lbl[m])
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel(r"photon count $I_0$"); a1.set_ylabel("mean absolute error")
a2.set_xscale("log"); a2.set_xlabel(r"photon count $I_0$"); a2.set_ylabel("SSIM index")
for a in (a1, a2):
    a.grid(alpha=.3); a.legend(fontsize=7)
fig.suptitle(rf"Gradient prior on the {cfg.phantom} phantom, $N_\theta$={cfg.angles}")
save(fig, "exp6_gradmin")

if not cfg.no_sweep:
    # MAE as a function of lambda, one curve per noise level: shows the optimum
    # moving, and how far the paper's constant sits from it.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, i0 in enumerate(cfg.i0):
        ax.plot(cfg.lam, lam_grid[i], "o-", ms=3, label=rf"$I_0$={i0:.0f}")
    ax.axvline(cfg.lam_a, color="k", ls="--", lw=1,
               label=rf"paper $\lambda\to${cfg.lam_a:g} (low noise)")
    ax.set_xscale("symlog", linthresh=0.03); ax.set_yscale("log")
    ax.set_xlabel(r"gradient weight $\lambda$   ($\lambda=0$ is plain MR-FBP)")
    ax.set_ylabel("mean absolute error")
    ax.set_title(r"MR-FBP$_{GM}$: the optimal $\lambda$ depends on the noise level")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=7)
    save(fig, "exp6_lambda")

fig, ax = plt.subplots(1, len(images) + 1, figsize=(3.2 * (len(images) + 1), 3.4))
for a, k in zip(ax, ["gt"] + list(images)):
    a.imshow(gt if k == "gt" else images[k], cmap="gray")
    a.set_title("ground truth" if k == "gt" else lbl[k], fontsize=10); a.axis("off")
fig.suptitle(rf"$I_0$ = {min(cfg.i0):.0f} (heaviest noise)")
save(fig, "exp6_images")
