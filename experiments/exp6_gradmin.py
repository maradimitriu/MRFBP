"""Experiment 6 -- MR-FBP_GM: adding a gradient prior (paper Sec. V, Figs. 14, 15).

The `blocks` phantom is piecewise constant, so its gradient is sparse. MR-FBP_GM
adds a penalty on the Sobel gradients of the reconstruction to the SAME linear
system (no extra projections), which should pay off most when the data is noisy.

lambda follows the paper's heuristic: lam = 27 + 1600/I0.

    python experiments/exp6_gradmin.py
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
P.add_argument("--lam-a", type=float, default=27.0, help="lambda = lam_a + lam_b / I0")
P.add_argument("--lam-b", type=float, default=1600.0)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--oversample", type=int, default=4)
P.add_argument("--cpu", action="store_true")
cfg = P.parse_args()
banner("exp6: MR-FBP with gradient minimisation", cfg)

methods = ["fbp-ram-lak", "sirt-200", "mrfbp"]
geom, p_clean, gt = simulate(cfg.phantom, cfg.n, cfg.n, cfg.angles, seed=cfg.seed,
                             oversample=cfg.oversample, use_gpu=not cfg.cpu)

res = {m: {"mae": [], "ssim": []} for m in methods + ["mrfbp-gm"]}
images = {}
for i0 in cfg.i0:
    p = add_poisson_noise(p_clean, i0, seed=cfg.seed)
    lam = cfg.lam_a + cfg.lam_b / i0
    for m in methods:
        x = reconstruct(m, geom, p)
        res[m]["mae"].append(mae(x, gt)); res[m]["ssim"].append(ssim(x, gt))
        if i0 == min(cfg.i0):
            images[m] = x
    x, _ = mrfbp_gm(geom, p, lam=lam)
    res["mrfbp-gm"]["mae"].append(mae(x, gt)); res["mrfbp-gm"]["ssim"].append(ssim(x, gt))
    if i0 == min(cfg.i0):
        images["mrfbp-gm"] = x
    print(f"  I0={i0:>9.0f}  lam={lam:7.2f}  " +
          "  ".join(f"{LABELS[m]}:{res[m]['mae'][-1]:.4f}" for m in methods + ["mrfbp-gm"]))

np.savez(RESULTS / "exp6_gradmin.npz", i0=cfg.i0,
         **{f"{m}|{k}": np.array(v[k]) for m, v in res.items() for k in v})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
for m in methods + ["mrfbp-gm"]:
    a1.plot(cfg.i0, res[m]["mae"], "o-", ms=4, label=LABELS[m])
    a2.plot(cfg.i0, res[m]["ssim"], "o-", ms=4, label=LABELS[m])
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel(r"photon count $I_0$"); a1.set_ylabel("mean absolute error")
a2.set_xscale("log"); a2.set_xlabel(r"photon count $I_0$"); a2.set_ylabel("SSIM index")
for a in (a1, a2):
    a.grid(alpha=.3); a.legend(fontsize=8)
fig.suptitle(rf"Gradient prior on the {cfg.phantom} phantom, $N_\theta$={cfg.angles}")
save(fig, "exp6_gradmin")

fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
for a, k in zip(ax, ["gt"] + methods + ["mrfbp-gm"]):
    a.imshow(gt if k == "gt" else images[k], cmap="gray")
    a.set_title("ground truth" if k == "gt" else LABELS[k], fontsize=10); a.axis("off")
fig.suptitle(rf"$I_0$ = {min(cfg.i0):.0f} (heaviest noise)")
save(fig, "exp6_images")
